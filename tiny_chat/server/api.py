import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field

from tiny_chat.config import get_config
from tiny_chat.messages import TinyChatBackground
from tiny_chat.utils import EpisodeLog

from .core import TinyChatServer, create_server

logger = logging.getLogger(__name__)

server_instance: TinyChatServer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    global server_instance

    logger.info('Starting TinyChat API Server...')

    async with create_server() as server:
        server_instance = server
        logger.info('TinyChat Server initialized')
        yield

    logger.info('TinyChat API Server shutting down...')


app = FastAPI(
    title='TinyChat API',
    description='Multi-agent conversation API powered by TinyChat',
    version='1.0.0',
    lifespan=lifespan,
)

config = get_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


def get_server() -> TinyChatServer:
    if server_instance is None:
        raise HTTPException(status_code=503, detail='Server not initialized')
    return server_instance


class AgentConfig(BaseModel):
    """Configuration for a single agent"""

    name: str = Field(description='Agent name')
    type: str = Field(default='llm', description='Agent type')
    goal: str | None = Field(default=None, description='Agent goal')
    profile: dict[str, Any] | None = Field(default=None, description='Agent profile')


class ConversationRequest(BaseModel):
    """Request to create a conversation"""

    agent_configs: list[AgentConfig] = Field(description='Agent configurations')
    background: dict[str, Any] | None = Field(
        default=None, description='Conversation background'
    )
    scenario: str | None = Field(default=None, description='Scenario description')
    max_turns: int | None = Field(
        default=None, description='Maximum conversation turns'
    )
    enable_evaluation: bool = Field(default=True, description='Enable evaluation')
    action_order: str | None = Field(
        default=None,
        description='Agent action order: simultaneous, round-robin, sequential, random, or agent_id_based',
    )
    speaking_order: list[int] | None = Field(
        default=None, description='List of agent IDs in speaking order'
    )
    model_name: str | None = Field(default=None, description='Model to use')
    return_log: bool = Field(default=False, description='Return detailed log')


class ConversationResponse(BaseModel):
    """Response from conversation creation"""

    conversation_id: str = Field(description='Unique conversation ID')
    status: str = Field(description='Conversation status')
    agent_count: int = Field(description='Number of agents')
    model_used: str = Field(description='Model used for the conversation')
    turn_count: int | None = Field(
        default=None, description='Number of turns completed'
    )
    evaluation_results: dict[str, Any] | None = Field(
        default=None, description='Evaluation results'
    )
    episode_log: EpisodeLog | None = Field(
        default=None, description='Detailed episode log'
    )


class HealthResponse(BaseModel):
    """Health check response"""

    status: str = Field(description='Service status')
    version: str = Field(description='API version')
    server_config: dict[str, Any] = Field(description='Server configuration summary')


class ModelInfo(BaseModel):
    """Model information"""

    name: str = Field(description='Model name')
    type: str = Field(description='Model provider type')
    healthy: bool = Field(description='Model health status')
    config: dict[str, Any] | None = Field(
        default=None, description='Model configuration'
    )


conversations: dict[str, dict[str, Any]] = {}


@app.get('/health', response_model=HealthResponse)
async def health_check(server: TinyChatServer = Depends(get_server)):
    """Health check endpoint"""
    config = get_config()

    return HealthResponse(
        status='healthy',
        version='1.0.0',
        server_config={
            'models': list(config.models.keys()),
            'default_model': config.default_model,
            'max_turns': config.max_turns,
            'evaluators': len(config.evaluators),
        },
    )


@app.get('/models', response_model=dict[str, ModelInfo])
async def list_models(server: TinyChatServer = Depends(get_server)):
    """List available models"""
    model_info = await server.get_model_info()

    models = {}
    for name, info in model_info.items():
        models[name] = ModelInfo(
            name=info['name'], type=info['type'], healthy=info['healthy']
        )

    return models


@app.get('/models/{model_name}', response_model=ModelInfo)
async def get_model_info(model_name: str, server: TinyChatServer = Depends(get_server)):
    """Get information about a specific model"""
    try:
        model_info = await server.get_model_info(model_name)

        return ModelInfo(
            name=model_info['name'],
            type=model_info['type'],
            healthy=model_info['healthy'],
            config=model_info['config'],
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post('/conversations', response_model=ConversationResponse)
async def create_conversation(
    request: ConversationRequest,
    background_tasks: BackgroundTasks,
    server: TinyChatServer = Depends(get_server),
):
    """Create and run a new conversation"""
    conversation_id = str(uuid.uuid4())

    try:
        agent_configs = [config.dict() for config in request.agent_configs]

        background = None
        if request.background:
            background = TinyChatBackground(**request.background)

        model_name = request.model_name or get_config().default_model
        model_info = await server.get_model_info()
        if model_name not in model_info:
            raise HTTPException(
                status_code=400, detail=f"Model '{model_name}' not available"
            )

        conversations[conversation_id] = {
            'status': 'running',
            'agent_count': len(agent_configs),
            'model_used': model_name,
            'created_at': asyncio.get_event_loop().time(),
        }

        async def run_conversation():
            try:
                logger.info(f'Starting conversation {conversation_id}')

                episode_log = await server.run_conversation(
                    agent_configs=agent_configs,
                    background=background,
                    max_turns=request.max_turns,
                    enable_evaluation=request.enable_evaluation,
                    action_order=request.action_order,
                    speaking_order=request.speaking_order,
                    scenario=request.scenario,
                    model_name=model_name,
                    return_log=request.return_log,
                )

                conversations[conversation_id].update(
                    {
                        'status': 'completed',
                        'episode_log': episode_log,
                        'completed_at': asyncio.get_event_loop().time(),
                    }
                )

                logger.info(f'Conversation {conversation_id} completed')

            except Exception as e:
                logger.error(f'Conversation {conversation_id} failed: {e}')
                conversations[conversation_id].update(
                    {
                        'status': 'failed',
                        'error': str(e),
                        'failed_at': asyncio.get_event_loop().time(),
                    }
                )

        background_tasks.add_task(run_conversation)

        return ConversationResponse(
            conversation_id=conversation_id,
            status='running',
            agent_count=len(agent_configs),
            model_used=model_name,
        )

    except Exception as e:
        logger.error(f'Failed to create conversation: {e}')
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/conversations/{conversation_id}', response_model=ConversationResponse)
async def get_conversation(conversation_id: str):
    """Get conversation status and results"""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail='Conversation not found')

    conv = conversations[conversation_id]

    # Extract turn count and evaluation results from episode log if available
    turn_count = None
    evaluation_results = None
    episode_log = None

    if conv.get('episode_log'):
        episode_log = conv['episode_log']
        if hasattr(episode_log, 'rewards'):
            turn_count = len(episode_log.messages) if episode_log.messages else None
        if hasattr(episode_log, 'reasoning'):
            evaluation_results = {'reasoning': episode_log.reasoning}

    return ConversationResponse(
        conversation_id=conversation_id,
        status=conv['status'],
        agent_count=conv['agent_count'],
        model_used=conv['model_used'],
        turn_count=turn_count,
        evaluation_results=evaluation_results,
        episode_log=episode_log,
    )


@app.get('/conversations')
async def list_conversations():
    """List all conversations"""
    return {
        'conversations': [
            {
                'conversation_id': conv_id,
                'status': conv['status'],
                'agent_count': conv['agent_count'],
                'model_used': conv['model_used'],
            }
            for conv_id, conv in conversations.items()
        ]
    }


@app.delete('/conversations/{conversation_id}')
async def delete_conversation(conversation_id: str):
    """Delete a conversation"""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail='Conversation not found')

    del conversations[conversation_id]
    return {'message': 'Conversation deleted'}


@app.get('/config')
async def get_server_config():
    """Get server configuration"""
    config = get_config()
    return {
        'models': {name: {'type': model.type} for name, model in config.models.items()},
        'default_model': config.default_model,
        'max_turns': config.max_turns,
        'action_order': config.action_order,
        'evaluators': [
            {'type': eval.type, 'enabled': eval.enabled} for eval in config.evaluators
        ],
    }


if __name__ == '__main__':
    import uvicorn

    config = get_config()

    uvicorn.run(
        'tiny_chat.server.api:app',
        host=config.api.host,
        port=config.api.port,
        workers=config.api.workers,
        reload=config.api.reload,
    )
