import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, validator


class ModelProviderConfig(BaseModel):
    """Configuration for a model provider"""

    name: str = Field(description='Model name')
    type: Literal[
        'openai',
        'anthropic',
        'together',
        'vllm',
        'ollama',
        'bedrock',
        'azure',
        'palm',
        'cohere',
        'replicate',
        'litellm',
        'custom',
    ] = Field(description='Provider type')
    api_base: Optional[str] = Field(default=None, description='API base URL')
    api_key: Optional[str] = Field(default=None, description='API key')
    temperature: float = Field(default=0.7, description='Generation temperature')
    max_tokens: Optional[int] = Field(default=None, description='Maximum tokens')
    timeout: int = Field(default=30, description='Request timeout in seconds')

    @validator('api_key')
    def resolve_api_key(cls, v, values):
        """Resolve API key from environment if not provided"""
        if v is None:
            provider_type = values.get('type', '').upper()
            env_key = f'{provider_type}_API_KEY'
            return os.getenv(env_key)
        return v


class EvaluatorConfig(BaseModel):
    """Configuration for an evaluator"""

    type: Literal['rule_based', 'llm', 'custom'] = Field(description='Evaluator type')
    model: Optional[str] = Field(
        default=None, description='Model to use for LLM evaluators'
    )
    config: Dict[str, Any] = Field(
        default_factory=dict, description='Evaluator-specific config'
    )
    enabled: bool = Field(default=True, description='Whether evaluator is enabled')


class LoggingConfig(BaseModel):
    """Logging configuration"""

    level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR'] = Field(default='INFO')
    format: str = Field(default='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_path: Optional[str] = Field(default=None, description='Log file path')
    max_file_size: str = Field(default='10MB', description='Maximum log file size')
    backup_count: int = Field(default=5, description='Number of backup log files')


class APIConfig(BaseModel):
    """API server configuration"""

    host: str = Field(default='0.0.0.0', description='Host to bind to')
    port: int = Field(default=8000, description='Port to bind to')
    workers: int = Field(default=1, description='Number of worker processes')
    reload: bool = Field(default=False, description='Enable auto-reload')
    cors_origins: List[str] = Field(
        default_factory=lambda: ['*'], description='CORS origins'
    )
    rate_limit: str = Field(default='100/minute', description='Rate limiting')


class ServerConfig(BaseModel):
    """Main server configuration"""

    # Core settings
    models: Dict[str, ModelProviderConfig] = Field(description='Available models')
    evaluators: List[EvaluatorConfig] = Field(
        default_factory=list, description='Evaluators'
    )
    default_model: str = Field(
        default='gpt-4o-mini', description='Default model to use'
    )

    # Environment settings
    max_turns: int = Field(default=20, description='Maximum conversation turns')
    action_order: Literal['simultaneous', 'round-robin', 'sequential', 'random'] = (
        Field(default='simultaneous', description='Agent action order')
    )
    available_action_types: List[str] = Field(
        default_factory=lambda: [
            'none',
            'speak',
            'non-verbal communication',
            'action',
            'leave',
        ],
        description='Available action types',
    )

    # System settings
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    enable_metrics: bool = Field(default=True, description='Enable metrics collection')
    metrics_port: int = Field(default=9090, description='Metrics server port')

    @validator('default_model')
    def validate_default_model(cls, v, values):
        """Ensure default model exists in models config"""
        models = values.get('models', {})
        if v not in models:
            raise ValueError(f"Default model '{v}' not found in models configuration")
        return v


class ConfigManager:
    """Manages server configuration loading and validation"""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or self._find_config_file()
        self._config: Optional[ServerConfig] = None

    def _find_config_file(self) -> Path:
        """Find configuration file in standard locations"""
        search_paths = [
            Path('config/tiny_chat.yaml'),
            Path('tiny_chat.yaml'),
            Path.home() / '.config' / 'tiny_chat' / 'config.yaml',
            Path('/etc/tiny_chat/config.yaml'),
        ]

        for path in search_paths:
            if path.exists():
                return path

        # Return default path if no config found
        return Path('config/tiny_chat.yaml')

    def load_config(self) -> ServerConfig:
        """Load configuration from file"""
        if self._config is not None:
            return self._config

        if not self.config_path.exists():
            # Create default configuration
            self._config = self._create_default_config()
            self.save_config(self._config)
        else:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            self._config = ServerConfig(**config_data)

        return self._config

    def save_config(self, config: ServerConfig) -> None:
        """Save configuration to file"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.config_path, 'w', encoding='utf-8') as f:
            # Convert to dict and write as YAML
            config_dict = config.dict()
            yaml.safe_dump(config_dict, f, default_flow_style=False, indent=2)

    def _create_default_config(self) -> ServerConfig:
        """Create default configuration"""
        return ServerConfig(
            models={
                'gpt-4o-mini': ModelProviderConfig(
                    name='gpt-4o-mini', type='openai', temperature=0.7
                ),
                'gpt-4': ModelProviderConfig(
                    name='gpt-4', type='openai', temperature=0.7
                ),
            },
            evaluators=[
                EvaluatorConfig(
                    type='rule_based',
                    config={'max_turn_number': 20, 'max_stale_turn': 2},
                ),
                EvaluatorConfig(
                    type='llm', model='gpt-4o-mini', config={'dimensions': 'sotopia'}
                ),
            ],
            default_model='gpt-4o-mini',
        )

    def get_model_config(self, model_name: str) -> ModelProviderConfig:
        """Get configuration for a specific model"""
        config = self.load_config()
        if model_name not in config.models:
            raise ValueError(f"Model '{model_name}' not found in configuration")
        return config.models[model_name]

    def reload_config(self) -> ServerConfig:
        """Reload configuration from file"""
        self._config = None
        return self.load_config()


# Global config manager instance
config_manager = ConfigManager()


def get_config() -> ServerConfig:
    """Get the current server configuration"""
    return config_manager.load_config()


def get_model_config(model_name: str) -> ModelProviderConfig:
    """Get model configuration by name"""
    return config_manager.get_model_config(model_name)
