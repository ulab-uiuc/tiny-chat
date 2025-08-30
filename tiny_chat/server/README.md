# TinyChat Server

Multi-agent conversation server with configuration management, model provider abstraction, plugin system, and HTTP API.

## Features

- **Configuration Management**: YAML config files with environment variable support
- **Unified Model Providers**: LiteLLM-based interface supporting OpenAI, Anthropic, Together, vLLM, Ollama, etc.
- **Plugin System**: Extensible evaluator plugins
- **HTTP API**: RESTful API for conversation management

## Quick Start

### 1. Install Dependencies

```bash
pip install fastapi uvicorn pydantic pyyaml litellm
```

### 2. Configuration

Create `config/tiny_chat.yaml`:

```yaml
models:
  gpt-4o-mini:
    name: gpt-4o-mini
    type: openai
    temperature: 0.7

default_model: gpt-4o-mini

evaluators:
  - type: rule_based
    enabled: true
    config:
      max_turn_number: 20
  - type: llm
    enabled: true
    model: gpt-4o-mini

api:
  host: 0.0.0.0
  port: 8000
```

### 3. Start Server

```bash
# CLI
python -m tiny_chat.server.cli serve

# Direct API
python -m tiny_chat.server.api

# Custom config
python -m tiny_chat.server.cli -c config/my_config.yaml serve
```

### 4. API Usage

```python
import requests

# Create conversation
response = requests.post('http://localhost:8000/conversations', json={
    "agent_configs": [
        {"name": "Alice", "type": "llm"},
        {"name": "Bob", "type": "llm"}
    ],
    "scenario": "Two friends discussing weekend plans",
    "max_turns": 10
})

conversation_id = response.json()["conversation_id"]

# Get results
result = requests.get(f'http://localhost:8000/conversations/{conversation_id}')
print(result.json())
```

## Model Provider Examples

### OpenAI

```yaml
models:
  gpt-4:
    name: gpt-4
    type: openai
    temperature: 0.7
    # Uses OPENAI_API_KEY env var
```

### Anthropic

```yaml
models:
  claude:
    name: claude-3-sonnet-20240229
    type: anthropic
    temperature: 0.7
    # Uses ANTHROPIC_API_KEY env var
```

### Together AI

```yaml
models:
  llama-70b:
    name: meta-llama/Llama-2-70b-chat-hf
    type: together
    temperature: 0.7
    # Uses TOGETHER_API_KEY env var
```

### vLLM

```yaml
models:
  llama-2-7b:
    name: llama-2-7b-chat
    type: vllm
    api_base: http://localhost:8000
    temperature: 0.8
```

### Ollama

```yaml
models:
  llama2:
    name: llama2
    type: ollama
    api_base: http://localhost:11434
    temperature: 0.8
```

### Azure OpenAI

```yaml
models:
  gpt-4-azure:
    name: gpt-4
    type: azure
    api_base: https://your-resource.openai.azure.com
    temperature: 0.7
    # Requires AZURE_API_KEY and AZURE_API_VERSION env vars
```

## Using Plugins

### Overview

Plugins provide an extensible way to evaluate conversations without changing core server logic. Each evaluator plugin implements a uniform async interface and can be configured via YAML or registered programmatically.

## Using WorkflowProvider

### Overview

WorkflowProvider is a highly extensible model provider that supports:

1. **Custom Request Formats**: Customize how requests are sent via RequestHandler
2. **Custom Response Parsing**: Customize how responses are parsed via ResponseParser
3. **Registration Mechanism**: Register custom processors
4. **Combination Strategies**: Support complex model combinations (e.g., SFT + Reward Model)

### Basic Usage

#### 1. Configuration File Usage

```yaml
# config/tiny_chat.yaml
models:
  my-workflow-model:
    name: my-workflow-model
    type: workflow
    custom_config:
      request_handler: "default" # Use default request handler
      response_parser: "default" # Use default response parser
      endpoint: "https://your-api.com/generate"
      headers:
        Authorization: "Bearer ${YOUR_API_KEY}"
        Content-Type: "application/json"
```

#### 2. Start Service

```bash
export YOUR_API_KEY="your-actual-api-key"
python -m tiny_chat.server.cli serve
```

#### 3. Use API

```python
import requests

response = requests.post('http://localhost:8000/conversations', json={
    "agent_configs": [
        {"name": "Alice", "type": "llm", "model": "my-workflow-model"},
        {"name": "Bob", "type": "llm", "model": "my-workflow-model"}
    ],
    "scenario": "Two friends discussing AI",
    "max_turns": 5
})
```

### Advanced Usage

#### 1. SFT + Reward Model Combination

```yaml
models:
  sft-reward-combo:
    name: sft-reward-combo
    type: workflow
    custom_config:
      request_handler: "sft_reward"
      response_parser: "sft_reward"
      sft_endpoint: "https://sft-api.com/generate"
      reward_endpoint: "https://reward-api.com/score"
      headers:
        Authorization: "Bearer ${API_KEY}"
```

#### 2. Custom Processors

```python
# my_custom_handlers.py
from tiny_chat.server.providers.workflow_provider import (
    RequestHandler, ResponseParser, WorkflowProvider
)

class MyCustomRequestHandler(RequestHandler):
    async def process(self, prompt: str, **kwargs) -> dict[str, Any]:
        # Custom request logic
        endpoint = self.config.get("endpoint")

        # Build special format request
        request_data = {
            "input_text": prompt,
            "params": {
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 100)
            }
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(endpoint, json=request_data)
            return response.json()

class MyCustomResponseParser(ResponseParser):
    def parse(self, raw_response: dict[str, Any]) -> ModelResponse:
        # Custom response parsing
        content = raw_response.get("generated_text", "")

        return ModelResponse(
            content=content,
            usage=raw_response.get("usage"),
            model="my-custom",
            metadata={"provider": "workflow_custom"}
        )

# Register custom processors
WorkflowProvider.register_request_handler("my_custom", MyCustomRequestHandler)
WorkflowProvider.register_response_parser("my_custom", MyCustomResponseParser)
```

#### 3. Use Custom Processors in Configuration

```yaml
models:
  custom-model:
    name: custom-model
    type: workflow
    custom_config:
      request_handler: "my_custom"
      response_parser: "my_custom"
      endpoint: "https://your-custom-api.com/v1/generate"
```

### Real-World Examples

#### 1. Ensemble Multiple Models for Voting

```python
class EnsembleRequestHandler(RequestHandler):
    async def process(self, prompt: str, **kwargs) -> dict[str, Any]:
        # Call multiple models
        model_endpoints = self.config.get("models", [])
        responses = []

        async with httpx.AsyncClient() as client:
            for endpoint in model_endpoints:
                response = await client.post(
                    endpoint["url"],
                    json={"prompt": prompt, **kwargs}
                )
                responses.append({
                    "model": endpoint["name"],
                    "response": response.json().get("text", ""),
                    "confidence": response.json().get("confidence", 0.5)
                })

        return {"responses": responses}

class EnsembleResponseParser(ResponseParser):
    def parse(self, raw_response: dict[str, Any]) -> ModelResponse:
        responses = raw_response["responses"]

        # Select highest confidence response
        best = max(responses, key=lambda x: x["confidence"])

        return ModelResponse(
            content=best["response"],
            metadata={
                "ensemble_size": len(responses),
                "selected_model": best["model"],
                "confidence": best["confidence"]
            }
        )
```

#### 2. Model with Post-Processing

```python
class PostProcessingRequestHandler(RequestHandler):
    async def process(self, prompt: str, **kwargs) -> dict[str, Any]:
        # 1. Call base model
        base_response = await self._call_base_model(prompt, **kwargs)

        # 2. Call post-processing model (e.g., grammar check, fact check)
        processed_response = await self._post_process(
            base_response.get("text", "")
        )

        return {
            "original": base_response.get("text", ""),
            "processed": processed_response.get("text", ""),
            "corrections": processed_response.get("corrections", [])
        }
```

### Complete Configuration Example

```yaml
models:
  # Simple custom API
  simple-api:
    name: simple-api
    type: workflow
    custom_config:
      request_handler: "default"
      response_parser: "default"
      endpoint: "https://api.example.com/generate"
      headers:
        Authorization: "Bearer ${API_KEY}"
        X-Custom-Header: "value"

  # SFT + Reward Model
  sft-reward:
    name: sft-reward
    type: workflow
    custom_config:
      request_handler: "sft_reward"
      response_parser: "sft_reward"
      sft_endpoint: "https://sft.example.com/generate"
      reward_endpoint: "https://reward.example.com/score"
      headers:
        Authorization: "Bearer ${SFT_API_KEY}"

  # Custom processor
  custom-handler:
    name: custom-handler
    type: workflow
    custom_config:
      request_handler: "my_custom"
      response_parser: "my_custom"
      endpoint: "https://custom.example.com/api"
      custom_param: "value"

default_model: sft-reward

evaluators:
  - type: llm
    enabled: true
    model: sft-reward
```

### Error Handling

WorkflowProvider provides complete error handling:

```python
try:
    response = await provider.generate("Hello world")
    print(response.content)
except RuntimeError as e:
    print(f"Generation failed: {e}")
    # Can try to fallback to other models
```

### Best Practices

1. **Modular Design**: Separate request processing and response parsing
2. **Error Handling**: Add appropriate exception handling in custom processors
3. **Configuration Validation**: Validate necessary configuration parameters
4. **Logging**: Add appropriate logs for debugging
5. **Performance Considerations**: Consider async processing and caching for complex workflows

### Extensibility

WorkflowProvider's design makes extension very easy:

1. **New RequestHandler**: Handle different API formats
2. **New ResponseParser**: Parse different response formats
3. **Combination Strategies**: Implement complex model combination logic
4. **Middleware Support**: Add authentication, caching, retry, etc.

This design allows you to easily integrate any existing AI model or service without modifying core code.

### Built-in Plugins

- **llm**: LLM-based evaluator (wraps `EpisodeLLMEvaluator`), configurable with a model and dimensions
- **rule_based**: Deterministic evaluator with simple rules (e.g., max turns)

List available plugin types at runtime via `PluginManager.get_available_types()`.

### Configure via YAML

Add evaluators to your `config/tiny_chat.yaml`:

```yaml
evaluators:
  - type: rule_based
    enabled: true
    config:
      max_turn_number: 20

  - type: llm
    enabled: true
    model: gpt-4o-mini # must match a model in models:
    config:
      dimensions: sotopia # optional, defaults to sotopia
```

The server loads these via the plugin manager and attaches them to each conversation environment.

### Programmatic Usage

```python
from tiny_chat.server.plugins.manager import PluginManager
from tiny_chat.server.config import EvaluatorConfig
from tiny_chat.server.providers import BaseModelProvider

manager = PluginManager()

# Example: create from config objects
cfg = EvaluatorConfig(type="llm", enabled=True, model="gpt-4o-mini", config={"dimensions": "sotopia"})
model_providers: dict[str, BaseModelProvider] = {"gpt-4o-mini": some_provider}
evaluator = manager.create_evaluator(cfg, model_providers)

loaded = manager.get_evaluators()
```

### Implement a Custom Evaluator Plugin

Create a new class that extends `EvaluatorPlugin` and implement `plugin_type` and `evaluate`:

```python
from typing import Any, List, Tuple
from tiny_chat.server.plugins.base import EvaluatorPlugin

class MyEvaluatorPlugin(EvaluatorPlugin):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.threshold = float(config.get("threshold", 0.5))

    @property
    def plugin_type(self) -> str:
        return "my_evaluator"

    async def evaluate(
        self,
        turn_number: int,
        messages: List[Tuple[str, Any]],
    ) -> List[Tuple[str, Tuple[Tuple[str, int | float | bool], str]]]:
        # Example: score each agent based on custom rules
        results: list[tuple[str, tuple[tuple[str, int], str]]] = []
        for sender, msg in messages:
            if sender == "Environment":
                continue
            score = 8  # compute a numeric score
            reasoning = "Scored by custom evaluator"
            results.append((sender, (("overall", score), reasoning)))
        return results
```

### Register Your Plugin Type

Register the new type so YAML can reference it:

```python
from tiny_chat.server.plugins.manager import PluginManager

manager = PluginManager()
manager.register_plugin("my_evaluator", MyEvaluatorPlugin)

# Now usable in YAML:
# evaluators:
#   - type: my_evaluator
#     enabled: true
#     config:
#       threshold: 0.7
```

### Return Format

`evaluate` must return a list of entries, one per agent you scored:

```python
[(agent_name, ((dimension_name, score), reasoning_string))]
```

- **agent_name**: `str` matching the agent in the conversation
- **dimension_name**: `str` name for the metric (e.g., "overall", "goal_achievement")
- **score**: `int | float | bool` numeric/boolean score
- **reasoning_string**: `str` brief explanation supporting the score

### Access the Terminal Evaluator (optional)

If your evaluator wraps a terminal evaluator (e.g., the LLM evaluator), implement `get_terminal_evaluator()` to expose it for end-of-episode scoring/prompts:

```python
def get_terminal_evaluator(self) -> Any:
    return self.evaluator  # underlying evaluator object
```

### End-to-End Example

1. Implement `MyEvaluatorPlugin` (above) and register it via `PluginManager.register_plugin`.

2. Reference it in YAML:

```yaml
evaluators:
  - type: my_evaluator
    enabled: true
    config:
      threshold: 0.7
```

3. Start the server and run a conversation. The plugin will contribute per-turn scores and reasoning which are aggregated by the environment.

## CLI Usage

### Starting the Server

```bash
# Basic startup
python -m tiny_chat.server.cli

# Custom configuration
python -m tiny_chat.server.cli --config config/my_config.yaml

# Custom host and port
python -m tiny_chat.server.cli --host 0.0.0.0 --port 8080

# Development mode (auto-reload)
python -m tiny_chat.server.cli --reload --verbose

# Multi-process mode
python -m tiny_chat.server.cli --workers 4
```

### Alternative Operations

Other operations are available through more appropriate interfaces:

#### 1. Configuration Validation

```bash
# Use Python script
python examples/check_config.py

# Use HTTP API
curl http://localhost:8000/health
```

#### 2. Health Checks

```bash
# Use Python script
python examples/check_health.py

# Use HTTP API
curl http://localhost:8000/health
curl http://localhost:8000/models
```

#### 3. Running Demos

```bash
# Use Python script
python examples/run_demo.py --agents Alice Bob --model gpt-4o-mini

# Use HTTP API
curl -X POST http://localhost:8000/conversations \
  -H "Content-Type: application/json" \
  -d '{
    "agent_configs": [
      {"name": "Alice", "type": "llm"},
      {"name": "Bob", "type": "llm"}
    ],
    "scenario": "A casual conversation",
    "model_name": "gpt-4o-mini"
  }'
```

#### 4. Viewing Metrics

```bash
# Use HTTP API
curl http://localhost:8000/conversations/{conversation_id}

# Access Prometheus metrics (if enabled)
curl http://localhost:9090/metrics
```

#### 5. Python API Usage

```python
import asyncio
from tiny_chat.server.core import create_server

async def main():
    async with create_server() as server:
        episode_log = await server.run_conversation(
            agent_configs=[
                {"name": "Alice", "type": "llm"},
                {"name": "Bob", "type": "llm"}
            ],
            scenario="A friendly conversation",
            model_name="gpt-4o-mini",
            max_turns=10
        )
        print(f"Conversation completed: {episode_log.reasoning}")

asyncio.run(main())
```

### Configuration Management

```python
from tiny_chat.server.config import ConfigManager
from pathlib import Path

# Load configuration
config_manager = ConfigManager(Path("config/tiny_chat.yaml"))
config = config_manager.load_config()

# Validate configuration
print(f"Models: {list(config.models.keys())}")
print(f"Default model: {config.default_model}")

# Save configuration
config_manager.save_config(config)
```

### Available HTTP API Endpoints

- `GET /health` - Health check
- `GET /models` - List available models
- `POST /conversations` - Start conversation
- `GET /conversations/{id}` - Get conversation status
- `GET /metrics` - Prometheus metrics (if enabled)

### Logging

```yaml
logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file_path: logs/tiny_chat.log
```

## Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000 9090

CMD ["python", "-m", "tiny_chat.server.cli"]
```

### Docker Compose Example

```yaml
version: "3.8"
services:
  tinychat:
    build: .
    ports:
      - "8000:8000"
      - "9090:9090"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    volumes:
      - ./config:/app/config
      - ./logs:/app/logs
    command: ["python", "-m", "tiny_chat.server.cli", "--host", "0.0.0.0"]
```
