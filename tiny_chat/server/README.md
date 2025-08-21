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

## Plugin Development

### Custom Evaluator

```python
from tiny_chat.server.plugins.base import EvaluatorPlugin

class MyEvaluatorPlugin(EvaluatorPlugin):
    @property
    def plugin_type(self) -> str:
        return "my_evaluator"
    
    async def evaluate(self, turn_number, messages):
        # Implement evaluation logic
        return [("agent_name", (("metric", score), "reasoning"))]
```

## CLI Commands

```bash
# Check config
python -m tiny_chat.server.cli check-config

# Health check
python -m tiny_chat.server.cli check-health

# Run demo
python -m tiny_chat.server.cli demo -a Alice -a Bob -s "Discuss AI future"

# View metrics
python -m tiny_chat.server.cli metrics
```

## Environment Variables

```bash
# OpenAI
export OPENAI_API_KEY="your-key"

# Anthropic
export ANTHROPIC_API_KEY="your-key"

# Together AI
export TOGETHER_API_KEY="your-key"

# Azure OpenAI
export AZURE_API_KEY="your-key"
export AZURE_API_VERSION="2023-12-01-preview"

# Cohere
export COHERE_API_KEY="your-key"

# Replicate
export REPLICATE_API_TOKEN="your-token"

# AWS Bedrock
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export AWS_REGION="us-east-1"
```

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

CMD ["python", "-m", "tiny_chat.server.cli", "serve"]
```
