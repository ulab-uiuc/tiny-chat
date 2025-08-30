<img src="assets/tiny_chat.png" style="width: 100%;"></img>

<h1 align="center">TinyChat: A lightweight platform for realistic social simulations</h1>

<div align="center">

[![Python 3.10](https://img.shields.io/badge/python-%E2%89%A53.10-blue)](https://www.python.org/downloads/release/python-3109/)
[![GitHub pull request](https://img.shields.io/badge/PRs-welcome-red)](https://github.com/hiyouga/LLaMA-Factory/pulls)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

# Introduction

**Tiny-Chat** is a lightweight , user-friendly and highly extensible multi-agent stimulation framework. Designed for social stimulation research.

### Core Features

- **🤖 Multi-Agent interactions**: Support for multiple AI agents with distinct personalities and goals

- **🌍 Flexible Environments**: Configurable conversation scenarios and relationship dynamics

- **📊 Built-in and Plugin Evaluation**: Multi-dimensional conversation evaluation using LLM and rule-based methods and Extensible evaluator plugins for custom assessment strategies

- **🚀 Server Architecture**: High-performance server with HTTP API and configuration management

# Installation

**Install from source**

```bash
# clone the repository
git@github.com:ulab-uiuc/tiny-chat.git
cd tiny-chat

# create conda environment
conda create -n tiny-chat python=3.10
conda activate tiny-chat

# Install Poetry
curl -sSL https://install.python-poetry.org | python3
export PATH="$HOME/.local/bin:$PATH"

# Install dependencies
poetry install
```

## Get Started

Before running any code, set your API key, you can also manage your api key in `.yaml`.

```bash
export OPENAI_API_KEY=your-key-here
# or use DEEPSEEK_API_KEY, ANTHROPIC_API_KEY, or OPENROUTER_API_KEY
```

### Basic Usage

```python
import asyncio
from tiny_chat.server import TinyChatServer, ServerConfig, ModelProviderConfig

async def basic_conversation():
    config = ServerConfig(
        models={
            'gpt-4o-mini': ModelProviderConfig(
                name='gpt-4o-mini',
                type='openai',
                temperature=0.7
            )
        },
        default_model='gpt-4o-mini'
    )

    async with TinyChatServer(config) as server:
        episode_log = await server.run_conversation(
            agent_configs=[
                {"name": "Alice", "type": "llm", "goal": "Convince Bob to join the project"},
                {"name": "Bob", "type": "llm", "goal": "Learn more about the project"}
            ],
            scenario="Two colleagues discussing a new project collaboration",
            max_turns=10
        )

        print(f"Conversation completed: {episode_log.reasoning}")

asyncio.run(basic_conversation())
```

### Configuration in YAML

Create `config/tiny_chat.yaml`:

```yaml
models:
  gpt-4o-mini:
    name: gpt-4o-mini
    type: openai
    temperature: 0.7

default_model: gpt-4o-mini
max_turns: 20

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

Start the server:

```bash
python -m tiny_chat.server.cli serve
```

### Examples

For detailed examples and use cases, check the `examples/` directory:

- `examples/human_agent_chat_demo.py` - Human-in-the-loop conversations
- `examples/multi_agent_chat_demo.py` - Multi-agent scenarios
- `examples/multi_agent_chat_obs_control.py` - Observation control examples

## Advanced Usage

### CLI Tools

```bash
# Generate environment profile
python -m tiny_chat.utils.cli env-profile "asking my boyfriend to stop being friends with his ex"

# Generate agent action
python -m tiny_chat.utils.cli action --agent Alice --goal "Convince Bob" --history "Previous conversation..."

# Generate script
python -m tiny_chat.utils.cli script --background-file background.json --agent-name Alice --agent-name Bob
```

### Plugin System

```python
from tiny_chat.server.plugins import PluginManager, EvaluatorConfig

# Create custom evaluator plugin
class CustomEvaluatorPlugin(EvaluatorPlugin):
    @property
    def plugin_type(self) -> str:
        return 'custom'

    def _create_evaluator(self) -> Evaluator:
        return CustomEvaluator()

# Register and use
manager = PluginManager()
manager.register_plugin('custom', CustomEvaluatorPlugin)
```
