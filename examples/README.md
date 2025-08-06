# Tiny-Chat Examples

This directory showcases various usage examples for Tiny-Chat.

## Example Files

```
examples/
├── README.md
├── two_agents_chat.py      # Two-agent conversation
├── human_agent_chat.py     # Human-AI interactive chat
└── multi_agents_chat.py    # Multi-agent conversation
```

## Running Examples

### Two-Agent Conversation
```bash
python examples/two_agents_chat.py
```

### Human-AI Interactive Chat
```bash
python examples/human_agent_chat.py
```

### Multi-Agent Conversation
```bash
python examples/multi_agents_chat.py
```

## Customizing Examples

You can create your own examples by modifying the following parameters:

### 1. Two-Agent Conversation Customization

In `two_agents_chat.py`, you can modify:

```python
# Agent configurations
agent_configs = [
    {
        "name": "Your Agent 1 Name",
        "agent_number": 1,
        "type": "llm",
        "model": "gpt-4o-mini",
        "goal": "Your Agent 1 Goal",
    },
    {
        "name": "Your Agent 2 Name", 
        "agent_number": 2,
        "type": "llm",
        "model": "gpt-4o-mini",
        "goal": "Your Agent 2 Goal",
    },
]

# Background settings
background = TwoAgentChatBackground(
    scenario="Your conversation scenario description",
    p1_background="Agent 1 background information",
    p2_background="Agent 2 background information", 
    p1_goal="Agent 1 specific goal",
    p2_goal="Agent 2 specific goal",
    p1_name="Agent 1 Name",
    p2_name="Agent 2 Name",
)

# Conversation parameters
await server.two_agent_run_conversation(
    agent_configs=agent_configs,
    background=background,
    max_turns=10,           # Maximum conversation turns
    enable_evaluation=True,  # Enable evaluation
)
```

### 2. Human-AI Interaction Customization

In `human_agent_chat.py`, you can modify:

```python
# AI agent configuration
agent = LLMAgent(
    agent_name="Your AI Assistant Name",
    model="gpt-4o-mini",
    api_key=api_key,
    goal="Your AI Assistant Goal"
)

# Conversation background
background = ChatBackground(
    scenario="Your conversation scenario",
    p1_background="AI Assistant background setting",
    p2_background="User background setting",
    p1_goal="AI Assistant goal",
    p2_goal="User goal",
    p1_name="AI Assistant Name",
    p2_name="User Name"
)
```

### 3. Multi-Agent Conversation Customization

Multi-agent conversations support 4 different action ordering modes. You can assign each one in [action_order](/tiny-chat/examples/multi_agents_chat.py#L74):

- **Sequential**: Agents take turns one by one in a fixed order
- **Round-robin**: Agents take turns in a rotating order (A→B→C→A→B→C...)
- **Simultaneous**: All agents act at the same time in each turn
- **Random**: A random agent is selected to act in each turn

## <span style="color: red;">WARNING: Simultaneous mode is Incomplete</span>

In `multi_agents_chat.py`, you can modify:

```python
# Agent configurations (supports 2-3 agents)
agent_configs = [
    {
        "name": "Agent 1 Name",
        "type": "llm",
        "model": "gpt-4o-mini",
        "goal": "Agent 1 Goal"
    },
    {
        "name": "Agent 2 Name", 
        "type": "llm",
        "model": "gpt-4o-mini",
        "goal": "Agent 2 Goal"
    },
    {
        "name": "Agent 3 Name",
        "type": "llm", 
        "model": "gpt-4o-mini",
        "goal": "Agent 3 Goal"
    }
]

background = MultiAgentChatBackground(
    scenario="Your multi-agent scenario description", 
    agent_configs=[
    {
        'name': 'Agent 1 Name',
        'background': 'Agent 1 detailed background',
        'goal': 'Agent 1 specific goal'
    },
    {
        'name': 'Agent 2 Name',
        'background': 'Agent 2 detailed background', 
        'goal': 'Agent 2 specific goal'
    },
    {
        'name': 'Agent 3 Name',
        'background': 'Agent 3 detailed background',
        'goal': 'Agent 3 specific goal'
    }
]
)

# Conversation parameters
await server.multi_agent_run_conversation(
    agent_configs=agent_configs,
    background=background,
    action_order='simultaneous', # sequential, round-robin, simultaneous, random
    max_turns=12,           # Maximum conversation turns
    enable_evaluation=True   # Enable evaluation
)
```
