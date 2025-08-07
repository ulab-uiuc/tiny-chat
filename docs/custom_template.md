# Custom Template Guide for Tiny Chat

## Getting Started

### Installation

The custom template system is included with Tiny Chat. No additional installation is required.

### Basic Setup

```python
from tiny_chat.utils.template import TemplateManager

# Initialize the template manager
manager = TemplateManager()

# Check available templates
print("Available templates:", manager.list_template_names())
```

## Template Basics

### Template Structure

Templates use Python's string formatting syntax with named parameters:

```python
template = """
You are {agent_name}, a {role} with the goal of {goal}.

Current situation: {situation}
Available actions: {actions}
Turn number: {turn_number}

Please respond appropriately based on your character and goals.
"""
```

### Parameter Types

- **Required Parameters**: Must be provided when formatting
- **Optional Parameters**: Can have default values
- **Dynamic Parameters**: Calculated at runtime

### Template Categories

1. **Action Templates**: For agent action generation
2. **Profile Templates**: For agent personality definition
3. **Goal Templates**: For objective setting
4. **Script Templates**: For conversation scripting
5. **Utility Templates**: For data processing and conversion

## Creating Custom Templates

### Method 1: Using Python API

```python
from tiny_chat.utils.template import TemplateManager

manager = TemplateManager()

# Create a custom action template
manager.add_template(
    "CUSTOM_ACTION",
    """
    You are {agent_name}, a {personality_type} {role}.

    Your current goal: {goal}
    Conversation history: {history}
    Available actions: {actions}

    Instructions:
    - Stay in character as a {personality_type} person
    - Work towards your goal: {goal}
    - Choose from available actions: {actions}
    - Be natural and engaging

    Please respond with your action.
    """
)

# Create a custom profile template
manager.add_template(
    "CUSTOM_PROFILE",
    """
    Character Profile: {name}

    Background: {background}
    Personality: {personality}
    Goals: {goals}
    Communication Style: {communication_style}

    This character should:
    - {behavior_guidelines}
    - {interaction_preferences}
    - {goal_priorities}
    """
)
```

### Method 2: Using CLI

```bash
# Add template via command line
python -m tiny_chat.utils.cli add-template CUSTOM_GREETING \
    "Hello {name}! I'm {agent_name}, a {role}. How can I help you today?"

# Add template with description
python -m tiny_chat.utils.cli add-template CUSTOM_GOAL \
    "Your objective is to {objective} while maintaining {constraints}." \
    --description "Custom goal template for specific scenarios"
```

### Method 3: Using REST API

```bash
# Add template via REST API
curl -X POST http://localhost:5000/api/templates \
     -H "Content-Type: application/json" \
     -d '{
       "name": "CUSTOM_ACTION",
       "content": "You are {agent_name}...",
       "description": "Custom action template"
     }'
```

## Template Management

### Listing Templates

```python
# List all templates
templates = manager.list_templates()
for name, content in templates.items():
    print(f"Template: {name}")
    print(f"Content: {content[:100]}...")
    print()

# List template names only
names = manager.list_template_names()
print("Available templates:", names)
```

### Updating Templates

```python
# Update existing template
manager.set_template(
    "CUSTOM_ACTION",
    """
    Updated template content...
    You are {agent_name}, now with enhanced instructions.
    """
)

# Update with save control
manager.set_template("CUSTOM_ACTION", new_content, save=False)
manager.save_templates()  # Save manually later
```

### Removing Templates

```python
# Remove template
manager.remove_template("CUSTOM_TEMPLATE")

# Remove with confirmation
try:
    manager.remove_template("IMPORTANT_TEMPLATE")
except ValueError as e:
    print(f"Cannot remove: {e}")
```

### Template Information

```python
# Get detailed template info
info = manager.get_template_info("CUSTOM_ACTION")
print(f"Name: {info['name']}")
print(f"Parameters: {info['parameters']}")
print(f"Length: {info['length']} characters")
print(f"Lines: {info['lines']}")
print(f"Last modified: {info['last_modified']}")
```

## Integration with Tiny Chat

### Using Custom Templates in Agent Generation

```python
from tiny_chat.generator.generate_template import agenerate_action
from tiny_chat.utils.template import TemplateManager

async def generate_custom_action():
    manager = TemplateManager()

    # Get custom template
    custom_template = manager.get_template("CUSTOM_ACTION")

    # Generate action using custom template
    action = await agenerate_action(
        model_name="gpt-4o-mini",
        history="Previous conversation...",
        turn_number=5,
        action_types=["speak", "none", "leave"],
        agent="Alice",
        goal="Be helpful and friendly",
        custom_template=custom_template,
        template_params={
            "agent_name": "Alice",
            "personality_type": "friendly",
            "role": "assistant",
            "goal": "Help the user",
            "actions": "speak, none, leave"
        }
    )

    return action
```

### Using Custom Templates in Server

```python
from tiny_chat.utils.server import TinyChatServer
from tiny_chat.utils.template import TemplateManager

class CustomTinyChatServer(TinyChatServer):
    def __init__(self, api_key=None):
        super().__init__(api_key)
        self.template_manager = TemplateManager()

    async def _create_agents(self, agent_configs, background=None):
        agents = {}

        for config in agent_configs:
            # Use custom template if specified
            if 'custom_template' in config:
                template_name = config['custom_template']
                template_params = config.get('template_params', {})

                # Format custom template
                custom_prompt = self.template_manager.format_template(
                    template_name, **template_params
                )

                # Add custom prompt to agent config
                config['custom_prompt'] = custom_prompt

            # Create agent with custom prompt
            agent = self._create_single_agent(config)
            agents[config['name']] = agent

        return agents

# Usage
server = CustomTinyChatServer(api_key="your-api-key")

agent_configs = [
    {
        "name": "Alice",
        "type": "llm",
        "model": "gpt-4o-mini",
        "custom_template": "FRIENDLY_ASSISTANT",
        "template_params": {
            "agent_name": "Alice",
            "personality_type": "friendly",
            "role": "assistant"
        }
    }
]

await server.two_agent_run_conversation(agent_configs)
```

## Advanced Features

### Template Inheritance

```python
# Create base template
manager.add_template(
    "BASE_ACTION",
    """
    You are {agent_name}.
    Your goal: {goal}
    Available actions: {actions}
    {custom_instructions}
    """
)

# Create specialized templates
manager.add_template(
    "FRIENDLY_ACTION",
    manager.get_template("BASE_ACTION").format(
        custom_instructions="Be warm, supportive, and genuinely interested."
    )
)

manager.add_template(
    "PROFESSIONAL_ACTION",
    manager.get_template("BASE_ACTION").format(
        custom_instructions="Be formal, concise, and business-like."
    )
)
```

### Template Macros

```python
# Define reusable macros
manager.add_macro("GREETING", "Hello {name}, nice to meet you!")
manager.add_macro("GOAL_REMINDER", "Remember your goal: {goal}")

# Use macros in templates
manager.add_template(
    "WELCOME_TEMPLATE",
    """
    {GREETING}
    {GOAL_REMINDER}
    How can I help you today?
    """
)
```

### Conditional Templates

```python
# Create conditional template
manager.add_template(
    "CONDITIONAL_ACTION",
    """
    You are {agent_name}.

    {% if turn_number == 1 %}
    This is the first turn. Introduce yourself warmly.
    {% elif turn_number > 10 %}
    The conversation has been going on for a while. Consider wrapping up.
    {% else %}
    Continue the conversation naturally.
    {% endif %}

    Available actions: {actions}
    """
)
```

### Template Validation

```python
# Validate template syntax
try:
    manager.validate_template("CUSTOM_ACTION")
    print("Template is valid")
except ValueError as e:
    print(f"Template error: {e}")

# Check required parameters
required = manager.get_required_parameters("CUSTOM_ACTION")
print(f"Required parameters: {required}")

# Validate template formatting
try:
    formatted = manager.format_template("CUSTOM_ACTION", agent_name="Alice")
    print("Template formatted successfully")
except KeyError as e:
    print(f"Missing parameter: {e}")
```

## Examples

### Example 1: Customer Service Agent

```python
# Create customer service template
manager.add_template(
    "CUSTOMER_SERVICE",
    """
    You are {agent_name}, a {experience_level} customer service representative.

    Your role: {role}
    Company: {company}
    Service area: {service_area}

    Guidelines:
    - Always be polite and professional
    - Listen carefully to customer concerns
    - Provide accurate and helpful information
    - Escalate complex issues when necessary
    - Follow company policies and procedures

    Current customer: {customer_name}
    Customer issue: {customer_issue}
    Available actions: {actions}

    Please respond appropriately to help the customer.
    """
)

# Use in conversation
agent_config = {
    "name": "ServiceAgent",
    "type": "llm",
    "model": "gpt-4o-mini",
    "custom_template": "CUSTOMER_SERVICE",
    "template_params": {
        "agent_name": "Sarah",
        "experience_level": "senior",
        "role": "technical support",
        "company": "TechCorp",
        "service_area": "software products",
        "customer_name": "John",
        "customer_issue": "login problems"
    }
}
```

### Example 2: Educational Tutor

```python
# Create educational tutor template
manager.add_template(
    "EDUCATIONAL_TUTOR",
    """
    You are {agent_name}, a {subject} tutor with {experience} years of teaching experience.

    Teaching style: {teaching_style}
    Student level: {student_level}
    Current topic: {current_topic}

    Teaching approach:
    - Adapt explanations to student's level
    - Use examples and analogies
    - Encourage questions and discussion
    - Provide constructive feedback
    - Make learning engaging and fun

    Student: {student_name}
    Student's question: {student_question}
    Available actions: {actions}

    Please help the student understand the topic.
    """
)
```

### Example 3: Creative Writing Assistant

```python
# Create creative writing template
manager.add_template(
    "CREATIVE_WRITER",
    """
    You are {agent_name}, a {genre} writer and creative writing assistant.

    Writing style: {writing_style}
    Expertise: {expertise}
    Creative approach: {creative_approach}

    Your role:
    - Help develop story ideas and plots
    - Provide writing tips and techniques
    - Suggest character development
    - Offer constructive feedback
    - Inspire creativity and imagination

    Current project: {project_type}
    Writer's request: {writer_request}
    Available actions: {actions}

    Please provide creative and helpful assistance.
    """
)
```
