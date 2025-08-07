# Tiny Chat - Lightweight multi-agent dialogue system

[English](/tiny-chat/tiny_chat/README.md) | [中文](/tiny-chat/tiny_chat/README_zh.md)

Tiny Chat is a simplified multi-agent chat system inspired by [Sotopia](https://github.com/sotopia-lab/sotopia), designed for creating and managing conversational AI agents in various social scenarios.

### Project Structure

#### 📁 `agents/` - Agent Management

- **Purpose**: Defines and manages different types of conversational agents
- **Key Components**:
  - `LLMAgent`: LLM-based agent using OpenAI API for natural conversations
  - `HumanAgent`: Framework for human-in-the-loop interactions
- **Features**: Goal-oriented behavior, message history management, context building

#### 📁 `envs/` - Environment Management

- **Purpose**: Manages the conversation environment and interaction rules
- **Key Components**:
  - `TinyChatEnvironment`: Main environment class handling agent interactions
- **Features**:
  - Multiple action types (speak, non-verbal, action, leave)
  - Different action orders (simultaneous, round-robin, random)
  - Turn-based conversation management
  - Environment state tracking

#### 📁 `profile/` - Agent & Environment Profiles

- **Purpose**: Defines data structures for agent personalities and conversation contexts
- **Key Components**:
  - `BaseAgentProfile`: Agent personality, background, and characteristics
  - `BaseEnvironmentProfile`: Conversation scenarios and constraints
  - `BaseRelationshipProfile`: Relationship dynamics between agents
- **Features**:
  - Personality traits (Big Five, MBTI, moral values)
  - Age and occupation constraints
  - Relationship types (stranger to family member)

#### 📁 `messages/` - Message System

- **Purpose**: Handles all communication between agents and environment
- **Key Components**:
  - `Message`: Base interface for all message types
  - `SimpleMessage`: Basic text messages
  - `Observation`: Environment state updates
  - `AgentAction`: Agent behavior actions
  - `ChatBackground`: Conversation context and goals
- **Features**: Natural language conversion, action parsing, conversation history

#### 📁 `generator/` - Content Generation

- **Purpose**: Generates conversation content using LLMs
- **Key Components**:
  - `generate_template.py`: Main generation functions using LiteLLM
  - `output_parsers.py`: Structured output parsing and validation
- **Features**:
  - Agent action generation
  - Environment profile generation
  - Script-like conversation generation
  - Goal generation from backgrounds

#### 📁 `evaluator/` - Conversation Evaluation

- **Purpose**: Evaluates conversation quality and agent performance
- **Key Components**:
  - `RuleBasedTerminatedEvaluator`: Rule-based conversation termination
  - `EpisodeLLMEvaluator`: LLM-based conversation evaluation
  - `TinyChatDimensions`: Evaluation metrics (goal achievement, social intelligence, etc.)
- **Features**:
  - Multi-dimensional evaluation
  - Automatic conversation termination
  - Performance scoring and analysis

#### 📁 `utils/` - Utility Functions

- **Purpose**: Provides helper functions and utilities
- **Key Components**:
  - `format_docstring.py`: Document string formatting utilities
  - `prompt.py`: All prompt extracted in this file
- **Features**: Code formatting and documentation helpers

#### 📁 `server.py` - Chat Server

- **Purpose**: High-level interface for running multi-agent conversations
- **Features**:
  - Conversation orchestration
  - Agent configuration management
  - Evaluation integration
  - Demo and testing capabilities

#### 📁 `logs.py` - Logging System

- **Purpose**: Manages logging and debugging information
- **Features**: Structured logging, error tracking, conversation monitoring
