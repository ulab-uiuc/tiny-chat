# Tiny Chat - 轻量级多智能体对话系统

[English](/tiny-chat/tiny_chat/README.md) | [中文](/tiny-chat/tiny_chat/README_zh.md)

Tiny Chat 是一个受 [Sotopia](https://github.com/sotopia-lab/sotopia) 启发的简化多智能体对话系统，专为在各种社交场景中创建和管理对话式 AI 智能体而设计。

### 项目结构

#### 📁 `agents/` - 智能体管理
- **用途**: 定义和管理不同类型的对话智能体
- **核心组件**:
  - `LLMAgent`: 基于 LLM 的智能体，使用 OpenAI API 进行自然对话
  - `HumanAgent`: 人机交互框架
- **功能**: 目标导向行为、消息历史管理、上下文构建

#### 📁 `envs/` - 环境管理
- **用途**: 管理对话环境和交互规则
- **核心组件**:
  - `TinyChatEnvironment`: 处理智能体交互的主要环境类
- **功能**:
  - 多种动作类型（说话、非语言交流、行动、离开）
  - 不同动作顺序（同时、轮换、随机）
  - 基于轮次的对话管理
  - 环境状态跟踪

#### 📁 `profile/` - 智能体与环境档案
- **用途**: 定义智能体个性和对话上下文的数据结构
- **核心组件**:
  - `BaseAgentProfile`: 智能体个性、背景和特征
  - `BaseEnvironmentProfile`: 对话场景和约束条件
  - `BaseRelationshipProfile`: 智能体间的关系动态
- **功能**:
  - 个性特征（大五人格、MBTI、道德价值观）
  - 年龄和职业约束
  - 关系类型（陌生人到家庭成员）

#### 📁 `messages/` - 消息系统
- **用途**: 处理智能体与环境之间的所有通信
- **核心组件**:
  - `Message`: 所有消息类型的基础接口
  - `SimpleMessage`: 基本文本消息
  - `Observation`: 环境状态更新
  - `AgentAction`: 智能体行为动作
  - `ChatBackground`: 对话上下文和目标
- **功能**: 自然语言转换、动作解析、对话历史

#### 📁 `generator/` - 内容生成
- **用途**: 使用 LLM 生成对话内容
- **核心组件**:
  - `generate_template.py`: 使用 LiteLLM 的主要生成函数
  - `output_parsers.py`: 结构化输出解析和验证
- **功能**:
  - 智能体动作生成
  - 环境档案生成
  - 脚本式对话生成
  - 从背景生成目标

#### 📁 `evaluator/` - 对话评估
- **用途**: 评估对话质量和智能体表现
- **核心组件**:
  - `RuleBasedTerminatedEvaluator`: 基于规则的对话终止
  - `EpisodeLLMEvaluator`: 基于 LLM 的对话评估
  - `TinyChatDimensions`: 评估指标（目标达成、社交智能等）
- **功能**:
  - 多维度评估
  - 自动对话终止
  - 表现评分和分析

#### 📁 `utils/` - 工具函数
- **用途**: 提供辅助函数和工具
- **核心组件**:
  - `format_docstring.py`: 文档字符串格式化工具
  - `prompt.py`: 所有的提示词 & 提示句子
- **功能**: 代码格式化和文档辅助

#### 📁 `server.py` - 聊天服务器
- **用途**: 运行多智能体对话的高级接口
- **功能**:
  - 对话编排
  - 智能体配置管理
  - 评估集成
  - 演示和测试功能

#### 📁 `logs.py` - 日志系统
- **用途**: 管理日志和调试信息
- **功能**: 结构化日志、错误跟踪、对话监控

