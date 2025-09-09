# Social Graph Knowledge Propagation Demo

这个demo实现了一个多智能体社交网络中的知识传播基准测试。

## 功能特点

- **知识传播模拟**: 模拟知识在社交网络中的传播过程
- **私有对话环境**: 智能体只能看到邻居的消息，模拟真实的社交网络
- **LLM语义评估**: 使用大语言模型进行语义理解和知识评估
- **完整的传播链追踪**: 追踪知识从源头到目标的完整传播路径

## 文件说明

### 核心文件

- `social_network_knowledge_demo.py` - 主要演示文件，展示GPT-5发布信息传播
- `demo_knowledge_evaluator.py` - LLM驱动的知识传播评估器
- `knowledge_agents.py` - 具有知识存储和学习能力的智能体
- `private_conversation_environment.py` - 私有对话环境实现

### 额外演示

- `knowledge_propagation_demo.py` - 使用自定义知识智能体的传播演示

## 场景设置

### 智能体角色

1. **Sam Altman** (知识源) - OpenAI CEO，拥有GPT-5发布信息
2. **Alice Chen** (研究员) - 从Sam获取信息，传递给Bob
3. **Bob Wilson** (记者) - 从Alice获取信息，传递给Reporter
4. **Reporter** (报告者) - 最终接收者，询问发布细节

### 社交网络拓扑

```
Sam Altman → Alice Chen → Bob Wilson → Reporter
```

每个智能体只能看到直接邻居的消息，确保知识必须通过传播链传递。

## 使用方法

### 运行主要演示

```bash
cd examples/social_graph_demo
python -m social_network_knowledge_demo
```

### 运行知识智能体演示

```bash
cd examples/social_graph_demo
python -m knowledge_propagation_demo
```

## 评估指标

### 知识传播评估

- **传播率**: 获得目标知识的智能体比例
- **准确率**: 传播信息的准确程度
- **完整性**: 知识内容的完整程度
- **置信度**: 评估结果的可信度

### LLM语义分析

评估器对每条消息进行语义分析，判断：

- 是否包含目标知识
- 知识的完整性和准确性
- 智能体对知识的理解程度

## 环境要求

- Python 3.8+
- OpenAI API Key (设置为环境变量 `OPENAI_API_KEY`)
- 依赖包：tiny_chat, litellm

## 输出示例

```
CUSTOM KNOWLEDGE EVALUATION
==================================================

KNOWLEDGE_PROPAGATION: 9.90/10
Comments:
Knowledge Propagation Analysis:
- Knowledge mentions: 4
- Agents reached: 3
- Propagation rate: 100.0%
- Accuracy rate: 97.5%
- Agents with knowledge: Sam Altman, Reporter, Alice Chen

Detailed LLM Analysis:
1. Sam Altman:
   Has Knowledge: True
   Completeness: 100.0%
   Accuracy: 100.0%
   Confidence: 95.0%
   Reasoning: The message explicitly states that the release of GPT-5 is targeted for May 2025...
```

## 扩展性

这个框架可以轻松扩展到：

- 不同的知识类型和内容
- 更复杂的社交网络拓扑
- 多种评估维度和指标
- 自定义智能体行为和目标
