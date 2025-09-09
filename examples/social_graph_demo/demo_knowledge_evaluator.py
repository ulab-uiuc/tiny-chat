"""
LLM-based knowledge propagation evaluator using semantic understanding.
"""

import json
import re
from typing import Any

from tiny_chat.evaluator import Evaluator
from tiny_chat.messages import Message
from tiny_chat.server.config import ModelProviderConfig
from tiny_chat.server.providers import ModelProviderFactory


class DemoKnowledgeEvaluator(Evaluator):
    """
    LLM-based evaluator that uses semantic understanding to assess
    knowledge propagation in conversations.
    """

    def __init__(
        self,
        target_knowledge: str = 'GPT-5 will be released in May 2025',
        knowledge_source: str = 'Sam Altman',
        target_agents: list[str] | None = None,
        model_name: str = 'gpt-4o-mini',
    ):
        """
        Initialize the knowledge evaluator.

        Args:
            target_knowledge: The specific knowledge to track
            knowledge_source: The original source agent
            target_agents: List of agents that should receive the knowledge
            model_name: LLM model to use for evaluation
        """
        self.target_knowledge = target_knowledge
        self.knowledge_source = knowledge_source
        self.target_agents = target_agents or []

        model_config = ModelProviderConfig(
            name=model_name, type='openai', temperature=0.1
        )
        self.model_provider = ModelProviderFactory.create_provider(model_config)

        self.agents_with_knowledge = set()
        self._evaluation_cache = {}

    async def _analyze_message_for_knowledge(
        self, agent_name: str, message_text: str
    ) -> dict[str, Any]:
        """Analyze a message for knowledge content using LLM."""
        cache_key = f'{agent_name}:{hash(message_text)}'
        if cache_key in self._evaluation_cache:
            return self._evaluation_cache[cache_key]

        if len(message_text.strip()) < 10 or 'did nothing' in message_text.lower():
            result = {
                'agent': agent_name,
                'has_knowledge': False,
                'completeness': 0.0,
                'accuracy': 0.0,
                'confidence': 0.0,
                'reasoning': 'Message too short or no action taken',
                'key_information': [],
                'message_snippet': message_text[:50] + '...'
                if len(message_text) > 50
                else message_text,
            }
            self._evaluation_cache[cache_key] = result
            return result

        evaluation_prompt = f"""
Analyze whether the following message demonstrates knowledge about this specific information:
TARGET KNOWLEDGE: "{self.target_knowledge}"
MESSAGE FROM {agent_name}:
{message_text}

Determine if this message shows that {agent_name} has learned or possesses the target knowledge.
Consider these factors:
1. Does the message explicitly mention the target knowledge?
2. Does the message show understanding of the key concepts?
3. Does the message indicate the agent has learned this information?
4. Even if not explicitly stated, does the context suggest knowledge acquisition?

Respond ONLY with a JSON object in this exact format:
{{
    "has_knowledge": true/false,
    "completeness": 0-100 (percentage of target knowledge demonstrated),
    "accuracy": 0-100 (accuracy of the information shared),
    "confidence": 0-100 (confidence in this evaluation),
    "reasoning": "detailed explanation of why this agent does/doesn't have the knowledge",
    "key_information_found": ["list", "of", "key", "pieces", "found"]
}}
"""

        try:
            from litellm import acompletion

            print(f'  Evaluating {agent_name} message with LLM...')
            response = await acompletion(
                model=self.model_provider._get_agenerate_model_name(),
                messages=[{'role': 'user', 'content': evaluation_prompt}],
                temperature=0.1,
            )
            print(f'  LLM response received for {agent_name}')

            response_content = response.choices[0].message.content

            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())

                def normalize_percentage(value):
                    if value is None:
                        return 0.0
                    if value <= 1.0:
                        return value * 100
                    return value

                llm_result = {
                    'agent': agent_name,
                    'has_knowledge': result.get('has_knowledge', False),
                    'completeness': normalize_percentage(
                        result.get('completeness', 0.0)
                    ),
                    'accuracy': normalize_percentage(result.get('accuracy', 0.0)),
                    'confidence': normalize_percentage(result.get('confidence', 0.0)),
                    'reasoning': result.get('reasoning', ''),
                    'key_information': result.get('key_information_found', []),
                    'message_snippet': message_text[:100] + '...'
                    if len(message_text) > 100
                    else message_text,
                }
                self._evaluation_cache[cache_key] = llm_result
                return llm_result
            else:
                return self._fallback_analysis(agent_name, message_text)

        except Exception as e:
            print(f'LLM evaluation failed for {agent_name}: {e}')
            return self._fallback_analysis(agent_name, message_text)

    def _fallback_analysis(self, agent_name: str, message_text: str) -> dict[str, Any]:
        """Fallback keyword-based analysis if LLM evaluation fails."""
        knowledge_keywords = ['gpt-5', 'gpt5', 'may 2025', 'release']

        message_lower = message_text.lower()
        keywords_found = [kw for kw in knowledge_keywords if kw in message_lower]

        has_knowledge = len(keywords_found) >= 2

        return {
            'agent': agent_name,
            'has_knowledge': has_knowledge,
            'completeness': len(keywords_found) * 25.0,
            'accuracy': 80.0 if has_knowledge else 0.0,
            'confidence': 60.0,
            'reasoning': f'Fallback keyword analysis. Found keywords: {keywords_found}',
            'key_information': keywords_found,
            'message_snippet': message_text[:100] + '...'
            if len(message_text) > 100
            else message_text,
        }

    def __call__(
        self, turn_number: int, messages: list[tuple[str, Message]]
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        """Synchronous evaluation using fallback analysis."""
        knowledge_analysis = []

        for source, message in messages:
            if source == 'Environment':
                continue

            message_text = message.to_natural_language()
            analysis = self._fallback_analysis(source, message_text)

            if analysis['has_knowledge']:
                knowledge_analysis.append(analysis)
                if source != self.knowledge_source:
                    self.agents_with_knowledge.add(source)

        total_target_agents = (
            len(self.target_agents)
            if self.target_agents
            else len(
                set(
                    msg[0]
                    for msg in messages
                    if msg[0] != 'Environment' and msg[0] != self.knowledge_source
                )
            )
        )
        agents_reached = len(self.agents_with_knowledge)

        if total_target_agents > 0:
            propagation_rate = agents_reached / total_target_agents
        else:
            propagation_rate = 0.0

        if knowledge_analysis:
            accuracy_rate = sum(a['accuracy'] for a in knowledge_analysis) / len(
                knowledge_analysis
            )
            if accuracy_rate <= 1.0:
                accuracy_rate = accuracy_rate * 100
        else:
            accuracy_rate = 0.0

        analysis_details = {
            'total_knowledge_mentions': len(knowledge_analysis),
            'agents_with_knowledge': list(self.agents_with_knowledge),
            'propagation_rate': propagation_rate,
            'accuracy_rate': accuracy_rate,
            'knowledge_analysis': knowledge_analysis,
        }

        accuracy_for_score = (
            accuracy_rate / 100 if accuracy_rate > 1.0 else accuracy_rate
        )
        overall_score = (propagation_rate * 0.6 + accuracy_for_score * 0.4) * 10

        comments = self._generate_evaluation_comments(analysis_details)

        return [('environment', (('knowledge_propagation', overall_score), comments))]

    async def __acall__(
        self, turn_number: int, messages: list[tuple[str, Message]]
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        """Async version using LLM evaluation - analyzes both speaking and listening."""

        knowledge_analysis = []

        knowledge_messages = []

        for source, message in messages:
            if source == 'Environment':
                continue

            message_text = message.to_natural_language()
            analysis = await self._analyze_message_for_knowledge(source, message_text)

            if analysis['has_knowledge']:
                knowledge_analysis.append(analysis)
                knowledge_messages.append((source, message_text, analysis))
                if source != self.knowledge_source:
                    self.agents_with_knowledge.add(source)

        await self._analyze_knowledge_reception(
            messages, knowledge_messages, knowledge_analysis
        )

        total_target_agents = (
            len(self.target_agents)
            if self.target_agents
            else len(
                set(
                    msg[0]
                    for msg in messages
                    if msg[0] != 'Environment' and msg[0] != self.knowledge_source
                )
            )
        )
        agents_reached = len(self.agents_with_knowledge)

        if total_target_agents > 0:
            propagation_rate = agents_reached / total_target_agents
        else:
            propagation_rate = 0.0

        if knowledge_analysis:
            accuracy_rate = sum(a['accuracy'] for a in knowledge_analysis) / len(
                knowledge_analysis
            )
            if accuracy_rate <= 1.0:
                accuracy_rate = accuracy_rate * 100
        else:
            accuracy_rate = 0.0

        analysis_details = {
            'total_knowledge_mentions': len(knowledge_analysis),
            'agents_with_knowledge': list(self.agents_with_knowledge),
            'propagation_rate': propagation_rate,
            'accuracy_rate': accuracy_rate,
            'knowledge_analysis': knowledge_analysis,
        }

        accuracy_for_score = (
            accuracy_rate / 100 if accuracy_rate > 1.0 else accuracy_rate
        )
        overall_score = (propagation_rate * 0.6 + accuracy_for_score * 0.4) * 10

        comments = self._generate_evaluation_comments(analysis_details)

        return [('environment', (('knowledge_propagation', overall_score), comments))]

    def _generate_evaluation_comments(self, details: dict[str, Any]) -> str:
        """Generate human-readable evaluation comments."""
        comments = []

        comments.append('Knowledge Propagation Analysis:')
        comments.append(f"- Knowledge mentions: {details['total_knowledge_mentions']}")
        comments.append(f"- Agents reached: {len(details['agents_with_knowledge'])}")
        comments.append(f"- Propagation rate: {details['propagation_rate']:.1%}")
        comments.append(f"- Accuracy rate: {details['accuracy_rate']:.1f}%")

        if details['agents_with_knowledge']:
            comments.append(
                f"- Agents with knowledge: {', '.join(details['agents_with_knowledge'])}"
            )
        else:
            comments.append('- No knowledge propagation detected')

        if details['knowledge_analysis']:
            comments.append('\nDetailed LLM Analysis:')
            for i, analysis in enumerate(details['knowledge_analysis'][:3], 1):
                comments.append(f"{i}. {analysis['agent']}:")
                comments.append(
                    f"   Has Knowledge: {analysis.get('has_knowledge', 'N/A')}"
                )
                comments.append(
                    f"   Completeness: {analysis.get('completeness', 0):.1f}%"
                )
                comments.append(f"   Accuracy: {analysis.get('accuracy', 0):.1f}%")
                comments.append(f"   Confidence: {analysis.get('confidence', 0):.1f}%")
                if analysis.get('reasoning'):
                    comments.append(f"   Reasoning: {analysis['reasoning']}")
                if analysis.get('key_information'):
                    comments.append(f"   Key Info: {analysis['key_information']}")
                if analysis.get('message_snippet'):
                    comments.append(f"   Message: \"{analysis['message_snippet']}\"")

        return '\n'.join(comments)

    async def _analyze_knowledge_reception(
        self, messages, knowledge_messages, knowledge_analysis
    ):
        """Analyze if agents received knowledge by listening to others."""

        conversation_flow = []
        for source, message in messages:
            if source != 'Environment':
                conversation_flow.append((source, message.to_natural_language()))

        for knowledge_source, knowledge_text, knowledge_info in knowledge_messages:
            potential_listeners = set(
                msg[0]
                for msg in messages
                if msg[0] != 'Environment' and msg[0] != knowledge_source
            )

            for listener in potential_listeners:
                if listener not in self.agents_with_knowledge:
                    reception_analysis = (
                        await self._analyze_knowledge_reception_for_agent(
                            listener,
                            knowledge_source,
                            knowledge_text,
                            conversation_flow,
                        )
                    )

                    if reception_analysis.get('received_knowledge', False):
                        knowledge_analysis.append(reception_analysis)
                        self.agents_with_knowledge.add(listener)

    async def _analyze_knowledge_reception_for_agent(
        self, listener, knowledge_source, knowledge_text, conversation_flow
    ):
        """Analyze if a specific agent received knowledge by listening."""

        knowledge_index = -1
        for i, (speaker, text) in enumerate(conversation_flow):
            if speaker == knowledge_source and knowledge_text[:50] in text:
                knowledge_index = i
                break

        if knowledge_index == -1:
            return {'received_knowledge': False}

        listener_responses = []
        for i in range(knowledge_index + 1, len(conversation_flow)):
            speaker, text = conversation_flow[i]
            if speaker == listener:
                listener_responses.append(text)

        if not listener_responses:
            return {
                'agent': listener,
                'received_knowledge': True,
                'completeness': 80.0,
                'accuracy': 90.0,
                'confidence': 70.0,
                'reasoning': f'{listener} was present when {knowledge_source} shared knowledge',
                'key_information': ['GPT-5', 'heard from ' + knowledge_source],
                'message_snippet': f'(Heard from {knowledge_source})',
            }

        response_text = ' '.join(listener_responses)

        reception_prompt = f"""
Analyze if {listener} received and understood knowledge from {knowledge_source}.

KNOWLEDGE SHARED BY {knowledge_source}:
{knowledge_text}

{listener}'S SUBSEQUENT RESPONSES:
{response_text}

Determine if {listener}'s responses show they received and understood the shared information.
Look for:
1. Acknowledgments like "thanks", "got it", "I'll share this"
2. References to GPT-5 or release information
3. Plans to pass information to others
4. Any indication they heard and processed the information

Respond ONLY with a JSON object:
{{
    "received_knowledge": true/false,
    "completeness": 0-100,
    "accuracy": 0-100,
    "confidence": 0-100,
    "reasoning": "explanation",
    "key_information_found": ["list", "of", "indicators"]
}}
"""

        try:
            from litellm import acompletion

            response = await acompletion(
                model=self.model_provider._get_agenerate_model_name(),
                messages=[{'role': 'user', 'content': reception_prompt}],
                temperature=0.1,
            )

            response_content = response.choices[0].message.content
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)

            if json_match:
                result = json.loads(json_match.group())
                return {
                    'agent': listener,
                    'received_knowledge': result.get('received_knowledge', False),
                    'completeness': result.get('completeness', 0.0),
                    'accuracy': result.get('accuracy', 0.0),
                    'confidence': result.get('confidence', 0.0),
                    'reasoning': result.get('reasoning', ''),
                    'key_information': result.get('key_information_found', []),
                    'message_snippet': response_text[:100] + '...'
                    if len(response_text) > 100
                    else response_text,
                }
            else:
                return {'received_knowledge': False}

        except Exception as e:
            print(f'Reception analysis failed for {listener}: {e}')
            return {'received_knowledge': False}
