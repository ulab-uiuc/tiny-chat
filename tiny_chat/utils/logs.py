from typing import Any

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from tiny_chat.profiles import BaseAgentProfile


class BaseEpisodeLog(BaseModel):
    environment: str
    agents: list[str]
    tag: str | None = Field(default='')
    models: list[str] = Field(default_factory=list)
    messages: list[list[tuple[str, str, str]]]
    reasoning: str = Field(default='')
    rewards: list[tuple[float, dict[str, float]] | float]
    rewards_prompt: str = Field(default='')

    @model_validator(mode='after')
    def validate_multi_agent_consistency(self) -> Self:
        agent_number = len(self.agents)

        assert (
            len(self.rewards) == agent_number
        ), f'Number of agents in rewards {len(self.rewards)} and agents {agent_number} do not match'

        if self.models and len(self.models) > 0:
            assert (
                len(self.models) == agent_number
            ), f'Number of models {len(self.models)} does not match number of agents {agent_number}'

        assert agent_number > 0, 'At least one agent must be specified'

        assert len(set(self.agents)) == agent_number, 'Agent names must be unique'

        for idx, reward in enumerate(self.rewards):
            if isinstance(reward, tuple):
                assert (
                    len(reward) >= 1
                ), f'Reward tuple for agent {idx} must have at least one element (the score)'
                assert isinstance(
                    reward[0], int | float
                ), f'First element of reward tuple for agent {idx} must be a number'
                if len(reward) > 1:
                    assert isinstance(
                        reward[1], dict
                    ), f'Second element of reward tuple for agent {idx} must be a dict'
            else:
                assert isinstance(
                    reward, int | float
                ), f'Reward for agent {idx} must be a number or tuple'

        return self

    def render_for_humans(self) -> tuple[list[BaseAgentProfile], list[str]]:
        agent_profiles = [
            BaseAgentProfile(pk=uuid_str, first_name=uuid_str, last_name='')
            for uuid_str in self.agents
        ]
        messages_and_rewards = []

        for idx, turn in enumerate(self.messages):
            turn_messages = self._process_turn_messages(turn, idx)
            messages_and_rewards.append('\n'.join(turn_messages))

        messages_and_rewards.append(f'The reasoning is:\n{self.reasoning}')

        rewards_text = self._format_rewards()
        messages_and_rewards.append(rewards_text)

        return agent_profiles, messages_and_rewards

    def _process_turn_messages(
        self, turn: list[tuple[str, str, str]], turn_idx: int
    ) -> list[str]:
        messages_in_this_turn = []

        if turn_idx == 0:
            min_messages = min(len(turn), len(self.agents))
            assert (
                len(turn) >= min_messages
            ), f'The first turn should have at least {min_messages} environment messages for {len(self.agents)} agents'

            for i in range(min_messages):
                if i < len(turn):
                    messages_in_this_turn.append(
                        f"{turn[i][1]}'s perspective (i.e., what {turn[i][1]} knows before the episode starts): {turn[i][2]}"
                    )

        for sender, receiver, message in turn:
            if receiver == 'Environment' and sender != 'Environment':
                if 'did nothing' not in message:
                    prefix = f'{sender} ' if 'said:' in message else f'{sender}: '
                    messages_in_this_turn.append(f'{prefix}{message}')
            elif sender == 'Environment':
                messages_in_this_turn.append(message)

        return messages_in_this_turn

    def _format_rewards(self) -> str:
        rewards_text = 'The rewards are:\n'
        for idx, (agent_id, reward) in enumerate(
            zip(self.agents, self.rewards, strict=True)
        ):
            agent_name = f'Agent {idx + 1}' if len(self.agents) > 1 else 'Agent'

            if isinstance(reward, tuple):
                main_score = reward[0]
                details = (
                    reward[1] if len(reward) > 1 and isinstance(reward[1], dict) else {}
                )
                rewards_text += f'{agent_name} ({agent_id}): {main_score}'
                if details:
                    detail_strs = [f'{k}: {v}' for k, v in details.items()]
                    rewards_text += f" (Details: {', '.join(detail_strs)})"
                rewards_text += '\n'
            else:
                rewards_text += f'{agent_name} ({agent_id}): {reward}\n'

        return rewards_text.rstrip()

    def get_agent_score(
        self, agent_id: str
    ) -> float | tuple[float, dict[str, float]] | None:
        try:
            agent_index = self.agents.index(agent_id)
            return self.rewards[agent_index]
        except (ValueError, IndexError):
            return None

    def get_all_scores(self) -> dict[str, float | tuple[float, dict[str, float]]]:
        return dict(zip(self.agents, self.rewards, strict=True))

    def get_average_score(self) -> float:
        if not self.rewards:
            return 0.0

        total_score = 0.0
        for reward in self.rewards:
            if isinstance(reward, tuple):
                total_score += reward[0]
            else:
                total_score += reward

        return total_score / len(self.rewards)

    def get_conversation_turns(self) -> int:
        return len(self.messages)

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            'environment': self.environment,
            'agent_count': len(self.agents),
            'agents': self.agents,
            'models': self.models,
            'turn_count': self.get_conversation_turns(),
            'average_score': self.get_average_score(),
            'individual_scores': self.get_all_scores(),
            'tag': self.tag,
            'reasoning_summary': (
                self.reasoning[:200] + '...'
                if len(self.reasoning) > 200
                else self.reasoning
            ),
        }


class EpisodeLog(BaseEpisodeLog):
    pass
