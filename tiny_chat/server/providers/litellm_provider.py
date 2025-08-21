from .base import BaseModelProvider


class LiteLLMProvider(BaseModelProvider):
    def _get_agenerate_model_name(self) -> str:
        if self.config.type == 'custom' and self.config.api_base:
            # Custom endpoint format: custom://model@url
            return f'custom://{self.config.name}@{self.config.api_base}'
        elif self.config.type == 'vllm' and self.config.api_base:
            # vLLM endpoint format: vllm://model@url
            return f'vllm://{self.config.name}@{self.config.api_base}'
        elif self.config.type == 'together':
            # Together format: together://model
            return f'together://{self.config.name}'
        else:
            return self.config.name

    async def check_health(self) -> bool:
        try:
            from tiny_chat.generator.output_parsers import StrOutputParser

            await self.agenerate(
                template='Hello',
                input_values={},
                output_parser=StrOutputParser(),
                temperature=0.1,
            )
            return True
        except Exception:
            return False
