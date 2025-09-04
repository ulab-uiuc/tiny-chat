from .base import BaseModelProvider


class LiteLLMProvider(BaseModelProvider):
    def _get_agenerate_model_name(self) -> str:
        from .utils import build_model_name_from_config

        return build_model_name_from_config(self.config)

    async def check_health(self) -> bool:
        try:
            from .output_parsers import StrOutputParser

            await self.agenerate(
                template="Hello",
                input_values={},
                output_parser=StrOutputParser(),
                temperature=0.1,
            )
            return True
        except Exception:
            return False
