import logging
from typing import Any

import httpx

from ..config import ModelProviderConfig
from .base import BaseModelProvider

logger = logging.getLogger(__name__)


class RequestHandler:
    def __init__(self, config: dict[str, Any]):
        self.config = config

    async def process(self, prompt: str, **kwargs) -> dict[str, Any]:
        raise NotImplementedError


class ResponseParser:

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def parse(self, raw_response: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class WorkflowProvider(BaseModelProvider):
    _request_handlers: dict[str, type[RequestHandler]] = {}
    _response_parsers: dict[str, type[ResponseParser]] = {}

    def __init__(self, config: ModelProviderConfig):
        super().__init__(config)

        custom_config = getattr(config, "custom_config", {})
        self.request_handler_type = custom_config.get("request_handler", "default")
        self.response_parser_type = custom_config.get("response_parser", "default")

        self.request_handler = self._create_request_handler(custom_config)
        self.response_parser = self._create_response_parser(custom_config)

    @classmethod
    def register_request_handler(cls, name: str, handler_class: type[RequestHandler]):
        cls._request_handlers[name] = handler_class
        logger.info(f"Registered request handler: {name}")

    @classmethod
    def register_response_parser(cls, name: str, parser_class: type[ResponseParser]):
        cls._response_parsers[name] = parser_class
        logger.info(f"Registered response parser: {name}")

    def _create_request_handler(self, config: dict[str, Any]) -> RequestHandler:
        handler_class = self._request_handlers.get(self.request_handler_type)
        if not handler_class:
            handler_class = DefaultRequestHandler

        return handler_class(config)

    def _create_response_parser(self, config: dict[str, Any]) -> ResponseParser:
        parser_class = self._response_parsers.get(self.response_parser_type)
        if not parser_class:
            parser_class = DefaultResponseParser

        return parser_class(config)

    def is_tinychat_mode(self) -> bool:
        return self.config.custom_config.get("use_tinychat_agenerate", True)

    async def custom_workflow_generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        try:
            raw_response = await self.request_handler.process(
                prompt, temperature=temperature, max_tokens=max_tokens, **kwargs
            )
            return self.response_parser.parse(raw_response)
        except Exception as e:
            logger.error(f"WorkflowProvider custom generation failed: {e}")
            raise RuntimeError(f"Custom generation failed: {e}") from e

    def _get_agenerate_model_name(self) -> str:
        """Convert provider config to agenerate-compatible model name

        For WorkflowProvider, we use a special format to indicate
        it's a complex workflow that can't be handled by agenerate directly.
        """
        # For workflow providers, we could fall back to a simpler model
        # or indicate that this is a complex workflow
        if hasattr(self.config, "custom_config") and self.config.custom_config:
            # Try to extract a fallback model if configured
            fallback = self.config.custom_config.get("fallback_model")
            if fallback:
                return fallback

        # Return a special identifier that indicates this is a workflow
        return f"workflow://{self.config.name}"

    async def check_health(self) -> bool:
        try:
            await self.generate(prompt="test", max_tokens=1)
            return True
        except Exception:
            return False


class DefaultRequestHandler(RequestHandler):
    async def process(self, prompt: str, **kwargs) -> dict[str, Any]:
        endpoint = self.config.get("endpoint", "")
        headers = self.config.get("headers", {})

        request_data = {"prompt": prompt, **kwargs}

        async with httpx.AsyncClient() as client:
            response = await client.post(
                endpoint, json=request_data, headers=headers, timeout=30.0
            )
            response.raise_for_status()

        return response.json()


class DefaultResponseParser(ResponseParser):
    def parse(self, raw_response: dict[str, Any]) -> dict[str, Any]:
        content = (
            raw_response.get("output")
            or raw_response.get("text")
            or raw_response.get("response")
            or raw_response.get("content")
            or str(raw_response)
        )

        return {
            "content": content,
            "usage": raw_response.get("usage"),
            "model": raw_response.get("model", "workflow"),
            "metadata": {"provider": "workflow", "raw_response": raw_response},
        }


# example: for reward model eval
class SFTRewardRequestHandler(RequestHandler):
    async def process(self, prompt: str, **kwargs) -> dict[str, Any]:
        sft_endpoint = self.config.get("sft_endpoint")
        reward_endpoint = self.config.get("reward_endpoint")
        headers = self.config.get("headers", {})
        num_candidates = kwargs.get("num_candidates", 10)

        async with httpx.AsyncClient() as client:
            sft_response = await client.post(
                sft_endpoint,
                json={
                    "prompt": prompt,
                    "num_return_sequences": num_candidates,
                    "temperature": kwargs.get("temperature", 0.8),
                    "max_length": kwargs.get("max_tokens", 100),
                },
                headers=headers,
            )
            sft_data = sft_response.json()
            candidates = sft_data.get("generations", [])

        async with httpx.AsyncClient() as client:
            reward_response = await client.post(
                reward_endpoint,
                json={"prompt": prompt, "responses": candidates},
                headers=headers,
            )
            reward_data = reward_response.json()
            scores = reward_data.get("scores", [])

        return {"candidates": candidates, "scores": scores, "prompt": prompt}


class SFTRewardResponseParser(ResponseParser):
    def parse(self, raw_response: dict[str, Any]) -> dict[str, Any]:
        candidates = raw_response.get("candidates", [])
        scores = raw_response.get("scores", [])

        if not candidates or not scores:
            raise ValueError("No valid candidates or scores received")

        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best_response = candidates[best_idx]
        best_score = scores[best_idx]

        return {
            "content": best_response,
            "usage": {
                "candidates_generated": len(candidates),
                "total_tokens": sum(len(c.split()) for c in candidates),
            },
            "model": "sft-reward",
            "metadata": {
                "best_score": best_score,
                "all_scores": scores,
                "all_candidates": candidates,
                "selection_index": best_idx,
                "provider": "workflow_sft_reward",
            },
        }


WorkflowProvider.register_request_handler("sft_reward", SFTRewardRequestHandler)
WorkflowProvider.register_response_parser("sft_reward", SFTRewardResponseParser)
