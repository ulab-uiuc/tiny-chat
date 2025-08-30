import os
from typing import Any
from urllib.parse import urlparse

from litellm import acompletion, completion

from tiny_chat.config import ModelProviderConfig


def prepare_model_config_from_provider(
    config: ModelProviderConfig,
) -> tuple[str | None, str | None, str]:
    """
    from ModelProviderConfig
    return: (api_base, api_key, effective_model_name)
    """
    api_base = config.api_base
    api_key = config.api_key
    effective_model = config.name

    if not api_key:
        provider_type = config.type.upper()
        env_key = f'{provider_type}_API_KEY'
        api_key = os.getenv(env_key)

    return api_base, api_key, effective_model


def _ensure_v1(url: str) -> str:
    """Ensure URL ends with /v1"""
    if not url:
        return 'http:///v1'

    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url

    if url.endswith('/v1'):
        return url
    trimmed = url[:-1] if url.endswith('/') else url
    return trimmed + '/v1'


def _parse_vllm_config(model_name: str) -> tuple[str | None, str | None, str]:
    """Parse vLLM configuration"""
    payload = model_name[len('vllm://') :]
    if '@' not in payload:
        raise ValueError(
            "vllm:// requires '<model>@<base_url>' or '<base_url>@<model>'"
        )
    a, b = payload.split('@', 1)
    if urlparse(a).scheme or a.startswith(
        ('http://', 'https://', 'localhost', '127.0.0.1')
    ):
        base, model = a, b
    else:
        model, base = a, b
    api_base = _ensure_v1(base)
    api_key = os.getenv('VLLM_API_KEY')  # optional
    return api_base, api_key, model


def _parse_together_config(model_name: str) -> tuple[str | None, str | None, str]:
    """Parse Together configuration"""
    model = model_name[len('together://') :]
    api_base = 'https://api.together.xyz/v1'
    api_key = os.getenv('TOGETHER_API_KEY')
    return api_base, api_key, model


def _parse_custom_config(
    model_name: str, is_legacy: bool = False
) -> tuple[str | None, str | None, str]:
    """Parse custom configuration"""
    prefix = 'custom/' if is_legacy else 'custom://'
    payload = model_name[len(prefix) :]
    if '@' not in payload:
        raise ValueError(f"{prefix} requires '<model>@<base_url>'")
    model, base = payload.split('@', 1)
    api_base = base
    api_key = os.getenv('CUSTOM_API_KEY', 'EMPTY')

    if is_legacy:
        effective_model = (
            f'openai/{model}' if not model.startswith('openai/') else model
        )
    else:
        effective_model = model

    return api_base, api_key, effective_model


def prepare_model_config_from_name(
    model_name: str,
) -> tuple[str | None, str | None, str]:
    """
    return: (api_base, api_key, effective_model_name)

    - OpenAI default:                    "gpt-4o-mini"
    - vLLM (OpenAI-compatible):          "vllm://<model>@<base>" or "vllm://<base>@<model>"
    - Together (OpenAI-compatible):      "together://<model>"
    - Custom OpenAI-compatible proxy:    "custom://<model>@<base>" or legacy "custom/<model>@<base>"
    """
    # vLLM
    if model_name.startswith('vllm://'):
        return _parse_vllm_config(model_name)

    # Together
    if model_name.startswith('together://'):
        return _parse_together_config(model_name)

    # Custom proxy (new scheme)
    if model_name.startswith('custom://'):
        return _parse_custom_config(model_name, is_legacy=False)

    # Custom proxy (legacy form)
    if model_name.startswith('custom/'):
        return _parse_custom_config(model_name, is_legacy=True)

    # default OpenAI
    return None, None, model_name


def build_model_name_from_config(config: ModelProviderConfig) -> str:
    """
    from ModelProviderConfig
    """
    if config.type == 'custom' and config.api_base:
        # Custom endpoint format: custom://model@url
        return f'custom://{config.name}@{config.api_base}'
    elif config.type == 'vllm' and config.api_base:
        # vLLM endpoint format: vllm://model@url
        return f'vllm://{config.name}@{config.api_base}'
    elif config.type == 'together':
        # Together format: together://model
        return f'together://{config.name}'
    else:
        return config.name


async def call_model_async(
    model_name: str,
    messages: list[dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> str:
    api_base, api_key, effective_model = prepare_model_config_from_name(model_name)

    response = await acompletion(
        model=effective_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        drop_params=True,
        base_url=api_base,
        api_key=api_key,
        **kwargs,
    )

    content = response.choices[0].message.content
    assert isinstance(content, str)
    return content


def call_model_sync(
    model_name: str,
    messages: list[dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> str:
    api_base, api_key, effective_model = prepare_model_config_from_name(model_name)

    response = completion(
        model=effective_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        drop_params=True,
        base_url=api_base,
        api_key=api_key,
        **kwargs,
    )

    content = response.choices[0].message.content
    assert isinstance(content, str)
    return content
