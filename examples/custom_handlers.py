import json
from typing import Any, Dict, List

import httpx

from tiny_chat.server.providers.base import ModelResponse
from tiny_chat.server.providers.workflow_provider import (
    RequestHandler,
    ResponseParser,
    WorkflowProvider,
)


class CustomFormatRequestHandler(RequestHandler):
    async def process(self, prompt: str, **kwargs) -> Dict[str, Any]:
        endpoint = self.config.get('endpoint')
        headers = self.config.get('headers', {})

        request_data = {
            'input_text': prompt,
            'generation_params': {
                'temperature': kwargs.get('temperature', 0.7),
                'max_new_tokens': kwargs.get('max_tokens', 100),
                'do_sample': True,
                'top_p': kwargs.get('top_p', 0.9),
            },
            'output_format': 'json',
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                endpoint, json=request_data, headers=headers, timeout=60.0
            )
            response.raise_for_status()

        return response.json()


class CustomFormatResponseParser(ResponseParser):
    def parse(self, raw_response: Dict[str, Any]) -> ModelResponse:
        generated_text = raw_response.get('generated_text', '')
        generation_info = raw_response.get('generation_info', {})

        usage = {
            'prompt_tokens': generation_info.get('input_length', 0),
            'completion_tokens': generation_info.get('output_length', 0),
            'total_tokens': generation_info.get('total_length', 0),
        }

        return ModelResponse(
            content=generated_text,
            usage=usage,
            model=generation_info.get('model_name', 'custom'),
            finish_reason=generation_info.get('finish_reason', 'stop'),
            metadata={
                'provider': 'flexible_custom',
                'generation_time': generation_info.get('generation_time'),
                'raw_response': raw_response,
            },
        )


class EnsembleRequestHandler(RequestHandler):
    async def process(self, prompt: str, **kwargs) -> Dict[str, Any]:
        model_endpoints = self.config.get('model_endpoints', [])
        headers = self.config.get('headers', {})

        responses = []

        async with httpx.AsyncClient() as client:
            tasks = []
            for endpoint_config in model_endpoints:
                request_data = {
                    'prompt': prompt,
                    'temperature': kwargs.get('temperature', 0.7),
                    'max_tokens': kwargs.get('max_tokens', 100),
                }

                task = client.post(
                    endpoint_config['url'],
                    json=request_data,
                    headers={**headers, **endpoint_config.get('headers', {})},
                    timeout=30.0,
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f'Model {i} failed: {result}')
                    continue

                try:
                    response_data = result.json()
                    responses.append(
                        {
                            'model': model_endpoints[i]['name'],
                            'response': response_data.get('text', ''),
                            'confidence': response_data.get('confidence', 0.5),
                        }
                    )
                except Exception as e:
                    print(f'Failed to parse response from model {i}: {e}')

        return {'responses': responses, 'prompt': prompt}


class EnsembleResponseParser(ResponseParser):
    def parse(self, raw_response: Dict[str, Any]) -> ModelResponse:
        responses = raw_response.get('responses', [])

        if not responses:
            raise ValueError('No valid responses from ensemble models')

        ensemble_method = self.config.get('ensemble_method', 'highest_confidence')

        if ensemble_method == 'highest_confidence':
            best_response = max(responses, key=lambda r: r.get('confidence', 0))
            content = best_response['response']

        elif ensemble_method == 'majority_vote':
            content = max(
                set(r['response'] for r in responses),
                key=lambda x: sum(1 for r in responses if r['response'] == x),
            )

        elif ensemble_method == 'concatenate':
            content = '\n---\n'.join(
                [f"Model {r['model']}: {r['response']}" for r in responses]
            )

        else:
            content = responses[0]['response']

        return ModelResponse(
            content=content,
            usage={'models_called': len(responses)},
            model='ensemble',
            metadata={
                'provider': 'flexible_ensemble',
                'ensemble_method': ensemble_method,
                'all_responses': responses,
                'selected_model': responses[0]['model'] if responses else None,
            },
        )


def register_custom_handlers():
    WorkflowProvider.register_request_handler(
        'custom_format', CustomFormatRequestHandler
    )
    WorkflowProvider.register_response_parser(
        'custom_format', CustomFormatResponseParser
    )

    WorkflowProvider.register_request_handler('ensemble', EnsembleRequestHandler)
    WorkflowProvider.register_response_parser('ensemble', EnsembleResponseParser)


# 使用示例
async def example_usage():
    """使用示例"""
    from tiny_chat.server.config import ModelProviderConfig
    from tiny_chat.server.providers import ModelProviderFactory

    # 注册自定义处理器
    register_custom_handlers()

    # 创建配置
    config = ModelProviderConfig(
        name='my-custom-model',
        type='flexible',
        custom_config={
            'request_handler': 'custom_format',
            'response_parser': 'custom_format',
            'endpoint': 'https://your-api.com/generate',
            'headers': {
                'Authorization': 'Bearer your-token',
                'Content-Type': 'application/json',
            },
        },
    )

    # 创建提供者
    provider = ModelProviderFactory.create_provider(config)

    # 使用
    response = await provider.generate(
        prompt='Tell me a story', temperature=0.8, max_tokens=200
    )

    print(f'Generated: {response.content}')
    print(f'Usage: {response.usage}')
    print(f'Metadata: {response.metadata}')


if __name__ == '__main__':
    import asyncio

    asyncio.run(example_usage())
