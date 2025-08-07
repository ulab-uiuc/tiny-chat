import os
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, cast
from dataclasses import dataclass
from enum import Enum

# LiteLLM imports
try:
    from litellm import acompletion
    from litellm.litellm_core_utils.get_supported_openai_params import (
        get_supported_openai_params,
    )
    from litellm.utils import supports_response_schema
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

# vLLM imports
try:
    from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams, LLM
    from vllm.utils import random_uuid
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

class LLMBackend(Enum):
    LITELLM = "litellm"
    VLLM = "vllm"


@dataclass
class LLMConfig:
    """LLM配置类"""
    backend: LLMBackend
    model_name: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    # vLLM specific
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.7  # 降低默认值以避免OOM
    max_model_len: Optional[int] = None
    # Custom parameters
    custom_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.custom_params is None:
            self.custom_params = {}


class BaseLLMGenerator(ABC):
    """LLM生成器基类"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """生成文本"""
        pass
    
    @abstractmethod
    async def generate_structured(
        self,
        messages: List[Dict[str, str]],
        schema: Dict[str, Any],
        **kwargs
    ) -> str:
        pass
    
    @abstractmethod
    def supports_structured_output(self) -> bool:
        pass


class LiteLLMGenerator(BaseLLMGenerator):
    """LiteLLM生成器"""
    
    def __init__(self, config: LLMConfig):
        if not LITELLM_AVAILABLE:
            raise ImportError("LiteLLM is not available. Please install it with: pip install litellm")
        
        super().__init__(config)
        self.model_name = self._process_model_name()
    
    def _process_model_name(self) -> str:
        if self.config.model_name.startswith('custom'):
            if '@' in self.config.model_name:
                model_name, base_url = self.config.model_name.split('@', 1)
                self.config.base_url = base_url
                self.config.api_key = os.environ.get('CUSTOM_API_KEY', 'EMPTY')
                return model_name.replace('custom/', 'openai/')
        return self.config.model_name
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """使用LiteLLM生成文本"""
        params = {
            'model': self.model_name,
            'messages': messages,
            'temperature': kwargs.get('temperature', self.config.temperature),
            'drop_params': True,
        }
        
        # 添加可选参数
        if self.config.base_url:
            params['base_url'] = self.config.base_url
        if self.config.api_key:
            params['api_key'] = self.config.api_key
        if response_format:
            params['response_format'] = response_format
        if self.config.max_tokens:
            params['max_tokens'] = self.config.max_tokens
        
        # 合并自定义参数
        params.update(self.config.custom_params)
        params.update(kwargs)
        
        response = await acompletion(**params)
        result = response.choices[0].message.content
        assert isinstance(result, str)
        return result
    
    async def generate_structured(
        self,
        messages: List[Dict[str, str]],
        schema: Dict[str, Any],
        **kwargs
    ) -> str:
        """生成结构化输出"""
        if not self.supports_structured_output():
            raise ValueError(f"Model {self.model_name} does not support structured output")
        
        return await self.generate(
            messages=messages,
            response_format=schema,
            **kwargs
        )
    
    def supports_structured_output(self) -> bool:
        """检查是否支持结构化输出"""
        if self.config.base_url:
            return True  # 假设自定义端点支持
        
        try:
            params = get_supported_openai_params(model=self.model_name)
            return (params is not None and 
                    'response_format' in params and 
                    supports_response_schema(model=self.model_name))
        except:
            return False


class VLLMGenerator(BaseLLMGenerator):
    """vLLM生成器"""
    
    def __init__(self, config: LLMConfig):
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not available. Please install it with: pip install vllm")
        
        super().__init__(config)
        self.engine = None
        self._initialize_engine()
    
    def _initialize_engine(self):
        """初始化vLLM引擎"""
        engine_args = AsyncEngineArgs(
            model=self.config.model_name,
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_model_len=self.config.max_model_len,
            **self.config.custom_params
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    def _create_sampling_params(self, **kwargs) -> SamplingParams:
        """创建采样参数"""
        return SamplingParams(
            temperature=kwargs.get('temperature', self.config.temperature),
            top_p=kwargs.get('top_p', self.config.top_p),
            max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
            frequency_penalty=kwargs.get('frequency_penalty', self.config.frequency_penalty),
            presence_penalty=kwargs.get('presence_penalty', self.config.presence_penalty),
        )
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """将消息格式转换为提示文本"""
        prompt = ""
        for message in messages:
            role = message['role']
            content = message['content']
            if role == 'system':
                prompt += f"System: {content}\n\n"
            elif role == 'user':
                prompt += f"User: {content}\n\n"
            elif role == 'assistant':
                prompt += f"Assistant: {content}\n\n"
        prompt += "Assistant: "
        return prompt
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """使用vLLM生成文本"""
        if self.engine is None:
            raise RuntimeError("vLLM engine is not initialized")
        
        prompt = self._messages_to_prompt(messages)
        sampling_params = self._create_sampling_params(**kwargs)
        
        # 如果需要JSON格式，添加到提示中
        if response_format and response_format.get('type') == 'json_object':
            prompt += "\n\nPlease respond with a valid JSON object."
        
        request_id = random_uuid()
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
        
        if final_output is None:
            raise RuntimeError("No output generated")
        
        return final_output.outputs[0].text.strip()
    
    async def generate_structured(
        self,
        messages: List[Dict[str, str]],
        schema: Dict[str, Any],
        **kwargs
    ) -> str:
        """生成结构化输出（通过提示引导）"""
        # vLLM本身不直接支持结构化输出，通过提示引导
        structured_prompt = {
            'role': 'system',
            'content': f"Please respond with a valid JSON object that follows this schema: {json.dumps(schema)}"
        }
        
        enhanced_messages = [structured_prompt] + messages
        return await self.generate(
            messages=enhanced_messages,
            response_format={'type': 'json_object'},
            **kwargs
        )
    
    def supports_structured_output(self) -> bool:
        """vLLM通过提示引导支持结构化输出"""
        return True


class LLMGenerator:
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.generator = self._create_generator()
    
    def _create_generator(self) -> BaseLLMGenerator:
        if self.config.backend == LLMBackend.LITELLM:
            return LiteLLMGenerator(self.config)
        elif self.config.backend == LLMBackend.VLLM:
            return VLLMGenerator(self.config)
        else:
            raise ValueError(f"Unsupported backend: {self.config.backend}")
    
    async def generate(
        self,
        template: str,
        input_values: Dict[str, str],
        temperature: float = 0.7,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """生成文本"""
        # 格式化模板
        formatted_content = template
        for key, value in input_values.items():
            formatted_content = formatted_content.replace(f'{{{key}}}', str(value))
        
        messages = [{'role': 'user', 'content': formatted_content}]
        
        return await self.generator.generate(
            messages=messages,
            response_format=response_format,
            temperature=temperature,
            **kwargs
        )
    
    async def generate_structured(
        self,
        template: str,
        input_values: Dict[str, str],
        schema: Dict[str, Any],
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """生成结构化输出"""
        # 格式化模板
        formatted_content = template
        for key, value in input_values.items():
            formatted_content = formatted_content.replace(f'{{{key}}}', str(value))
        
        messages = [{'role': 'user', 'content': formatted_content}]
        
        return await self.generator.generate_structured(
            messages=messages,
            schema=schema,
            temperature=temperature,
            **kwargs
        )
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """聊天补全"""
        return await self.generator.generate(
            messages=messages,
            temperature=temperature,
            **kwargs
        )
    
    def supports_structured_output(self) -> bool:
        """是否支持结构化输出"""
        return self.generator.supports_structured_output()
    
    def switch_backend(self, new_config: LLMConfig):
        """切换后端"""
        self.config = new_config
        self.generator = self._create_generator()


def create_litellm_generator(
    model_name: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    **kwargs
) -> LLMGenerator:
    """创建LiteLLM生成器"""
    config = LLMConfig(
        backend=LLMBackend.LITELLM,
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        custom_params=kwargs
    )
    return LLMGenerator(config)


def create_vllm_generator(
    model_name: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.7,  # 降低默认值以避免OOM
    max_model_len: Optional[int] = None,
    temperature: float = 0.7,
    **kwargs
) -> LLMGenerator:
    """创建vLLM生成器"""
    config = LLMConfig(
        backend=LLMBackend.VLLM,
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        temperature=temperature,
        custom_params=kwargs
    )
    return LLMGenerator(config)