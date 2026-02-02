"""
AWS Bedrock Provider

Implements LLMClient for Amazon Bedrock.
Supports Claude, Llama, Mistral, Titan, and other models.
"""

import os
import json
import time
import logging
from typing import Optional, List, Dict, Any

from ..client import (
    LLMClient,
    LLMResponse,
    ModelInfo,
    RateLimitError,
    AuthenticationError,
    ProviderError,
)

logger = logging.getLogger(__name__)

# Model configurations for AWS Bedrock
BEDROCK_MODELS = {
    # Anthropic Claude models
    'anthropic.claude-3-5-sonnet-20241022-v2:0': {
        'display_name': 'Claude 3.5 Sonnet v2',
        'max_tokens': 200000,
        'max_output_tokens': 8192,
        'cost_per_1k_input': 0.003,
        'cost_per_1k_output': 0.015,
        'provider': 'anthropic',
    },
    'anthropic.claude-3-sonnet-20240229-v1:0': {
        'display_name': 'Claude 3 Sonnet',
        'max_tokens': 200000,
        'max_output_tokens': 4096,
        'cost_per_1k_input': 0.003,
        'cost_per_1k_output': 0.015,
        'provider': 'anthropic',
    },
    'anthropic.claude-3-haiku-20240307-v1:0': {
        'display_name': 'Claude 3 Haiku',
        'max_tokens': 200000,
        'max_output_tokens': 4096,
        'cost_per_1k_input': 0.00025,
        'cost_per_1k_output': 0.00125,
        'provider': 'anthropic',
    },
    # Meta Llama models
    'meta.llama3-1-405b-instruct-v1:0': {
        'display_name': 'Llama 3.1 405B Instruct',
        'max_tokens': 128000,
        'max_output_tokens': 4096,
        'cost_per_1k_input': 0.00532,
        'cost_per_1k_output': 0.016,
        'provider': 'meta',
    },
    'meta.llama3-1-70b-instruct-v1:0': {
        'display_name': 'Llama 3.1 70B Instruct',
        'max_tokens': 128000,
        'max_output_tokens': 4096,
        'cost_per_1k_input': 0.00099,
        'cost_per_1k_output': 0.00099,
        'provider': 'meta',
    },
    # Mistral models
    'mistral.mistral-large-2407-v1:0': {
        'display_name': 'Mistral Large',
        'max_tokens': 128000,
        'max_output_tokens': 8192,
        'cost_per_1k_input': 0.004,
        'cost_per_1k_output': 0.012,
        'provider': 'mistral',
    },
    # Amazon Titan
    'amazon.titan-text-premier-v1:0': {
        'display_name': 'Amazon Titan Text Premier',
        'max_tokens': 32000,
        'max_output_tokens': 3072,
        'cost_per_1k_input': 0.0005,
        'cost_per_1k_output': 0.0015,
        'provider': 'amazon',
    },
}

DEFAULT_MODEL = 'anthropic.claude-3-sonnet-20240229-v1:0'


class BedrockClient(LLMClient):
    """
    AWS Bedrock client.

    Requires boto3: uv pip install boto3

    Authentication uses standard AWS credential chain:
    - Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    - Shared credentials file (~/.aws/credentials)
    - IAM role (when running on EC2/Lambda/ECS)
    """

    def __init__(
        self,
        model: Optional[str] = None,
        region: str = "us-east-1",
        profile: Optional[str] = None,
        timeout: float = 120.0,
        **kwargs
    ):
        """
        Initialize AWS Bedrock client.

        Args:
            model: Bedrock model ID
            region: AWS region (default: us-east-1)
            profile: AWS profile name (optional)
            timeout: Request timeout in seconds
        """
        super().__init__(model=model or DEFAULT_MODEL, **kwargs)

        self.region = region
        self.profile = profile
        self.timeout = timeout
        self._client = None

    def _get_client(self):
        """Lazy initialization of Bedrock client."""
        if self._client is None:
            try:
                import boto3
                from botocore.config import Config
            except ImportError:
                raise ProviderError(
                    "boto3 package not installed. Run: uv pip install boto3"
                )

            config = Config(
                read_timeout=self.timeout,
                connect_timeout=30,
                retries={'max_attempts': 3}
            )

            session_kwargs = {}
            if self.profile:
                session_kwargs['profile_name'] = self.profile

            session = boto3.Session(**session_kwargs)
            self._client = session.client(
                'bedrock-runtime',
                region_name=self.region,
                config=config
            )

        return self._client

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate completion using AWS Bedrock."""
        client = self._get_client()
        start_time = time.time()

        model_config = BEDROCK_MODELS.get(self.model, {})
        model_provider = model_config.get('provider', 'anthropic')

        # Build request based on model provider
        if model_provider == 'anthropic':
            body = self._build_anthropic_request(
                prompt, system, temperature, max_tokens, stop_sequences
            )
        elif model_provider == 'meta':
            body = self._build_meta_request(
                prompt, system, temperature, max_tokens, stop_sequences
            )
        elif model_provider == 'mistral':
            body = self._build_mistral_request(
                prompt, system, temperature, max_tokens, stop_sequences
            )
        else:
            body = self._build_generic_request(
                prompt, system, temperature, max_tokens, stop_sequences
            )

        try:
            response = client.invoke_model(
                modelId=self.model,
                body=json.dumps(body),
                contentType='application/json',
                accept='application/json'
            )

            latency_ms = (time.time() - start_time) * 1000

            response_body = json.loads(response['body'].read())

            # Parse response based on provider
            content, input_tokens, output_tokens, finish_reason = self._parse_response(
                response_body, model_provider
            )

            return LLMResponse(
                content=content,
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached=False,
                latency_ms=latency_ms,
                finish_reason=finish_reason,
                metadata={
                    'region': self.region,
                    'provider': model_provider,
                }
            )

        except Exception as e:
            error_str = str(e).lower()

            if 'throttl' in error_str or 'rate' in error_str:
                raise RateLimitError(f"Rate limit exceeded: {e}")
            elif 'access denied' in error_str or 'credentials' in error_str:
                raise AuthenticationError(f"Authentication failed: {e}")
            else:
                raise ProviderError(f"Bedrock API error: {e}")

    def _build_anthropic_request(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
        stop_sequences: Optional[List[str]]
    ) -> Dict:
        """Build request body for Anthropic Claude models."""
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]
        }

        if system:
            body["system"] = system

        if stop_sequences:
            body["stop_sequences"] = stop_sequences

        return body

    def _build_meta_request(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
        stop_sequences: Optional[List[str]]
    ) -> Dict:
        """Build request body for Meta Llama models."""
        full_prompt = prompt
        if system:
            full_prompt = f"<|system|>\n{system}\n<|user|>\n{prompt}\n<|assistant|>"

        return {
            "prompt": full_prompt,
            "max_gen_len": max_tokens,
            "temperature": temperature,
        }

    def _build_mistral_request(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
        stop_sequences: Optional[List[str]]
    ) -> Dict:
        """Build request body for Mistral models."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        return {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

    def _build_generic_request(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
        stop_sequences: Optional[List[str]]
    ) -> Dict:
        """Build generic request body."""
        full_prompt = f"{system}\n\n{prompt}" if system else prompt

        return {
            "inputText": full_prompt,
            "textGenerationConfig": {
                "maxTokenCount": max_tokens,
                "temperature": temperature,
            }
        }

    def _parse_response(
        self,
        response_body: Dict,
        model_provider: str
    ) -> tuple:
        """Parse response based on model provider."""
        if model_provider == 'anthropic':
            content = ""
            if response_body.get('content'):
                for block in response_body['content']:
                    if block.get('type') == 'text':
                        content += block.get('text', '')

            return (
                content,
                response_body.get('usage', {}).get('input_tokens', 0),
                response_body.get('usage', {}).get('output_tokens', 0),
                response_body.get('stop_reason', 'stop')
            )

        elif model_provider == 'meta':
            return (
                response_body.get('generation', ''),
                response_body.get('prompt_token_count', 0),
                response_body.get('generation_token_count', 0),
                response_body.get('stop_reason', 'stop')
            )

        elif model_provider == 'mistral':
            choices = response_body.get('choices', [{}])
            content = choices[0].get('message', {}).get('content', '')
            usage = response_body.get('usage', {})
            return (
                content,
                usage.get('prompt_tokens', 0),
                usage.get('completion_tokens', 0),
                choices[0].get('finish_reason', 'stop')
            )

        else:
            # Generic/Titan
            results = response_body.get('results', [{}])
            return (
                results[0].get('outputText', ''),
                response_body.get('inputTextTokenCount', 0),
                results[0].get('tokenCount', 0),
                results[0].get('completionReason', 'stop')
            )

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        # Approximate: ~4 chars per token
        return len(text) // 4 + 1

    def get_model_info(self) -> ModelInfo:
        """Get information about the current model."""
        model_config = BEDROCK_MODELS.get(self.model, {
            'display_name': self.model,
            'max_tokens': 100000,
            'max_output_tokens': 4096,
            'cost_per_1k_input': 0.01,
            'cost_per_1k_output': 0.03,
        })

        return ModelInfo(
            provider='aws',
            model_id=self.model,
            display_name=model_config['display_name'],
            max_tokens=model_config['max_tokens'],
            max_output_tokens=model_config['max_output_tokens'],
            supports_system_prompt=True,
            supports_streaming=True,
            cost_per_1k_input=model_config['cost_per_1k_input'],
            cost_per_1k_output=model_config['cost_per_1k_output'],
        )
