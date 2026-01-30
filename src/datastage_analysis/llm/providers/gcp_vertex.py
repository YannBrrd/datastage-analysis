"""
GCP Vertex AI Provider

Implements LLMClient for Google Cloud Vertex AI.
Supports Gemini, PaLM, Claude, and other models.
"""

import os
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

# Model configurations for Vertex AI
VERTEX_MODELS = {
    # Gemini models
    'gemini-1.5-pro': {
        'display_name': 'Gemini 1.5 Pro',
        'max_tokens': 2097152,  # 2M context
        'max_output_tokens': 8192,
        'cost_per_1k_input': 0.00125,
        'cost_per_1k_output': 0.005,
    },
    'gemini-1.5-flash': {
        'display_name': 'Gemini 1.5 Flash',
        'max_tokens': 1048576,  # 1M context
        'max_output_tokens': 8192,
        'cost_per_1k_input': 0.000075,
        'cost_per_1k_output': 0.0003,
    },
    'gemini-1.0-pro': {
        'display_name': 'Gemini 1.0 Pro',
        'max_tokens': 32760,
        'max_output_tokens': 8192,
        'cost_per_1k_input': 0.0005,
        'cost_per_1k_output': 0.0015,
    },
    # Claude models on Vertex
    'claude-3-5-sonnet@20240620': {
        'display_name': 'Claude 3.5 Sonnet',
        'max_tokens': 200000,
        'max_output_tokens': 8192,
        'cost_per_1k_input': 0.003,
        'cost_per_1k_output': 0.015,
    },
    'claude-3-opus@20240229': {
        'display_name': 'Claude 3 Opus',
        'max_tokens': 200000,
        'max_output_tokens': 4096,
        'cost_per_1k_input': 0.015,
        'cost_per_1k_output': 0.075,
    },
    'claude-3-haiku@20240307': {
        'display_name': 'Claude 3 Haiku',
        'max_tokens': 200000,
        'max_output_tokens': 4096,
        'cost_per_1k_input': 0.00025,
        'cost_per_1k_output': 0.00125,
    },
}

DEFAULT_MODEL = 'gemini-1.5-pro'


class VertexClient(LLMClient):
    """
    GCP Vertex AI client.

    Requires: pip install google-cloud-aiplatform

    Authentication uses Application Default Credentials:
    - gcloud auth application-default login
    - GOOGLE_APPLICATION_CREDENTIALS environment variable
    - Service account when running on GCP
    """

    def __init__(
        self,
        model: Optional[str] = None,
        project_id: Optional[str] = None,
        location: str = "us-central1",
        timeout: float = 120.0,
        **kwargs
    ):
        """
        Initialize Vertex AI client.

        Args:
            model: Model ID
            project_id: GCP project ID (default: from GOOGLE_CLOUD_PROJECT)
            location: GCP region (default: us-central1)
            timeout: Request timeout in seconds
        """
        super().__init__(model=model or DEFAULT_MODEL, **kwargs)

        self.project_id = project_id or os.environ.get('GOOGLE_CLOUD_PROJECT') or os.environ.get('GCP_PROJECT_ID')
        self.location = location
        self.timeout = timeout
        self._initialized = False

        if not self.project_id:
            raise AuthenticationError(
                "GCP project ID not found. Set GOOGLE_CLOUD_PROJECT environment variable."
            )

    def _initialize(self):
        """Initialize Vertex AI SDK."""
        if self._initialized:
            return

        try:
            import vertexai
            vertexai.init(project=self.project_id, location=self.location)
            self._initialized = True
        except ImportError:
            raise ProviderError(
                "google-cloud-aiplatform package not installed. "
                "Run: pip install google-cloud-aiplatform"
            )
        except Exception as e:
            raise AuthenticationError(f"Failed to initialize Vertex AI: {e}")

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate completion using Vertex AI."""
        self._initialize()
        start_time = time.time()

        # Determine if using Gemini or Claude
        is_gemini = self.model.startswith('gemini')
        is_claude = 'claude' in self.model

        if is_gemini:
            return self._complete_gemini(
                prompt, system, temperature, max_tokens, stop_sequences, start_time
            )
        elif is_claude:
            return self._complete_claude(
                prompt, system, temperature, max_tokens, stop_sequences, start_time
            )
        else:
            # Default to Gemini-style
            return self._complete_gemini(
                prompt, system, temperature, max_tokens, stop_sequences, start_time
            )

    def _complete_gemini(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
        stop_sequences: Optional[List[str]],
        start_time: float
    ) -> LLMResponse:
        """Complete using Gemini models."""
        try:
            from vertexai.generative_models import GenerativeModel, GenerationConfig
        except ImportError:
            raise ProviderError("vertexai package not properly installed")

        try:
            model = GenerativeModel(
                self.model,
                system_instruction=system if system else None,
            )

            generation_config = GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                stop_sequences=stop_sequences or [],
            )

            response = model.generate_content(
                prompt,
                generation_config=generation_config,
            )

            latency_ms = (time.time() - start_time) * 1000

            content = response.text if hasattr(response, 'text') else ""

            # Get usage metadata
            input_tokens = 0
            output_tokens = 0
            if hasattr(response, 'usage_metadata'):
                input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
                output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)

            finish_reason = "stop"
            if response.candidates:
                finish_reason = str(response.candidates[0].finish_reason.name)

            return LLMResponse(
                content=content,
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached=False,
                latency_ms=latency_ms,
                finish_reason=finish_reason,
                metadata={
                    'project': self.project_id,
                    'location': self.location,
                }
            )

        except Exception as e:
            error_str = str(e).lower()

            if 'quota' in error_str or 'rate' in error_str:
                raise RateLimitError(f"Rate limit exceeded: {e}")
            elif 'permission' in error_str or 'credentials' in error_str:
                raise AuthenticationError(f"Authentication failed: {e}")
            else:
                raise ProviderError(f"Vertex AI error: {e}")

    def _complete_claude(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
        stop_sequences: Optional[List[str]],
        start_time: float
    ) -> LLMResponse:
        """Complete using Claude models on Vertex."""
        try:
            from anthropic import AnthropicVertex
        except ImportError:
            raise ProviderError(
                "anthropic package not installed. Run: pip install anthropic[vertex]"
            )

        try:
            client = AnthropicVertex(
                project_id=self.project_id,
                region=self.location,
            )

            messages = [{"role": "user", "content": prompt}]

            kwargs = {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": messages,
                "temperature": temperature,
            }

            if system:
                kwargs["system"] = system

            if stop_sequences:
                kwargs["stop_sequences"] = stop_sequences

            response = client.messages.create(**kwargs)

            latency_ms = (time.time() - start_time) * 1000

            content = ""
            if response.content:
                for block in response.content:
                    if hasattr(block, 'text'):
                        content += block.text

            return LLMResponse(
                content=content,
                model=response.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cached=False,
                latency_ms=latency_ms,
                finish_reason=response.stop_reason or "stop",
                metadata={
                    'id': response.id,
                    'project': self.project_id,
                }
            )

        except Exception as e:
            error_str = str(e).lower()

            if 'quota' in error_str or 'rate' in error_str:
                raise RateLimitError(f"Rate limit exceeded: {e}")
            elif 'permission' in error_str or 'credentials' in error_str:
                raise AuthenticationError(f"Authentication failed: {e}")
            else:
                raise ProviderError(f"Vertex AI Claude error: {e}")

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        # Approximate: ~4 chars per token
        return len(text) // 4 + 1

    def get_model_info(self) -> ModelInfo:
        """Get information about the current model."""
        model_config = VERTEX_MODELS.get(self.model, {
            'display_name': self.model,
            'max_tokens': 32000,
            'max_output_tokens': 8192,
            'cost_per_1k_input': 0.01,
            'cost_per_1k_output': 0.03,
        })

        return ModelInfo(
            provider='gcp',
            model_id=self.model,
            display_name=model_config['display_name'],
            max_tokens=model_config['max_tokens'],
            max_output_tokens=model_config['max_output_tokens'],
            supports_system_prompt=True,
            supports_streaming=True,
            cost_per_1k_input=model_config['cost_per_1k_input'],
            cost_per_1k_output=model_config['cost_per_1k_output'],
        )
