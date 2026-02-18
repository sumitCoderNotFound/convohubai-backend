"""
ConvoHubAI - LLM Service
Handles integration with OpenAI, Anthropic, Groq (FREE), and Deepgram
"""
import os
from typing import AsyncGenerator, Optional, List, Dict, Any
from openai import AsyncOpenAI
import anthropic
import httpx
from app.core.config import settings


class LLMService:
    """Service for interacting with LLM providers."""
    
    def __init__(self):
        # Initialize OpenAI client
        self.openai_client = None
        if settings.openai_api_key:
            self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
        
        # Initialize Anthropic client
        self.anthropic_client = None
        if settings.anthropic_api_key:
            self.anthropic_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        
        # Initialize Groq client (uses OpenAI-compatible API)
        self.groq_client = None
        groq_api_key = getattr(settings, 'groq_api_key', None) or os.getenv('GROQ_API_KEY')
        if groq_api_key:
            self.groq_client = AsyncOpenAI(
                api_key=groq_api_key,
                base_url="https://api.groq.com/openai/v1"
            )
        
        # Deepgram API key
        self.deepgram_api_key = getattr(settings, 'deepgram_api_key', None) or os.getenv('DEEPGRAM_API_KEY')
    
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        model: str = "gpt-4o-mini",
        provider: str = "openai",
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            system_prompt: Optional system prompt to prepend
            model: Model name to use
            provider: 'openai', 'anthropic', 'groq', or 'deepgram'
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated response text
        """
        if provider == "openai":
            return await self._openai_generate(
                messages, system_prompt, model, temperature, max_tokens
            )
        elif provider == "anthropic":
            return await self._anthropic_generate(
                messages, system_prompt, model, temperature, max_tokens
            )
        elif provider == "groq":
            return await self._groq_generate(
                messages, system_prompt, model, temperature, max_tokens
            )
        elif provider == "deepgram":
            # Deepgram is primarily STT/TTS, fallback to Groq for LLM
            if self.groq_client:
                return await self._groq_generate(
                    messages, system_prompt, "llama-3.3-70b-versatile", temperature, max_tokens
                )
            raise ValueError("Deepgram requires Groq as fallback for LLM")
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    async def generate_response_stream(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        model: str = "gpt-4o-mini",
        provider: str = "openai",
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response from the LLM.
        
        Yields chunks of text as they are generated.
        """
        if provider == "openai":
            async for chunk in self._openai_stream(
                messages, system_prompt, model, temperature, max_tokens
            ):
                yield chunk
        elif provider == "anthropic":
            async for chunk in self._anthropic_stream(
                messages, system_prompt, model, temperature, max_tokens
            ):
                yield chunk
        elif provider == "groq":
            async for chunk in self._groq_stream(
                messages, system_prompt, model, temperature, max_tokens
            ):
                yield chunk
        elif provider == "deepgram":
            # Fallback to Groq for streaming
            if self.groq_client:
                async for chunk in self._groq_stream(
                    messages, system_prompt, "llama-3.3-70b-versatile", temperature, max_tokens
                ):
                    yield chunk
            else:
                raise ValueError("Deepgram requires Groq as fallback for LLM")
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    # ============================================
    # OpenAI Methods
    # ============================================
    
    async def _openai_generate(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate response using OpenAI."""
        if not self.openai_client:
            raise ValueError("OpenAI API key not configured")
        
        # Build messages list
        openai_messages = []
        if system_prompt:
            openai_messages.append({"role": "system", "content": system_prompt})
        openai_messages.extend(messages)
        
        response = await self.openai_client.chat.completions.create(
            model=model,
            messages=openai_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        return response.choices[0].message.content
    
    async def _openai_stream(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> AsyncGenerator[str, None]:
        """Stream response using OpenAI."""
        if not self.openai_client:
            raise ValueError("OpenAI API key not configured")
        
        # Build messages list
        openai_messages = []
        if system_prompt:
            openai_messages.append({"role": "system", "content": system_prompt})
        openai_messages.extend(messages)
        
        stream = await self.openai_client.chat.completions.create(
            model=model,
            messages=openai_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    # ============================================
    # Groq Methods (FREE!)
    # ============================================
    
    async def _groq_generate(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate response using Groq (FREE!)."""
        if not self.groq_client:
            raise ValueError("Groq API key not configured. Get FREE key at https://console.groq.com")
        
        # Build messages list
        groq_messages = []
        if system_prompt:
            groq_messages.append({"role": "system", "content": system_prompt})
        groq_messages.extend(messages)
        
        # Map model names if needed
        groq_model = self._map_groq_model(model)
        
        response = await self.groq_client.chat.completions.create(
            model=groq_model,
            messages=groq_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        return response.choices[0].message.content
    
    async def _groq_stream(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> AsyncGenerator[str, None]:
        """Stream response using Groq (FREE!)."""
        if not self.groq_client:
            raise ValueError("Groq API key not configured. Get FREE key at https://console.groq.com")
        
        # Build messages list
        groq_messages = []
        if system_prompt:
            groq_messages.append({"role": "system", "content": system_prompt})
        groq_messages.extend(messages)
        
        # Map model names if needed
        groq_model = self._map_groq_model(model)
        
        stream = await self.groq_client.chat.completions.create(
            model=groq_model,
            messages=groq_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    def _map_groq_model(self, model: str) -> str:
        """Map model name to Groq model."""
        model_map = {
            # Current Groq models
            "llama-3.3-70b-versatile": "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant": "llama-3.1-8b-instant",
            "mixtral-8x7b-32768": "mixtral-8x7b-32768",
            "gemma2-9b-it": "gemma2-9b-it",
            # Aliases
            "llama3": "llama-3.3-70b-versatile",
            "mixtral": "mixtral-8x7b-32768",
        }
        return model_map.get(model, model)
    
    # ============================================
    # Anthropic Methods
    # ============================================
    
    async def _anthropic_generate(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate response using Anthropic."""
        if not self.anthropic_client:
            raise ValueError("Anthropic API key not configured")
        
        # Map model names
        anthropic_model = self._map_anthropic_model(model)
        
        # Convert messages format
        anthropic_messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
        ]
        
        response = await self.anthropic_client.messages.create(
            model=anthropic_model,
            max_tokens=max_tokens,
            system=system_prompt or "",
            messages=anthropic_messages,
        )
        
        return response.content[0].text
    
    async def _anthropic_stream(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> AsyncGenerator[str, None]:
        """Stream response using Anthropic."""
        if not self.anthropic_client:
            raise ValueError("Anthropic API key not configured")
        
        # Map model names
        anthropic_model = self._map_anthropic_model(model)
        
        # Convert messages format
        anthropic_messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
        ]
        
        async with self.anthropic_client.messages.stream(
            model=anthropic_model,
            max_tokens=max_tokens,
            system=system_prompt or "",
            messages=anthropic_messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text
    
    def _map_anthropic_model(self, model: str) -> str:
        """Map model name to Anthropic model."""
        model_map = {
            "claude-3-opus": "claude-3-opus-20240229",
            "claude-3-sonnet": "claude-3-sonnet-20240229",
            "claude-3-haiku": "claude-3-haiku-20240307",
        }
        return model_map.get(model, model)
    
    # ============================================
    # Utility Methods
    # ============================================
    
    def get_available_providers(self) -> List[str]:
        """Get list of configured providers."""
        providers = []
        if self.openai_client:
            providers.append("openai")
        if self.groq_client:
            providers.append("groq")
        if self.anthropic_client:
            providers.append("anthropic")
        if self.deepgram_api_key:
            providers.append("deepgram")
        return providers
    
    def is_provider_available(self, provider: str) -> bool:
        """Check if a provider is configured."""
        if provider == "openai":
            return self.openai_client is not None
        elif provider == "groq":
            return self.groq_client is not None
        elif provider == "anthropic":
            return self.anthropic_client is not None
        elif provider == "deepgram":
            return self.deepgram_api_key is not None
        return False


# Singleton instance
llm_service = LLMService()