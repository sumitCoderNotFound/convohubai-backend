"""
ConvoHubAI - LLM Service
Handles integration with OpenAI, Anthropic, and other LLM providers
"""
import os
from typing import AsyncGenerator, Optional, List, Dict, Any
from openai import AsyncOpenAI
import anthropic
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
            provider: 'openai' or 'anthropic'
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


# Singleton instance
llm_service = LLMService()