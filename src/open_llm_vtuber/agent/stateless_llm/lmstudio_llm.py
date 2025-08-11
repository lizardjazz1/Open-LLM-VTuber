"""LM Studio LLM implementation."""

import asyncio
from typing import AsyncIterator, List, Dict, Any
from loguru import logger
from .openai_compatible_llm import AsyncLLM


class LMStudioLLM(AsyncLLM):
    """LM Studio LLM implementation using OpenAI-compatible API."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:1234/v1",
        llm_api_key: str = "default_api_key",
        organization_id: str = "z",
        project_id: str = "z",
        temperature: float = 1.0,
    ):
        """
        Initialize LM Studio LLM.

        Args:
            model: Model name to use
            base_url: LM Studio API base URL (default: http://localhost:1234/v1)
            llm_api_key: API key (not used for local LM Studio)
            organization_id: Organization ID (not used for local LM Studio)
            project_id: Project ID (not used for local LM Studio)
            temperature: Sampling temperature
        """
        super().__init__(
            model=model,
            base_url=base_url,
            llm_api_key=llm_api_key,
            organization_id=organization_id,
            project_id=project_id,
            temperature=temperature,
        )
        
        logger.info(f"Initialized LM Studio LLM with model: {model}")
        logger.info(f"LM Studio API URL: {base_url}")

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        system: str = None,
        tools: List[Dict[str, Any]] = None,
    ) -> AsyncIterator[str]:
        """
        Generate chat completion using LM Studio.

        Args:
            messages: List of message dictionaries
            system: Optional system prompt
            tools: Optional list of tools

        Yields:
            Response chunks as strings
        """
        try:
            # Use the parent class implementation which handles OpenAI-compatible API
            async for chunk in super().chat_completion(messages, system, tools):
                yield chunk
        except Exception as e:
            logger.error(f"LM Studio chat completion error: {e}")
            raise 