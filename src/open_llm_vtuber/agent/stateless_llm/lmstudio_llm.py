"""LM Studio LLM implementation."""

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
        use_harmony: bool = False,
        max_tokens: int = 150,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        seed: int | None = None,
        # New VTuber-specific parameters
        stop_sequences: list[str] | None = None,
        repetition_penalty: float = 1.1,
        length_penalty: float = 0.8,
        stream: bool = True,  # Add stream parameter
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
            use_harmony: Whether to use Harmony encoding
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            frequency_penalty: Frequency penalty for repetition
            presence_penalty: Presence penalty for repetition
            stop: Stop sequences
            seed: Random seed
            stop_sequences: Additional stop sequences for VTuber responses
            repetition_penalty: Penalty for repetition (1.0-2.0)
            length_penalty: Penalty for length (0.0-2.0)
            stream: Whether to use streaming mode (default: True)
        """
        super().__init__(
            model=model,
            base_url=base_url,
            llm_api_key=llm_api_key,
            organization_id=organization_id,
            project_id=project_id,
            temperature=temperature,
            use_harmony=use_harmony,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            seed=seed,
        )

        # Store VTuber-specific parameters
        self.stop_sequences = stop_sequences
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty
        self.stream_enabled = stream  # Store stream setting

        logger.info(f"Initialized LM Studio LLM with model: {model}")
        logger.info(f"LM Studio API URL: {base_url}")
        logger.info(
            f"VTuber settings: repetition_penalty={repetition_penalty}, length_penalty={length_penalty}, stream={stream}"
        )

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
        # For LM Studio, we want to disable tools by default to ensure text streaming works
        if tools is not None:
            logger.warning(
                "LM Studio detected; disabling tools for streaming compatibility."
            )
            tools = None

        try:
            # Apply VTuber-specific stop sequences if available
            original_stop = self.stop
            if self.stop_sequences:
                # Combine with existing stop sequences
                combined_stop = (self.stop or []) + self.stop_sequences
                self.stop = combined_stop
                logger.debug(f"Applied VTuber stop sequences: {self.stop_sequences}")

            # If streaming is disabled, use non-streaming mode
            if not self.stream_enabled:
                logger.info("LM Studio streaming disabled, using non-streaming mode")
                try:
                    # Prepare messages with system prompt
                    messages_with_system = messages
                    if system:
                        messages_with_system = [
                            {"role": "system", "content": system},
                            *messages,
                        ]

                    # Make non-streaming request
                    response = await self.client.chat.completions.create(
                        messages=messages_with_system,
                        model=self.model,
                        stream=False,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p,
                        frequency_penalty=self.frequency_penalty,
                        presence_penalty=self.presence_penalty,
                        stop=self.stop if self.stop else None,
                        seed=self.seed,
                    )

                    # Extract content and yield as single chunk
                    content = response.choices[0].message.content or ""
                    logger.debug(f"LM Studio non-streaming response: '{content}'")
                    yield content
                    return

                except Exception as e:
                    logger.error(f"LM Studio non-streaming request failed: {e}")
                    yield f"[Error: {str(e)}]"
                    return

            # Use streaming mode (default)
            logger.debug("LM Studio using streaming mode")
            async for chunk in super().chat_completion(messages, system, tools):
                yield chunk

        except Exception as e:
            logger.error(f"LM Studio chat completion error: {e}")
            # Provide a fallback response if the model fails
            fallback_response = (
                "Извините, произошла ошибка при обработке запроса. Попробуйте еще раз."
            )
            logger.warning(f"Using fallback response: {fallback_response}")
            yield fallback_response
        finally:
            # Restore original stop sequences
            if self.stop_sequences:
                self.stop = original_stop
