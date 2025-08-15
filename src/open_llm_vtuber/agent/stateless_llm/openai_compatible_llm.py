"""Description: This file contains the implementation of the `AsyncLLM` class.
This class is responsible for handling asynchronous interaction with OpenAI API compatible
endpoints for language generation.
"""

from typing import AsyncIterator, List, Dict, Any
from openai import (
    AsyncStream,
    AsyncOpenAI,
    APIError,
    APIConnectionError,
    RateLimitError,
    NotGiven,
    NOT_GIVEN,
)
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from loguru import logger
import time

from .stateless_llm_interface import StatelessLLMInterface
from ...mcpp.types import ToolCallObject
from ...debug_settings import ensure_log_sinks

_DEBUG_WS_UNUSED, DEBUG_LLM = ensure_log_sinks()

# Harmony imports
try:
    from openai_harmony import (
        load_harmony_encoding,
        HarmonyEncodingName,
    )

    HARMONY_AVAILABLE = True
except ImportError:
    HARMONY_AVAILABLE = False
    logger.warning("openai-harmony not available. Harmony mode will be disabled.")


class AsyncLLM(StatelessLLMInterface):
    def __init__(
        self,
        model: str,
        base_url: str,
        llm_api_key: str = "z",
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
        fallback_model: str | None = None,
    ):
        """
        Initializes an instance of the `AsyncLLM` class.

        Parameters:
        - model (str): The model to be used for language generation.
        - base_url (str): The base URL for the OpenAI API.
        - organization_id (str, optional): The organization ID for the OpenAI API. Defaults to "z".
        - project_id (str, optional): The project ID for the OpenAI API. Defaults to "z".
        - llm_api_key (str, optional): The API key for the OpenAI API. Defaults to "z".
        - temperature (float, optional): What sampling temperature to use, between 0 and 2. Defaults to 1.0.
        - use_harmony (bool, optional): Whether to use OpenAI Harmony format. Defaults to False.
        """
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.use_harmony = use_harmony and HARMONY_AVAILABLE
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop
        self.seed = seed
        self.fallback_model = fallback_model
        self.client = AsyncOpenAI(
            base_url=base_url,
            organization=organization_id,
            project=project_id,
            api_key=llm_api_key,
        )
        self.support_tools = True

        # Heuristic: LM Studio OpenAI-compatible server often misbehaves with tool_calls in stream
        # Disable tools by default for LM Studio endpoints
        try:
            if (
                "127.0.0.1:1234" in (base_url or "")
                or "lmstudio" in (base_url or "").lower()
            ):
                self.support_tools = False
                logger.warning(
                    "LM Studio detected; disabling tools for streaming compatibility."
                )
        except Exception:
            pass

        # Инициализируем Harmony если доступен и включен
        if self.use_harmony:
            try:
                self.harmony_enc = load_harmony_encoding(
                    HarmonyEncodingName.HARMONY_GPT_OSS
                )
                logger.info("Harmony encoding initialized for OpenAICompatibleLLM")
            except Exception as e:
                logger.error(f"Failed to initialize Harmony encoding: {e}")
                self.use_harmony = False

        logger.info(
            f"Initialized AsyncLLM with the parameters: {self.base_url}, {self.model}, use_harmony={self.use_harmony}"
        )

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        system: str = None,
        tools: List[Dict[str, Any]] | NotGiven = NOT_GIVEN,
    ) -> AsyncIterator[str | List[ChoiceDeltaToolCall]]:
        """
        Generates a chat completion using the OpenAI API asynchronously.

        Parameters:
        - messages (List[Dict[str, Any]]): The list of messages to send to the API.
        - system (str, optional): System prompt to use for this completion.
        - tools (List[Dict[str, str]], optional): List of tools to use for this completion.

        Yields:
        - str: The content of each chunk from the API response.
        - List[ChoiceDeltaToolCall]: The tool calls detected in the response.

        Raises:
        - APIConnectionError: When the server cannot be reached
        - RateLimitError: When a 429 status code is received
        - APIError: For other API-related errors
        """
        stream = None
        # Tool call related state variables
        accumulated_tool_calls = {}
        in_tool_call = False

        if DEBUG_LLM:
            try:
                logger.bind(dst="llm").log(
                    "INFO",
                    (
                        f"LLM OUT (openai-compatible): model={self.model}, "
                        f"params={{'temperature': {self.temperature}, 'top_p': {self.top_p}, 'max_tokens': {self.max_tokens}}}, "
                        f"system_prompt={system}, "
                        f"messages={messages}"
                    ),
                )
            except Exception:
                pass
        try:
            t_start = time.perf_counter()
            t_first_chunk: float | None = None
            has_yielded_text: bool = False
            # If system prompt is provided, add it to the messages
            messages_with_system = messages
            if system:
                messages_with_system = [
                    {"role": "system", "content": system},
                    *messages,
                ]
            logger.debug(
                f"Full request to LLM: messages_with_system={messages_with_system}"
            )

            available_tools = tools if self.support_tools else NOT_GIVEN

            stream: AsyncStream[
                ChatCompletionChunk
            ] = await self.client.chat.completions.create(
                messages=messages_with_system,
                model=self.model,
                stream=True,
                temperature=self.temperature,
                max_tokens=getattr(
                    self, "max_tokens", 150
                ),  # Ограничиваем длину ответа
                top_p=getattr(self, "top_p", 1.0),
                frequency_penalty=getattr(self, "frequency_penalty", 0.0),
                presence_penalty=getattr(self, "presence_penalty", 0.0),
                stop=getattr(self, "stop", None)
                if getattr(self, "stop", None)
                else NOT_GIVEN,
                seed=getattr(self, "seed", None)
                if getattr(self, "seed", None) is not None
                else NOT_GIVEN,
                tools=available_tools,
            )
            logger.debug(
                f"Tool Support: {self.support_tools}, Available tools: {available_tools}"
            )

            async for chunk in stream:
                if t_first_chunk is None:
                    t_first_chunk = time.perf_counter()
                    logger.bind(component="perf").info(
                        {
                            "stage": "llm_ttfb_ms",
                            "latency_ms": int((t_first_chunk - t_start) * 1000),
                            "model": self.model,
                        }
                    )

                if self.support_tools:
                    has_tool_calls = (
                        hasattr(chunk.choices[0].delta, "tool_calls")
                        and chunk.choices[0].delta.tool_calls
                    )

                    if has_tool_calls:
                        logger.debug(
                            f"Tool calls detected in chunk: {chunk.choices[0].delta.tool_calls}"
                        )
                        in_tool_call = True
                        # Process tool calls in the current chunk
                        for tool_call in chunk.choices[0].delta.tool_calls:
                            index = (
                                tool_call.index if hasattr(tool_call, "index") else 0
                            )

                            # Initialize tool call for this index if needed
                            if index not in accumulated_tool_calls:
                                accumulated_tool_calls[index] = {
                                    "index": index,
                                    "id": getattr(tool_call, "id", None),
                                    "type": getattr(tool_call, "type", None),
                                    "function": {"name": "", "arguments": ""},
                                }

                            # Update tool call information
                            if hasattr(tool_call, "id") and tool_call.id:
                                accumulated_tool_calls[index]["id"] = tool_call.id
                            if hasattr(tool_call, "type") and tool_call.type:
                                accumulated_tool_calls[index]["type"] = tool_call.type

                            # Update function information
                            if hasattr(tool_call, "function"):
                                if (
                                    hasattr(tool_call.function, "name")
                                    and tool_call.function.name
                                ):
                                    accumulated_tool_calls[index]["function"][
                                        "name"
                                    ] = tool_call.function.name
                                if (
                                    hasattr(tool_call.function, "arguments")
                                    and tool_call.function.arguments
                                ):
                                    accumulated_tool_calls[index]["function"][
                                        "arguments"
                                    ] += tool_call.function.arguments

                        continue

                    # If we were in a tool call but now we're not, yield the tool call result
                    elif in_tool_call and not has_tool_calls:
                        in_tool_call = False
                        # Convert accumulated tool calls to the required format and output
                        logger.info(f"Complete tool calls: {accumulated_tool_calls}")

                        # Use the from_dict method to create a ToolCallObject instance from a dictionary
                        complete_tool_calls = [
                            ToolCallObject.from_dict(tool_data)
                            for tool_data in accumulated_tool_calls.values()
                        ]

                        logger.info(
                            f"Yielding complete tool calls ({len(complete_tool_calls)}) to agent"
                        )
                        yield complete_tool_calls
                        accumulated_tool_calls = {}  # Reset for potential future tool calls

                # Process regular content chunks
                if len(chunk.choices) == 0:
                    logger.info("Empty chunk received")
                    continue
                elif chunk.choices[0].delta.content is None:
                    chunk.choices[0].delta.content = ""

                content = chunk.choices[0].delta.content
                if content:
                    # Log chunks only in DEBUG mode to reduce noise
                    if DEBUG_LLM:
                        logger.debug(f"🔥 LLM yielding chunk: {repr(content)}")
                        try:
                            logger.bind(dst="llm").log(
                                "DEBUG",
                                f"LLM stream content chunk: {repr(content)}",
                            )
                        except Exception:
                            pass
                    has_yielded_text = True
                    yield content

            # If stream ends while still in a tool call, make sure to yield the tool call
            if in_tool_call and accumulated_tool_calls:
                logger.info(f"Final tool call at stream end: {accumulated_tool_calls}")

                # Create a ToolCallObject instance from a dictionary using the from_dict method.
                complete_tool_calls = [
                    ToolCallObject.from_dict(tool_data)
                    for tool_data in accumulated_tool_calls.values()
                ]

                yield complete_tool_calls

            # Fallback: if no text chunks were yielded, try a non-stream completion and output its content
            if not has_yielded_text:
                try:
                    logger.warning(
                        "No text chunks yielded during stream; attempting non-stream fallback."
                    )
                    completion = await self.client.chat.completions.create(
                        messages=messages_with_system,
                        model=self.model,
                        stream=False,
                        temperature=self.temperature,
                        max_tokens=getattr(self, "max_tokens", 150),
                        top_p=getattr(self, "top_p", 1.0),
                        frequency_penalty=getattr(self, "frequency_penalty", 0.0),
                        presence_penalty=getattr(self, "presence_penalty", 0.0),
                        stop=getattr(self, "stop", None)
                        if getattr(self, "stop", None)
                        else NOT_GIVEN,
                        seed=getattr(self, "seed", None)
                        if getattr(self, "seed", None) is not None
                        else NOT_GIVEN,
                        tools=available_tools,
                    )
                    final_text = ""
                    try:
                        final_text = completion.choices[0].message.content or ""
                    except Exception:
                        final_text = ""
                    # Try reasoning_content if provider returned only reasoning
                    if not final_text:
                        try:
                            rc = getattr(
                                completion.choices[0].message, "reasoning_content", None
                            )
                            if isinstance(rc, str) and rc.strip():
                                final_text = rc
                                logger.info(
                                    "✅ Non-stream fallback used reasoning_content."
                                )
                        except Exception:
                            pass
                    if final_text:
                        logger.info(
                            f"✅ Non-stream fallback produced content (len={len(final_text)})."
                        )
                        yield final_text
                    else:
                        logger.warning(
                            "Non-stream fallback returned empty content as well."
                        )
                except Exception as e:
                    logger.error(f"Non-stream fallback failed: {e}")

        except APIConnectionError as e:
            logger.error(
                f"Error calling the chat endpoint: Connection error. Failed to connect to the LLM API. \nCheck the configurations and the reachability of the LLM backend. \nSee the logs for details. \nTroubleshooting with documentation: https://open-llm-vtuber.github.io/docs/faq#%E9%81%87%E5%88%B0-error-calling-the-chat-endpoint-%E9%94%99%E8%AF%AF%E6%80%8E%E4%B9%88%E5%8A%9E \n{e.__cause__}"
            )
            yield "Error calling the chat endpoint: Connection error. Failed to connect to the LLM API. Check the configurations and the reachability of the LLM backend. See the logs for details. Troubleshooting with documentation: [https://open-llm-vtuber.github.io/docs/faq#%E9%81%87%E5%88%B0-error-calling-the-chat-endpoint-%E9%94%99%E8%AF%AF%E6%80%8E%E4%B9%88%E5%8A%9E]"

        except RateLimitError as e:
            logger.error(
                f"Error calling the chat endpoint: Rate limit exceeded: {e.response}"
            )
            yield "Error calling the chat endpoint: Rate limit exceeded. Please try again later. See the logs for details."

        except APIError as e:
            if "does not support tools" in str(e):
                self.support_tools = False
                logger.warning(
                    f"{self.model} does not support tools. Disabling tool support."
                )
                yield "__API_NOT_SUPPORT_TOOLS__"
                return
            # Auto-retry for LM Studio when model id is wrong but suggestions are provided
            try:
                err_text = str(e)
                if (
                    "model_not_found" in err_text
                    or 'Model "' in err_text
                    and "not found" in err_text
                ):
                    # Heuristic: extract suggested models and prefer one matching the family
                    suggestions: list[str] = []
                    try:
                        if "Your models:" in err_text:
                            tail = err_text.split("Your models:")[-1]
                            for line in tail.splitlines():
                                line = line.strip().strip("'\" ")
                                if line and all(
                                    k not in line.lower()
                                    for k in ["error", "param", "code"]
                                ):
                                    suggestions.append(line)
                    except Exception:
                        suggestions = []
                    alt_model = None
                    # Prefer same family replacement (e.g., qwen3-30b-...)
                    fam = None
                    try:
                        fam = (
                            self.model.split("/")[-1].split(":")[0]
                            if self.model
                            else None
                        )
                    except Exception:
                        fam = None
                    if fam:
                        for s in suggestions:
                            if s.startswith(fam):
                                alt_model = s
                                break
                    # Fallback to configured model from YAML if provided
                    if not alt_model and self.fallback_model:
                        alt_model = self.fallback_model
                    if alt_model:
                        logger.warning(
                            f"Model '{self.model}' not found. Retrying once with '{alt_model}'."
                        )
                        try:
                            completion = await self.client.chat.completions.create(
                                messages=[{"role": "system", "content": system}]
                                + messages
                                if system
                                else messages,
                                model=alt_model,
                                stream=False,
                                temperature=self.temperature,
                                max_tokens=getattr(self, "max_tokens", 150),
                                top_p=getattr(self, "top_p", 1.0),
                                frequency_penalty=getattr(
                                    self, "frequency_penalty", 0.0
                                ),
                                presence_penalty=getattr(self, "presence_penalty", 0.0),
                                stop=getattr(self, "stop", None)
                                if getattr(self, "stop", None)
                                else NOT_GIVEN,
                                seed=getattr(self, "seed", None)
                                if getattr(self, "seed", None) is not None
                                else NOT_GIVEN,
                                tools=NOT_GIVEN,
                            )
                            text = ""
                            try:
                                text = completion.choices[0].message.content or ""
                            except Exception:
                                text = ""
                            if not text:
                                try:
                                    rc = getattr(
                                        completion.choices[0].message,
                                        "reasoning_content",
                                        None,
                                    )
                                    if isinstance(rc, str) and rc.strip():
                                        text = rc
                                except Exception:
                                    pass
                            if text:
                                yield text
                                return
                        except Exception:
                            pass
            except Exception:
                pass
            logger.error(f"LLM API: Error occurred: {e}")
            logger.info(f"Base URL: {self.base_url}")
            logger.info(f"Model: {self.model}")
            logger.info(f"Messages: {messages}")
            logger.info(f"temperature: {self.temperature}")
            yield "Error calling the chat endpoint: Error occurred while generating response. See the logs for details."

        finally:
            # make sure the stream is properly closed
            # so when interrupted, no more tokens will being generated.
            if stream:
                logger.debug("Chat completion finished.")
                await stream.close()
                logger.debug("Stream closed.")
