from typing import (
    AsyncIterator,
    List,
    Dict,
    Any,
    Callable,
    Literal,
    Union,
    Optional,
)
from loguru import logger
import json

from .agent_interface import AgentInterface
from ..output_types import SentenceOutput, DisplayText
from ..stateless_llm.stateless_llm_interface import StatelessLLMInterface
from ..stateless_llm.claude_llm import AsyncLLM as ClaudeAsyncLLM
from ..stateless_llm.openai_compatible_llm import AsyncLLM as OpenAICompatibleAsyncLLM
from ...chat_history_manager import get_history, get_metadata, update_metadate
from ..transformers import (
    sentence_divider,
    actions_extractor,
    tts_filter,
    display_processor,
)
from ...config_manager import TTSPreprocessorConfig
from ..input_types import BatchInput, TextSource
from prompts import prompt_loader
from ...mcpp.tool_manager import ToolManager
from ...mcpp.json_detector import StreamJSONDetector
from ...mcpp.types import ToolCallObject
from ...mcpp.tool_executor import ToolExecutor
import asyncio
from ...debug_settings import ensure_log_sinks
import time
from ...logging_utils import truncate_and_hash

_DEBUG_WS_UNUSED, DEBUG_LLM = ensure_log_sinks()


class BasicMemoryAgent(AgentInterface):
    """Agent with basic chat memory and tool calling support."""

    _system: str = "You are a helpful assistant."

    # Keep a bounded short-term memory window to avoid context explosion
    _max_memory_messages: int = 30

    def __init__(
        self,
        llm: StatelessLLMInterface,
        system: str,
        live2d_model,
        tts_preprocessor_config: TTSPreprocessorConfig = None,
        faster_first_response: bool = True,
        segment_method: str = "pysbd",
        use_mcpp: bool = False,
        interrupt_method: Literal["system", "user"] = "user",
        tool_prompts: Dict[str, str] = None,
        tool_manager: Optional[ToolManager] = None,
        tool_executor: Optional[ToolExecutor] = None,
        mcp_prompt_string: str = "",
        summarize_max_tokens: int = 256,
        summarize_timeout_s: int = 25,
        sentiment_max_tokens: int = 96,
        sentiment_timeout_s: int = 12,
    ):
        """Initialize agent with LLM and configuration."""
        super().__init__()
        self._memory = []
        self._live2d_model = live2d_model
        self._tts_preprocessor_config = tts_preprocessor_config
        self._faster_first_response = faster_first_response
        self._segment_method = segment_method
        self._use_mcpp = use_mcpp
        self.interrupt_method = interrupt_method
        self._tool_prompts = tool_prompts or {}
        self._interrupt_handled = False
        self.prompt_mode_flag = False

        self._tool_manager = tool_manager
        self._tool_executor = tool_executor
        self._mcp_prompt_string = mcp_prompt_string
        self._json_detector = StreamJSONDetector()

        # Configurable LLM limits/timeouts for memory ops
        self._summarize_max_tokens = int(max(32, summarize_max_tokens))
        self._summarize_timeout_s = int(max(5, summarize_timeout_s))
        self._sentiment_max_tokens = int(max(32, sentiment_max_tokens))
        self._sentiment_timeout_s = int(max(5, sentiment_timeout_s))

        self._formatted_tools_openai = []
        self._formatted_tools_claude = []
        if self._tool_manager:
            self._formatted_tools_openai = self._tool_manager.get_formatted_tools(
                "OpenAI"
            )
            self._formatted_tools_claude = self._tool_manager.get_formatted_tools(
                "Claude"
            )
            logger.debug(
                f"Agent received pre-formatted tools - OpenAI: {len(self._formatted_tools_openai)}, Claude: {len(self._formatted_tools_claude)}"
            )
        else:
            logger.debug(
                "ToolManager not provided, agent will not have pre-formatted tools."
            )

        self._set_llm(llm)
        self.set_system(system if system else self._system)

        if self._use_mcpp and not all(
            [
                self._tool_manager,
                self._tool_executor,
                self._json_detector,
            ]
        ):
            logger.warning(
                "use_mcpp is True, but some MCP components are missing in the agent. Tool calling might not work as expected."
            )
        elif not self._use_mcpp and any(
            [
                self._tool_manager,
                self._tool_executor,
                self._json_detector,
            ]
        ):
            logger.warning(
                "use_mcpp is False, but some MCP components were passed to the agent."
            )

        logger.info("BasicMemoryAgent initialized.")

    def _set_llm(self, llm: StatelessLLMInterface):
        """Set the LLM for chat completion."""
        self._llm = llm
        self.chat = self._chat_function_factory()

    def set_system(self, system: str):
        """Set the system prompt."""
        logger.debug(f"Memory Agent: Setting system prompt: '''{system}'''")

        if self.interrupt_method == "user":
            system = f"{system}\n\nIf you received `[interrupted by user]` signal, you were interrupted."

        self._system = system

    def _add_message(
        self,
        message: Union[str, List[Dict[str, Any]]],
        role: str,
        display_text: DisplayText | None = None,
        skip_memory: bool = False,
    ):
        """Add message to memory."""
        if skip_memory:
            return

        text_content = ""
        if isinstance(message, list):
            for item in message:
                if item.get("type") == "text":
                    text_content += item["text"] + " "
            text_content = text_content.strip()
        elif isinstance(message, str):
            text_content = message
        else:
            logger.warning(
                f"_add_message received unexpected message type: {type(message)}"
            )
            text_content = str(message)

        if not text_content and role == "assistant":
            return

        message_data = {
            "role": role,
            "content": text_content,
        }

        if display_text:
            if display_text.name:
                message_data["name"] = display_text.name
            if display_text.avatar:
                message_data["avatar"] = display_text.avatar

        if (
            self._memory
            and self._memory[-1]["role"] == role
            and self._memory[-1]["content"] == text_content
        ):
            return

        self._memory.append(message_data)

    def set_memory_from_history(self, conf_uid: str, history_uid: str) -> None:
        """Load memory from chat history."""
        messages = get_history(conf_uid, history_uid)

        # Optional: load previously stored summary from metadata
        summary_text = None
        try:
            meta = get_metadata(conf_uid, history_uid) or {}
            summary_text = meta.get("summary")
        except Exception:
            summary_text = None

        self._memory = []
        if summary_text:
            # Prepend summary to guide the assistant without consuming many tokens
            self._memory.append(
                {
                    "role": "system",
                    "content": f"Conversation summary so far: {summary_text}",
                }
            )

        # Take only the latest N messages (excluding metadata already filtered)
        tail = (
            messages[-self._max_memory_messages :]
            if self._max_memory_messages > 0
            else messages
        )
        for msg in tail:
            role = "user" if msg["role"] == "human" else "assistant"
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                self._memory.append({"role": role, "content": content})
            else:
                logger.debug("Skipping empty/non-string message from history")

        logger.info(
            f"Loaded {len(self._memory)} messages from history (window size {self._max_memory_messages}{' + summary' if summary_text else ''})."
        )

    def update_history_summary(self, conf_uid: str, history_uid: str) -> None:
        """Create a compact rolling summary and store it in metadata.

        This is a lightweight, model-free summarization that keeps only
        short, recent highlights to prime long-term memory.
        """
        try:
            msgs = get_history(conf_uid, history_uid)
            if not msgs:
                return
            # Take a small balanced window: more user than assistant
            recent = msgs[-40:]
            user_chunks = []
            ai_chunks = []
            for m in recent:
                role = m.get("role")
                content = (m.get("content") or "").strip()
                if not content:
                    continue
                if role == "human" and len(user_chunks) < 16:
                    user_chunks.append(content)
                elif role == "ai" and len(ai_chunks) < 8:
                    ai_chunks.append(content)

            def compress(parts, prefix):
                out = []
                for p in parts:
                    p = p.replace("\n", " ").strip()
                    if len(p) > 160:
                        p = p[:157] + "…"
                    out.append(f"{prefix} {p}")
                return out

            lines = compress(user_chunks, "Пользователь:") + compress(
                ai_chunks, "VTuber:"
            )
            summary = " | ".join(lines)
            if len(summary) > 700:
                summary = summary[:697] + "…"

            update_metadate(conf_uid, history_uid, {"summary": summary})
        except Exception as e:
            logger.debug(f"update_history_summary skipped: {e}")

    async def summarize_texts(self, texts: list[str]) -> dict:
        """Summarize and categorize a set of texts into memory buckets.

        Returns structured dict with keys: facts_about_user, past_events, self_beliefs, objectives, emotions, key_facts.
        """
        try:
            if not texts:
                return {}
            llm = getattr(self, "_llm", None)
            if not llm or not getattr(llm, "client", None):
                return {}
            joined = "\n".join([t.strip() for t in texts if t and isinstance(t, str)])[
                :8000
            ]
            # Prefer externalized consolidation prompt if configured
            system = None
            try:
                util_name = (self._tool_prompts or {}).get(
                    "memory_consolidation_prompt", ""
                )
                if util_name:
                    system = prompt_loader.load_util(util_name)
            except Exception:
                system = None
            if not system:
                system = (
                    "You are a memory organizer for a virtual VTuber. Read conversation excerpts and extract only concise, factual, non-redundant memory."
                    " Output strict JSON with fields: facts_about_user[], past_events[], self_beliefs[], objectives[], emotions[{name,score}], key_facts[{text,importance,tags[]}]."
                    " Use Russian language. importance in [0,1]. Keep items short."
                )
            user = (
                "Тексты диалога (последние сообщения):\n"
                + joined
                + "\n\nЗадача: кратко выдели: факты о пользователе, события, убеждения персонажа, цели; эмоции (name/score); ключевые факты (text/importance/tags)."
            )
            if DEBUG_LLM:
                try:
                    logger.bind(dst="llm").log(
                        "INFO",
                        (
                            f"LLM OUT (basic_memory_agent): model={llm.model if hasattr(llm, 'model') else 'unknown'}, "
                            f"messages={[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]}"
                        ),
                    )
                except Exception:
                    pass
            # Ограничим длину ответа и введём таймаут (из конфигурации)
            try:
                resp = await asyncio.wait_for(
                    llm.client.chat.completions.create(
                        model=llm.model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        temperature=0.2,
                        top_p=getattr(llm, "top_p", 1.0),
                        max_tokens=self._summarize_max_tokens,
                    ),
                    timeout=float(self._summarize_timeout_s),
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "summarize_texts: LLM timeout (25s), returning empty summary"
                )
                return {}
            content = (resp.choices and resp.choices[0].message.content) or "{}"
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
            # best-effort fallback: try to extract JSON substring
            import re

            m = re.search(r"\{[\s\S]*\}", content)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    return {}
            return {}
        except Exception as e:
            logger.debug(f"summarize_texts failed: {e}")
            return {}

    async def analyze_sentiment(self, text: str) -> dict:
        """Analyze sentiment of a single message and return {label, score[-1,1]} in Russian.

        Returns an empty dict on failure.
        """
        try:
            if not text or not isinstance(text, str):
                return {}
            llm = getattr(self, "_llm", None)
            if not llm or not getattr(llm, "client", None):
                return {}
            system = (
                "Ты детектор настроения и токсичности. Классифицируй короткий текст. "
                'Верни строгий JSON: {"label": string, "score": number от -1 до 1}. '
                "score < 0 — негатив/оскорбление, > 0 — позитив/поддержка, около 0 — нейтрально. "
                "Только JSON, без пояснений."
            )
            user = f"Текст: {text}\nЗадача: Дай label (например: 'оскорбление', 'поддержка', 'нейтрально') и score в [-1,1]."
            if DEBUG_LLM:
                try:
                    logger.bind(dst="llm").log(
                        "INFO",
                        (
                            f"LLM OUT (basic_memory_agent): model={llm.model if hasattr(llm, 'model') else 'unknown'}, "
                            f"messages={[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]}"
                        ),
                    )
                except Exception:
                    pass
            try:
                resp = await asyncio.wait_for(
                    llm.client.chat.completions.create(
                        model=llm.model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        temperature=0.2,
                        top_p=getattr(llm, "top_p", 1.0),
                        max_tokens=self._sentiment_max_tokens,
                    ),
                    timeout=float(self._sentiment_timeout_s),
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "analyze_sentiment: LLM timeout (12s), returning empty result"
                )
                return {}
            content = (resp.choices and resp.choices[0].message.content) or "{}"
            try:
                data = json.loads(content)
                if isinstance(data, dict) and "score" in data:
                    try:
                        s = float(data["score"])  # clamp to [-1,1]
                        data["score"] = max(-1.0, min(1.0, s))
                    except Exception:
                        data["score"] = 0.0
                    # normalize label
                    if not isinstance(data.get("label"), str):
                        data["label"] = "нейтрально"
                    return data
            except Exception:
                pass
            # fallback extract JSON
            import re

            m = re.search(r"\{[\s\S]*\}", content)
            if m:
                try:
                    data = json.loads(m.group(0))
                    if isinstance(data, dict):
                        if "score" in data:
                            try:
                                s = float(data["score"])
                                data["score"] = max(-1.0, min(1.0, s))
                            except Exception:
                                data["score"] = 0.0
                        if not isinstance(data.get("label"), str):
                            data["label"] = "нейтрально"
                        return data
                except Exception:
                    return {}
            return {}
        except Exception as e:
            logger.debug(f"analyze_sentiment failed: {e}")
            return {}

    def handle_interrupt(self, heard_response: str) -> None:
        """Handle user interruption."""
        if self._interrupt_handled:
            return

        self._interrupt_handled = True

        if self._memory and self._memory[-1]["role"] == "assistant":
            if not self._memory[-1]["content"].endswith("..."):
                self._memory[-1]["content"] = heard_response + "..."
            else:
                self._memory[-1]["content"] = heard_response + "..."
        else:
            if heard_response:
                self._memory.append(
                    {
                        "role": "assistant",
                        "content": heard_response + "...",
                    }
                )

        interrupt_role = "system" if self.interrupt_method == "system" else "user"
        self._memory.append(
            {
                "role": interrupt_role,
                "content": "[Interrupted by user]",
            }
        )
        logger.info(f"Handled interrupt with role '{interrupt_role}'.")

    def _to_text_prompt(self, input_data: BatchInput) -> str:
        """Format input data to text prompt."""
        message_parts = []

        for text_data in input_data.texts:
            if text_data.source == TextSource.INPUT:
                # Local/server-origin messages
                name = text_data.from_name or "User"
                message_parts.append(f"[Server:{name}] {text_data.content}")
            elif text_data.source == TextSource.CLIPBOARD:
                message_parts.append(
                    f"[User shared content from clipboard: {text_data.content}]"
                )
            elif text_data.source == TextSource.TWITCH:
                # Twitch prefix as requested
                nick = text_data.from_name or "User"
                message_parts.append(f"[Twitch:{nick}] {text_data.content}")
            else:
                # Future sources
                try:
                    from ..input_types import TextSource as _TS

                    if text_data.source == _TS.DISCORD:
                        nick = text_data.from_name or "User"
                        message_parts.append(f"[Discord:{nick}] {text_data.content}")
                    elif text_data.source == _TS.TELEGRAM:
                        nick = text_data.from_name or "User"
                        message_parts.append(f"[Telegram:{nick}] {text_data.content}")
                except Exception:
                    # Fallback
                    message_parts.append(text_data.content)

        if input_data.images:
            message_parts.append("\n[User has also provided images]")

        return "\n".join(message_parts).strip()

    def _to_messages(self, input_data: BatchInput) -> List[Dict[str, Any]]:
        """Prepare messages for LLM API call."""
        messages = self._memory.copy()
        user_content = []
        text_prompt = self._to_text_prompt(input_data)
        if text_prompt:
            user_content.append({"type": "text", "text": text_prompt})

        if input_data.images:
            image_added = False
            for img_data in input_data.images:
                if isinstance(img_data.data, str) and img_data.data.startswith(
                    "data:image"
                ):
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": img_data.data, "detail": "auto"},
                        }
                    )
                    image_added = True
                else:
                    logger.error(
                        f"Invalid image data format: {type(img_data.data)}. Skipping image."
                    )

            if not image_added and not text_prompt:
                logger.warning(
                    "User input contains images but none could be processed."
                )

        if user_content:
            user_message = {"role": "user", "content": user_content}
            messages.append(user_message)

            skip_memory = False
            if input_data.metadata and input_data.metadata.get("skip_memory", False):
                skip_memory = True

            if not skip_memory:
                self._add_message(
                    text_prompt if text_prompt else "[User provided image(s)]", "user"
                )
        else:
            logger.warning("No content generated for user message.")

        return messages

    async def _claude_tool_interaction_loop(
        self,
        initial_messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        """Handle Claude interaction loop with tool support."""
        messages = initial_messages.copy()
        current_turn_text = ""
        start_ts = time.perf_counter()
        first_token_ts: float | None = None
        chunk_count = 0
        char_count = 0
        pending_tool_calls = []
        current_assistant_message_content = []

        while True:
            stream = self._llm.chat_completion(messages, self._system, tools=tools)
            pending_tool_calls.clear()
            current_assistant_message_content.clear()

            # Получаем поток как асинхронный итератор
            if asyncio.iscoroutine(stream):
                stream = await stream

            async for event in stream:
                if event["type"] == "text_delta":
                    text = event["text"]
                    current_turn_text += text
                    if first_token_ts is None:
                        first_token_ts = time.perf_counter()
                    chunk_count += 1
                    char_count += len(text)
                    yield text
                    if (
                        not current_assistant_message_content
                        or current_assistant_message_content[-1]["type"] != "text"
                    ):
                        current_assistant_message_content.append(
                            {"type": "text", "text": text}
                        )
                    else:
                        current_assistant_message_content[-1]["text"] += text
                elif event["type"] == "tool_use_complete":
                    tool_call_data = event["data"]
                    logger.info(
                        f"Tool request: {tool_call_data['name']} (ID: {tool_call_data['id']})"
                    )
                    pending_tool_calls.append(tool_call_data)
                    current_assistant_message_content.append(
                        {
                            "type": "tool_use",
                            "id": tool_call_data["id"],
                            "name": tool_call_data["name"],
                            "input": tool_call_data["input"],
                        }
                    )
                # elif event["type"] == "message_delta":
                #     if event["data"]["delta"].get("stop_reason"):
                #         stop_reason = event["data"]["delta"].get("stop_reason")
                elif event["type"] == "message_stop":
                    break
                elif event["type"] == "error":
                    logger.error(f"LLM API Error: {event['message']}")
                    yield f"[Error from LLM: {event['message']}]"
                    return

            if pending_tool_calls:
                filtered_assistant_content = [
                    block
                    for block in current_assistant_message_content
                    if not (
                        block.get("type") == "text"
                        and not block.get("text", "").strip()
                    )
                ]

                if filtered_assistant_content:
                    messages.append(
                        {"role": "assistant", "content": filtered_assistant_content}
                    )
                    assistant_text_for_memory = "".join(
                        [
                            c["text"]
                            for c in filtered_assistant_content
                            if c["type"] == "text"
                        ]
                    ).strip()
                    if assistant_text_for_memory:
                        self._add_message(assistant_text_for_memory, "assistant")

                tool_results_for_llm = []
                if not self._tool_executor:
                    logger.error(
                        "Claude Tool interaction requested but ToolExecutor is not available."
                    )
                    yield "[Error: ToolExecutor not configured]"
                    return

                tool_executor_iterator = self._tool_executor.execute_tools(
                    tool_calls=pending_tool_calls,
                    caller_mode="Claude",
                )
                try:
                    while True:
                        update = await anext(tool_executor_iterator)
                        if update.get("type") == "final_tool_results":
                            tool_results_for_llm = update.get("results", [])
                            break
                        else:
                            yield update
                except StopAsyncIteration:
                    logger.warning(
                        "Tool executor finished without final results marker."
                    )

                if tool_results_for_llm:
                    messages.append({"role": "user", "content": tool_results_for_llm})

                # stop_reason = None
                continue
            else:
                if current_turn_text:
                    self._add_message(current_turn_text, "assistant")
                # Structured log of final assistant text with basic timings
                duration_ms = int((time.perf_counter() - start_ts) * 1000)
                ttft_ms = (
                    int((first_token_ts - start_ts) * 1000) if first_token_ts else None
                )
                logger.bind(
                    component="llm",
                    provider="claude",
                    model=getattr(self._llm, "model", None),
                    base_url=getattr(self._llm, "base_url", None),
                    ttft_ms=ttft_ms,
                    duration_ms=duration_ms,
                    chunks=chunk_count,
                    chars=char_count,
                ).info(
                    {"event": "llm_response", **truncate_and_hash(current_turn_text)}
                )
                return

    async def _openai_tool_interaction_loop(
        self,
        initial_messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        """Handle OpenAI interaction with tool support."""
        logger.info(f"🚀 Starting OpenAI tool interaction loop with {len(tools)} tools")
        messages = initial_messages.copy()
        current_turn_text = ""
        pending_tool_calls: Union[List[ToolCallObject], List[Dict[str, Any]]] = []
        current_system_prompt = self._system

        while True:
            if self.prompt_mode_flag:
                if self._mcp_prompt_string:
                    current_system_prompt = (
                        f"{self._system}\n\n{self._mcp_prompt_string}"
                    )
                else:
                    logger.warning("Prompt mode active but mcp_prompt_string is empty!")
                    current_system_prompt = self._system
                tools_for_api = None
            else:
                current_system_prompt = self._system
                tools_for_api = tools

            stream = self._llm.chat_completion(
                messages, current_system_prompt, tools=tools_for_api
            )
            pending_tool_calls.clear()
            current_turn_text = ""
            assistant_message_for_api = None
            detected_prompt_json = None
            goto_next_while_iteration = False

            # Получаем поток как асинхронный итератор
            if asyncio.iscoroutine(stream):
                stream = await stream

            async for event in stream:
                logger.debug(f"🔍 Processing event: {type(event)} = {event}")
                logger.debug(f"🔍 prompt_mode_flag: {self.prompt_mode_flag}")
                if self.prompt_mode_flag:
                    if isinstance(event, str):
                        current_turn_text += event
                        if self._json_detector:
                            potential_json = self._json_detector.process_chunk(event)
                            if potential_json:
                                try:
                                    if isinstance(potential_json, list):
                                        detected_prompt_json = potential_json
                                    elif isinstance(potential_json, dict):
                                        detected_prompt_json = [potential_json]

                                    if detected_prompt_json:
                                        break
                                except Exception as e:
                                    logger.error(f"Error parsing detected JSON: {e}")
                                    if self._json_detector:
                                        self._json_detector.reset()
                                    yield f"[Error parsing tool JSON: {e}]"
                                    goto_next_while_iteration = True
                                    break
                        yield event
                else:
                    logger.debug(
                        f"🔍 Processing event in else block: {type(event)} = {event}"
                    )
                    if isinstance(event, str):
                        current_turn_text += event
                        yield event
                    elif isinstance(event, list) and all(
                        isinstance(tc, ToolCallObject) for tc in event
                    ):
                        pending_tool_calls = event
                        assistant_message_for_api = {
                            "role": "assistant",
                            "content": current_turn_text if current_turn_text else None,
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": tc.type,
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                for tc in pending_tool_calls
                            ],
                        }
                        break
                    elif event == "__API_NOT_SUPPORT_TOOLS__":
                        logger.warning(
                            f"LLM {getattr(self._llm, 'model', '')} has no native tool support. Switching to prompt mode."
                        )
                        logger.info("🔄 Processing __API_NOT_SUPPORT_TOOLS__ signal")
                        logger.info(
                            f"🔄 Current prompt_mode_flag: {self.prompt_mode_flag}"
                        )
                        # Добавляем детальное логирование
                        if self._tool_manager:
                            available_tools = getattr(
                                self._tool_manager, "_formatted_tools_openai", []
                            )
                            tool_names = [
                                tool.get("function", {}).get("name", "unknown")
                                for tool in available_tools
                            ]
                            logger.warning(
                                f"Available tools that will be used in prompt mode: {tool_names}"
                            )
                        logger.warning(
                            "Prompt mode will use JSON detection for tool calls instead of native API support"
                        )
                        self.prompt_mode_flag = True
                        logger.info(
                            f"🔄 Set prompt_mode_flag to: {self.prompt_mode_flag}"
                        )
                        if self._tool_manager:
                            self._tool_manager.disable()
                        if self._json_detector:
                            self._json_detector.reset()
                        goto_next_while_iteration = True
                        break
            if goto_next_while_iteration:
                continue

            if detected_prompt_json:
                logger.info("Processing tools detected via prompt mode JSON.")
                self._add_message(current_turn_text, "assistant")

                parsed_tools = self._tool_executor.process_tool_from_prompt_json(
                    detected_prompt_json
                )
                if parsed_tools:
                    tool_results_for_llm = []
                    if not self._tool_executor:
                        logger.error(
                            "Prompt Tool interaction requested but ToolExecutor/MCPClient is not available."
                        )
                        yield "[Error: ToolExecutor/MCPClient not configured for prompt mode]"
                        continue

                    tool_executor_iterator = self._tool_executor.execute_tools(
                        tool_calls=parsed_tools,
                        caller_mode="Prompt",
                    )
                    try:
                        while True:
                            update = await anext(tool_executor_iterator)
                            if update.get("type") == "final_tool_results":
                                tool_results_for_llm = update.get("results", [])
                                break
                            else:
                                yield update
                    except StopAsyncIteration:
                        logger.warning(
                            "Prompt mode tool executor finished without final results marker."
                        )

                    if tool_results_for_llm:
                        result_strings = [
                            res.get("content", "Error: Malformed result")
                            for res in tool_results_for_llm
                        ]
                        combined_results_str = "\n".join(result_strings)
                        messages.append(
                            {"role": "user", "content": combined_results_str}
                        )
                        logger.info(
                            f"🔧 Tool results added to messages: {combined_results_str}"
                        )
                continue

            elif pending_tool_calls and assistant_message_for_api:
                messages.append(assistant_message_for_api)
                if current_turn_text:
                    self._add_message(current_turn_text, "assistant")

                tool_results_for_llm = []
                if not self._tool_executor:
                    logger.error(
                        "OpenAI Tool interaction requested but ToolExecutor/MCPClient is not available."
                    )
                    yield "[Error: ToolExecutor/MCPClient not configured for OpenAI mode]"
                    continue

                tool_executor_iterator = self._tool_executor.execute_tools(
                    tool_calls=pending_tool_calls,
                    caller_mode="OpenAI",
                )
                try:
                    while True:
                        update = await anext(tool_executor_iterator)
                        if update.get("type") == "final_tool_results":
                            tool_results_for_llm = update.get("results", [])
                            break
                        else:
                            yield update
                except StopAsyncIteration:
                    logger.warning(
                        "OpenAI tool executor finished without final results marker."
                    )

                if tool_results_for_llm:
                    messages.extend(tool_results_for_llm)
                    logger.info(
                        f"🔧 Tool results added to messages: {tool_results_for_llm}"
                    )
                continue

            else:
                if current_turn_text:
                    self._add_message(current_turn_text, "assistant")
                    logger.info(f"💬 Final response generated: {current_turn_text}")
                else:
                    logger.warning("❌ No response generated after tool execution")
                return

    def _chat_function_factory(
        self,
    ) -> Callable[[BatchInput], AsyncIterator[Union[SentenceOutput, Dict[str, Any]]]]:
        """Create the chat pipeline function."""

        @tts_filter(self._tts_preprocessor_config)
        @display_processor()
        @actions_extractor(self._live2d_model)
        @sentence_divider(
            faster_first_response=self._faster_first_response,
            segment_method=self._segment_method,
            valid_tags=["think"],
        )
        async def chat_with_memory(
            input_data: BatchInput,
        ) -> AsyncIterator[Union[str, Dict[str, Any]]]:
            """Process chat with memory and tools."""
            self.reset_interrupt()
            self.prompt_mode_flag = False

            messages = self._to_messages(input_data)
            tools = None
            tool_mode = None
            llm_supports_native_tools = False

            if self._use_mcpp and self._tool_manager:
                tools = None
                if isinstance(self._llm, ClaudeAsyncLLM):
                    tool_mode = "Claude"
                    tools = self._formatted_tools_claude
                    llm_supports_native_tools = True
                elif isinstance(self._llm, OpenAICompatibleAsyncLLM):
                    tool_mode = "OpenAI"
                    tools = self._formatted_tools_openai
                    llm_supports_native_tools = True
                else:
                    logger.warning(
                        f"LLM type {type(self._llm)} not explicitly handled for tool mode determination."
                    )

                if llm_supports_native_tools and not tools:
                    logger.warning(
                        f"No tools available/formatted for '{tool_mode}' mode, despite MCP being enabled."
                    )

            if self._use_mcpp and tool_mode == "Claude":
                logger.debug(
                    f"Starting Claude tool interaction loop with {len(tools)} tools."
                )
                async for output in self._claude_tool_interaction_loop(
                    messages, tools if tools else []
                ):
                    yield output
                return
            elif self._use_mcpp and tool_mode == "OpenAI":
                logger.debug(
                    f"Starting OpenAI tool interaction loop with {len(tools)} tools."
                )
                logger.info(
                    f"🔧 Tool mode: {tool_mode}, Tools count: {len(tools) if tools else 0}"
                )
                async for output in self._openai_tool_interaction_loop(
                    messages, tools if tools else []
                ):
                    yield output
                return
            else:
                logger.info("Starting simple chat completion.")
                token_stream = self._llm.chat_completion(messages, self._system)
                complete_response = ""
                async for event in token_stream:
                    text_chunk = ""
                    if isinstance(event, dict) and event.get("type") == "text_delta":
                        text_chunk = event.get("text", "")
                    elif isinstance(event, str):
                        text_chunk = event
                    else:
                        continue
                    if text_chunk:
                        yield text_chunk
                        complete_response += text_chunk
                if complete_response:
                    # Try to parse JSON response and extract the "response" field
                    try:
                        import json

                        parsed_json = json.loads(complete_response)
                        if isinstance(parsed_json, dict) and "response" in parsed_json:
                            # Extract only the response field from JSON
                            complete_response = parsed_json["response"]
                            logger.info(
                                f"Extracted response from JSON: {complete_response}"
                            )
                    except (json.JSONDecodeError, KeyError):
                        # If not valid JSON or no response field, use as-is
                        logger.debug(
                            "Response is not valid JSON or missing response field, using as-is"
                        )
                    # Deduplicate repeated sentences heuristically
                    try:
                        parts = [
                            p.strip()
                            for p in complete_response.replace("\n", " ").split(".")
                        ]
                        seen = set()
                        out = []
                        for p in parts:
                            if not p:
                                continue
                            key = p.lower()
                            if key in seen:
                                continue
                            seen.add(key)
                            out.append(p)
                        complete_response = ". ".join(out).strip()
                        if complete_response and not complete_response.endswith(
                            (".", "!", "?")
                        ):
                            complete_response += "."
                    except Exception:
                        pass

                    self._add_message(complete_response, "assistant")

        return chat_with_memory

    async def chat(
        self,
        input_data: BatchInput,
    ) -> AsyncIterator[Union[SentenceOutput, Dict[str, Any]]]:
        """Run chat pipeline."""
        chat_func_decorated = self._chat_function_factory()
        async for output in chat_func_decorated(input_data):
            yield output

    def reset_interrupt(self) -> None:
        """Reset interrupt flag."""
        self._interrupt_handled = False

    def start_group_conversation(
        self, human_name: str, ai_participants: List[str]
    ) -> None:
        """Start a group conversation."""
        if not self._tool_prompts:
            logger.warning("Tool prompts dictionary is not set.")
            return

        other_ais = ", ".join(name for name in ai_participants)
        prompt_name = self._tool_prompts.get("group_conversation_prompt", "")

        if not prompt_name:
            logger.warning("No group conversation prompt name found.")
            return

        try:
            group_context = prompt_loader.load_util(prompt_name).format(
                human_name=human_name, other_ais=other_ais
            )
            self._memory.append({"role": "user", "content": group_context})
        except FileNotFoundError:
            logger.error(f"Group conversation prompt file not found: {prompt_name}")
        except KeyError as e:
            logger.error(f"Missing formatting key in group conversation prompt: {e}")
        except Exception as e:
            logger.error(f"Failed to load group conversation prompt: {e}")
