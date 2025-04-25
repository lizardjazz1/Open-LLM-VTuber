import json
from typing import AsyncIterator, List, Dict, Any, Callable, Literal, Union, Optional, Tuple
from loguru import logger
import types
import datetime

from .agent_interface import AgentInterface
from ..output_types import SentenceOutput, DisplayText
from ..stateless_llm.stateless_llm_interface import StatelessLLMInterface
from ..stateless_llm.claude_llm import AsyncLLM as ClaudeAsyncLLM
from ..stateless_llm.openai_compatible_llm import AsyncLLM as OpenAICompatibleAsyncLLM
from ...chat_history_manager import get_history
from ..transformers import (
    sentence_divider,
    actions_extractor,
    tts_filter,
    display_processor,
)
from ...config_manager import TTSPreprocessorConfig
from ..input_types import BatchInput, TextSource, ImageSource
from prompts import prompt_loader
from ...mcpp.client import MCPClient
from ...mcpp.server_manager import MCPServerManager
from ...mcpp.tool_manager import ToolManager
from ...mcpp.json_detector import StreamJSONDetector
from ...mcpp.types import ToolCallObject


class BasicMemoryAgent(AgentInterface):
    """
    Agent with basic chat memory using a list to store messages.
    Implements text-based responses with sentence processing pipeline.
    Supports native tool calling (Claude, OpenAI) and fallback prompt-based tool calling.
    """

    _system: str = "You are a helpful assistant."

    def __init__(
        self,
        llm: StatelessLLMInterface,
        system: str,
        live2d_model,
        tts_preprocessor_config: TTSPreprocessorConfig = None,
        faster_first_response: bool = True,
        segment_method: str = "pysbd",
        use_mcpp: bool = False,
        mcp_prompt: str = None,
        interrupt_method: Literal["system", "user"] = "user",
        tool_prompts: Dict[str, str] = None,
    ):
        """
        Initialize the agent with LLM, system prompt and configuration

        Args:
            llm: `StatelessLLMInterface` - The LLM to use
            system: `str` - System prompt
            live2d_model: `Live2dModel` - Model for expression extraction
            tts_preprocessor_config: `TTSPreprocessorConfig` - Configuration for TTS preprocessing
            faster_first_response: `bool` - Whether to enable faster first response
            segment_method: `str` - Method for sentence segmentation
            use_mcpp: `bool` - Whether to enable MCP Plus for tool execution
            mcp_prompt: `str` - MCP prompt for prompt mode
            interrupt_method: `Literal["system", "user"]` -
                Methods for writing interruptions signal in chat history.
            tool_prompts: `Dict[str, str]` - Dictionary of tool prompts from system_config

        """
        super().__init__()
        self._memory = []
        self._live2d_model = live2d_model
        self._tts_preprocessor_config = tts_preprocessor_config
        self._faster_first_response = faster_first_response
        self._segment_method = segment_method
        self._use_mcpp = use_mcpp
        self._mcp_prompt = mcp_prompt
        self._mcp_server_manager = MCPServerManager() if use_mcpp else None
        self._tool_manager = ToolManager() if use_mcpp else None
        self._json_detector = StreamJSONDetector() if use_mcpp else None
        self.prompt_mode_flag = False
        self.interrupt_method = interrupt_method
        self._tool_prompts = tool_prompts or {}
        self._interrupt_handled = False
        self._set_llm(llm)
        self.set_system(system if system else self._system)
        logger.info("BasicMemoryAgent initialized.")
        if use_mcpp:
            if self._mcp_server_manager and self._tool_manager:
                logger.info("MCP Plus enabled with ToolManager and ServerManager.")
                if self._mcp_prompt:
                    logger.info("MCP Prompt mode is configured as a fallback.")
                if not self._json_detector:
                    logger.warning(
                        "MCP Plus enabled but JSON Detector failed to initialize (needed for prompt mode)."
                    )
            else:
                logger.warning(
                    "MCP Plus is enabled but ToolManager or ServerManager failed to initialize."
                )
        else:
            logger.info("MCP Plus (Tool Use) is disabled.")

    def _set_llm(self, llm: StatelessLLMInterface):
        """
        Set the (stateless) LLM to be used for chat completion.
        Instead of assigning directly to `self.chat`, store it to `_chat_function`
        so that the async method chat remains intact.

        Args:
            llm: StatelessLLMInterface - the LLM instance.
        """
        self._llm = llm
        self.chat = self._chat_function_factory()

    def set_system(self, system: str):
        """
        Set the system prompt
        system: str
            the system prompt
        """
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
        """
        Add a message to the memory. Stores only the primary text content for simplicity.
        Tool calls and results are handled during the API interaction, not explicitly stored here.

        Args:
            message: Message content (string or list of content items like text, images)
            role: Message role ('user' or 'assistant')
            display_text: Optional display information containing name and avatar
            skip_memory: If True, message will not be added to memory (for system prompts)
        """
        if skip_memory:
            logger.debug(
                f"Skipping adding {role} message to memory due to skip_memory flag"
            )
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
            logger.debug("Skipping empty assistant message for memory.")
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
            logger.debug("Skipping duplicate message addition to memory.")
            return

        self._memory.append(message_data)
        logger.trace(
            f"Added message to memory: Role={role}, Content='{text_content[:50]}...'"
        )

    def set_memory_from_history(self, conf_uid: str, history_uid: str) -> None:
        """Load the memory from chat history"""
        messages = get_history(conf_uid, history_uid)

        self._memory = []
        for msg in messages:
            role = "user" if msg["role"] == "human" else "assistant"
            content = msg["content"]
            if isinstance(content, str) and content:
                self._memory.append(
                    {
                        "role": role,
                        "content": content,
                    }
                )
            else:
                logger.warning(f"Skipping invalid message from history: {msg}")
        logger.info(f"Loaded {len(self._memory)} messages from history.")

    def handle_interrupt(self, heard_response: str) -> None:
        """
        Handle an interruption by the user.

        Args:
            heard_response: str - The part of the AI response heard by the user before interruption
        """
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
        logger.info(
            f"Handled interrupt. Added interruption signal as role '{interrupt_role}'."
        )

    def _to_text_prompt(self, input_data: BatchInput) -> str:
        """
        Format BatchInput into a text prompt string (primarily for non-image LLMs or text part).

        Args:
            input_data: BatchInput - The input data containing texts and images

        Returns:
            str - Formatted message string
        """
        message_parts = []

        for text_data in input_data.texts:
            if text_data.source == TextSource.INPUT:
                message_parts.append(text_data.content)
            elif text_data.source == TextSource.CLIPBOARD:
                message_parts.append(
                    f"[User shared content from clipboard: {text_data.content}]"
                )

        if input_data.images:
            message_parts.append("\n[User has also provided images]")

        return "\n".join(message_parts).strip()

    def _to_messages(self, input_data: BatchInput) -> List[Dict[str, Any]]:
        """
        Prepare the initial list of messages for the LLM API call,
        combining history and the current user input (including images).
        Uses the standard 'image_url' format. Claude adapter will convert if needed.
        """
        messages = self._memory.copy()
        user_content = []
        text_prompt = self._to_text_prompt(input_data)
        if text_prompt:
            user_content.append({"type": "text", "text": text_prompt})

        # Use image_url format - expected by OpenAI and convertible by Claude adapter
        if input_data.images:
            image_added = False
            for img_data in input_data.images:
                # Directly use the data URI assuming it's correctly formatted
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

            # Check if we should skip adding to memory
            skip_memory = False
            if input_data.metadata and input_data.metadata.get("skip_memory", False):
                skip_memory = True
                logger.debug(
                    "Skipping adding input to AI's internal memory due to skip_memory flag"
                )

            if not skip_memory:
                self._add_message(
                    text_prompt if text_prompt else "[User provided image(s)]", "user"
                )
        else:
            logger.warning("No content generated for user message.")

        return messages

    async def _execute_tools(
        self,
        tool_calls: Union[List[Dict[str, Any]], List[ToolCallObject]],
        caller_mode: Literal["Claude", "OpenAI", "Prompt"],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Executes tools, returns LLM results and status update events."""
        llm_results = []
        status_updates = []

        if not self._mcp_server_manager or not self._tool_manager:
            logger.error("MCP Server/Tool Manager not available, cannot execute tools.")
            for i, call in enumerate(tool_calls):
                tool_id = getattr(call, "id", call.get("id", f"error_{i}_{datetime.datetime.utcnow().isoformat()}"))
                tool_name = getattr(call, "function.name", call.get("name", "Unknown"))
                error_content = "Error: Tool execution environment not configured."
                status_updates.append({
                    "type": "tool_call_status",
                    "tool_id": tool_id,
                    "tool_name": tool_name,
                    "status": "error",
                    "content": error_content,
                    "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
                })
            return llm_results, status_updates

        logger.info(
            f"Attempting to execute {len(tool_calls)} tool(s) for {caller_mode} caller."
        )
        async with MCPClient(self._mcp_server_manager) as client:
            for call in tool_calls:
                (
                    tool_name,
                    tool_id,
                    tool_input,
                    is_error,
                    result_content,
                    parse_error,
                ) = self._parse_tool_call(call)

                if parse_error:
                    logger.warning(
                        f"Skipping tool call due to parsing error: {result_content}"
                    )
                    status_updates.append({
                        "type": "tool_call_status",
                        "tool_id": tool_id or f"parse_error_{datetime.datetime.utcnow().isoformat()}",
                        "tool_name": tool_name or "Unknown Tool",
                        "status": "error",
                        "content": result_content,
                        "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
                    })
                else:
                    status_updates.append({
                        "type": "tool_call_status",
                        "tool_id": tool_id,
                        "tool_name": tool_name,
                        "status": "running",
                        "content": f"Input: {json.dumps(tool_input)}",
                        "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
                    })

                    is_error, result_content = await self._run_single_tool(
                        client, tool_name, tool_id, tool_input
                    )

                    status_updates.append({
                        "type": "tool_call_status",
                        "tool_id": tool_id,
                        "tool_name": tool_name,
                        "status": "error" if is_error else "completed",
                        "content": result_content,
                        "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
                    })

                formatted_result = self._format_tool_result(
                    caller_mode, tool_id, result_content, is_error
                )
                if formatted_result:
                    llm_results.append(formatted_result)

        logger.info(f"Finished executing tools. Returning {len(llm_results)} LLM result(s) and {len(status_updates)} status update(s).")
        return llm_results, status_updates

    def _parse_tool_call(self, call: Union[Dict[str, Any], ToolCallObject]) -> tuple:
        """Parses a tool call from either Claude or OpenAI format."""
        tool_name: str = ""
        tool_id: str = ""
        tool_input: Any = None
        is_error: bool = False
        result_content: str = ""
        parse_error: bool = False

        # Check for OpenAI native format first
        if isinstance(call, ToolCallObject):
            tool_name = call.function.name
            tool_id = call.id
            try:
                tool_input = json.loads(call.function.arguments)
            except json.JSONDecodeError:
                logger.error(
                    f"Failed to decode OpenAI tool arguments for '{tool_name}': {call.function.arguments}"
                )
                result_content = (
                    f"Error: Invalid arguments format for tool '{tool_name}'."
                )
                is_error = True
                parse_error = True
        # Check for Claude native format OR Prompt mode simple dict format
        elif isinstance(call, dict):
            tool_id = call.get("id")
            tool_name = call.get("name")
            # Claude passes 'input', prompt mode passes 'args'
            tool_input = call.get("input", call.get("args"))

            # Handle case where tool_input is None or empty
            if tool_input is None:
                logger.warning(
                    f"Empty input for tool '{tool_name}' (ID: {tool_id}). Using empty object."
                )
                tool_input = {}  # Default to empty object

            if not tool_id or not tool_name:
                logger.error(
                    f"Invalid Dict tool call structure (Claude or Prompt): {call}"
                )
                result_content = "Error: Invalid tool call structure from LLM."
                is_error = True
                parse_error = True
        else:
            logger.error(f"Unsupported tool call type: {type(call)}")
            result_content = "Error: Unsupported tool call type."
            is_error = True
            parse_error = True

        return tool_name, tool_id, tool_input, is_error, result_content, parse_error

    async def _run_single_tool(
        self, client: MCPClient, tool_name: str, tool_id: str, tool_input: Any
    ) -> tuple[bool, str]:
        """Runs a single tool using the MCPClient."""
        logger.info(
            f"Executing tool: {tool_name} (ID: {tool_id}) with input: {tool_input}"
        )
        tool_info = self._tool_manager.get_tool(tool_name)
        is_error = False
        result_content = ""

        # Ensure tool_input is never None
        if tool_input is None:
            logger.warning(
                f"Null tool_input received for '{tool_name}'. Using empty object."
            )
            tool_input = {}

        if not tool_info:
            logger.error(f"Tool '{tool_name}' not found in ToolManager.")
            result_content = f"Error: Tool '{tool_name}' is not available."
            is_error = True
        else:
            try:
                await client.connect_to_server(tool_info.related_server)
                tool_request_object = types.SimpleNamespace(
                    name=tool_name,
                    args=tool_input,
                    id=tool_id,
                    server=tool_info.related_server,
                )
                response = await client.call_tool(tool_request_object)
                result_content = str(response)
                # is_error remains False
                logger.info(f"Tool '{tool_name}' executed successfully.")
                logger.debug(f"Tool '{tool_name}' result: {result_content}")
            except Exception as e:
                logger.exception(
                    f"Error executing tool '{tool_name}' (ID: {tool_id}): {e}"
                )
                result_content = f"Error executing tool '{tool_name}': {e}"
                is_error = True
        return is_error, result_content

    def _format_tool_result(
        self,
        caller_mode: Literal["Claude", "OpenAI", "Prompt"],
        tool_id: str,
        result_content: str,
        is_error: bool,
    ) -> Dict[str, Any] | None:
        """Formats the tool execution result for the specific LLM API."""
        if caller_mode == "Claude":
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": result_content,
                "is_error": is_error,
            }
        elif caller_mode == "OpenAI":
            return {"role": "tool", "tool_call_id": tool_id, "content": result_content}
        elif caller_mode == "Prompt":
            logger.debug(
                f"Formatting result for Prompt mode. ID: {tool_id}, Error: {is_error}"
            )
            return {
                "tool_id": tool_id,
                "content": result_content,
                "is_error": is_error,
            }
        else:
            logger.debug(
                f"Formatting result for Prompt mode. ID: {tool_id}, Error: {is_error}"
            )
            return {
                "tool_id": tool_id,
                "content": result_content,
                "is_error": is_error,
            }

    async def _claude_tool_interaction_loop(
        self,
        initial_messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        """Handles the interaction loop with Claude, yielding text or tool status events."""
        messages = initial_messages.copy()
        current_turn_text = ""
        pending_tool_calls = []
        current_assistant_message_content = []
        stop_reason = None

        while True:
            logger.debug(f"Calling Claude API. Current message count: {len(messages)}")
            stream = self._llm.chat_completion(messages, self._system, tools=tools)
            pending_tool_calls.clear()
            current_assistant_message_content.clear()

            async for event in stream:
                if event["type"] == "text_delta":
                    text = event["text"]
                    current_turn_text += text
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
                elif event["type"] == "tool_use_start":
                    logger.debug(
                        f"Tool use started: {event['data']['name']} (ID: {event['data']['id']})"
                    )
                elif event["type"] == "tool_use_complete":
                    tool_call_data = event["data"]
                    logger.info(
                        f"Tool use request received: {tool_call_data['name']} (ID: {tool_call_data['id']})"
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
                elif event["type"] == "message_delta":
                    if event["data"]["delta"].get("stop_reason"):
                        stop_reason = event["data"]["delta"].get("stop_reason")
                        logger.debug(f"Message delta stop reason: {stop_reason}")
                elif event["type"] == "message_stop":
                    logger.debug("Message stop event received.")
                    break
                elif event["type"] == "error":
                    logger.error(f"LLM API Error: {event['message']}")
                    yield f"[Error from LLM: {event['message']}]"
                    return

            logger.debug(
                f"Finished processing stream segment. Stop reason: {stop_reason}"
            )

            if pending_tool_calls:
                logger.info(
                    f"Claude requested {len(pending_tool_calls)} tool(s). Executing..."
                )
                filtered_assistant_content = [
                    block
                    for block in current_assistant_message_content
                    if not (
                        block.get("type") == "text"
                        and not block.get("text", "").strip()
                    )
                ]

                if filtered_assistant_content:
                    logger.debug(
                        f"Appending assistant message with content: {filtered_assistant_content}"
                    )
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
                else:
                    logger.warning(
                        "Tool calls pending but no valid assistant content (text/tool_use) was generated."
                    )

                llm_results, status_updates = await self._execute_tools(
                    pending_tool_calls,
                    caller_mode="Claude",
                )

                for update in status_updates:
                    yield update

                if not llm_results:
                    logger.error(
                        "Claude tool execution failed to produce LLM results. Stopping interaction."
                    )
                    return

                messages.append({"role": "user", "content": llm_results})
                logger.debug(
                    f"Appended Claude LLM tool results. New message count: {len(messages)}"
                )

                stop_reason = None
                continue
            else:
                logger.info("No tool calls requested by Claude. Interaction complete.")
                if current_turn_text:
                    self._add_message(current_turn_text, "assistant")
                else:
                    logger.info("No text generated in the final response.")
                return

    def _process_tool_from_prompt_json(
        self, data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process tool data detected from JSON in prompt mode LLM response.
        Returns a list of simplified tool call dicts.
        """
        parsed_tools = []
        for item in data:
            server = item.get("mcp_server")
            tool_name = item.get("tool")
            arguments_str = item.get("arguments")
            if all([server, tool_name, arguments_str]):
                try:
                    args_dict = json.loads(arguments_str)
                    parsed_tools.append(
                        {
                            "name": tool_name,
                            "server": server,
                            "args": args_dict,
                            "id": f"prompt_tool_{len(parsed_tools)}",
                        }
                    )
                    logger.info(f"Parsed tool call from prompt JSON: {tool_name}")
                except json.JSONDecodeError:
                    logger.error(
                        f"Failed to decode arguments JSON in prompt mode tool call: {arguments_str}"
                    )
                except Exception as e:
                    logger.error(
                        f"Error processing prompt mode tool dict: {item} - {e}"
                    )
            else:
                logger.warning(
                    f"Skipping invalid tool structure in prompt mode JSON: {item}"
                )
        return parsed_tools

    async def _openai_tool_interaction_loop(
        self,
        initial_messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        """Handles OpenAI interaction, yielding text or tool status events."""
        messages = initial_messages.copy()
        current_turn_text = ""
        pending_tool_calls: Union[List[ToolCallObject], List[Dict[str, Any]]] = []
        current_system_prompt = self._system

        while True:
            if self.prompt_mode_flag:
                logger.info("OpenAI loop running in prompt mode.")
                if self._mcp_prompt:
                    current_system_prompt = f"{self._system}\n\n{self._mcp_prompt}"
                else:
                    logger.warning(
                        "Prompt mode active but mcp_prompt is not configured!"
                    )
                    current_system_prompt = self._system
                tools_for_api = None
            else:
                current_system_prompt = self._system
                tools_for_api = tools

            logger.debug(
                f"Calling OpenAI compatible API. Prompt Mode: {self.prompt_mode_flag}. Message count: {len(messages)}"
            )
            stream = self._llm.chat_completion(
                messages, current_system_prompt, tools=tools_for_api
            )
            pending_tool_calls.clear()
            current_turn_text = ""
            assistant_message_for_api = None
            detected_prompt_json = None
            goto_next_while_iteration = False

            async for event in stream:
                if self.prompt_mode_flag:
                    if isinstance(event, str):
                        current_turn_text += event
                        if self._json_detector:
                            potential_json = self._json_detector.process_chunk(event)
                            if potential_json:
                                logger.info(
                                    "Detected potential JSON tool call in prompt mode stream."
                                )
                                try:
                                    if isinstance(potential_json, list):
                                        detected_prompt_json = potential_json
                                    elif isinstance(potential_json, dict):
                                        detected_prompt_json = [potential_json]
                                    else:
                                        logger.warning(
                                            f"JSON detector returned unexpected type: {type(potential_json)}"
                                        )

                                    if detected_prompt_json:
                                        logger.debug(
                                            f"Successfully parsed JSON: {detected_prompt_json}"
                                        )
                                        break
                                except Exception as e:
                                    logger.error(
                                        f"Error fully parsing detected JSON: {e}"
                                    )
                                    self._json_detector.reset()
                        yield event
                    else:
                        logger.warning(
                            f"Received non-string event in prompt mode stream: {type(event)}"
                        )
                else:
                    if isinstance(event, str):
                        current_turn_text += event
                        yield event
                    elif isinstance(event, list) and all(isinstance(tc, ToolCallObject) for tc in event):
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
                            f"LLM {getattr(self._llm, 'model', '')} reported no native tool support. Switching to prompt mode."
                        )
                        self.prompt_mode_flag = True
                        if self._tool_manager:
                            self._tool_manager.disable()
                        if self._json_detector:
                            self._json_detector.reset()
                        goto_next_while_iteration = True
                        break
                    else:
                        logger.warning(
                            f"Received unexpected event type from native OpenAI stream: {type(event)}"
                        )
            if goto_next_while_iteration:
                logger.info("Restarting interaction loop iteration to apply prompt mode.")
                continue

            if detected_prompt_json:
                logger.info("Processing tools detected via prompt mode JSON.")
                self._add_message(current_turn_text, "assistant")

                parsed_tools = self._process_tool_from_prompt_json(detected_prompt_json)
                if parsed_tools:
                    llm_results, status_updates = await self._execute_tools(
                        parsed_tools,
                        caller_mode="Prompt",
                    )
                    for update in status_updates:
                        yield update

                    if llm_results:
                        result_strings = [res.get("content", "Error: Malformed result") for res in llm_results]
                        combined_results_str = "\n".join(result_strings)
                        messages.append({"role": "user", "content": combined_results_str})
                        logger.debug(f"Appended prompt mode LLM tool results. Msg count: {len(messages)}")
                    else:
                        if not any(upd['status'] == 'error' for upd in status_updates):
                           logger.error("Prompt mode tool execution finished but produced no results for LLM.")
                continue

            elif pending_tool_calls and assistant_message_for_api:
                messages.append(assistant_message_for_api)
                if current_turn_text:
                    self._add_message(
                        current_turn_text, "assistant"
                    )

                logger.debug("Executing OpenAI native tools...")
                llm_results, status_updates = await self._execute_tools(
                    pending_tool_calls,
                    caller_mode="OpenAI",
                )

                for update in status_updates:
                    yield update

                if not llm_results:
                     logger.error("OpenAI native tool execution failed to produce results for LLM.")
                     return

                messages.extend(llm_results)
                logger.debug(f"Appended OpenAI native LLM tool results. Msg count: {len(messages)}")
                continue

            else:
                logger.info(
                    "No tool calls requested or fallback executed. Interaction complete for this turn."
                )
                if current_turn_text:
                    self._add_message(current_turn_text, "assistant")
                else:
                    logger.info("No text generated in the final response.")
                return

    def _chat_function_factory(
        self,
    ) -> Callable[[BatchInput], AsyncIterator[Union[SentenceOutput, Dict[str, Any]]]]:
        """
        Creates the chat pipeline. The decorated function yields text or tool status dicts.
        The final decorated result yields SentenceOutput or tool status dicts.
        """

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
            """
            Yields raw text chunks or tool status event dictionaries.
            Decorators above will process the text chunks.
            """
            self.reset_interrupt()
            self.prompt_mode_flag = False
            if self._tool_manager:
                self._tool_manager.enable()
            if self._json_detector:
                self._json_detector.reset()

            messages = self._to_messages(input_data)
            tools = None
            tool_mode = None
            llm_supports_native_tools = False

            if self._use_mcpp and self._tool_manager and self._mcp_server_manager:
                if isinstance(self._llm, ClaudeAsyncLLM):
                    logger.info("LLM is Claude type, preparing Claude tools.")
                    tool_mode = "Claude"
                    tools = self._tool_manager.get_all_tools(mode=tool_mode)
                    llm_supports_native_tools = True
                elif isinstance(self._llm, OpenAICompatibleAsyncLLM):
                    logger.info(
                        "LLM is OpenAI compatible type, preparing OpenAI tools (will check support via API)."
                    )
                    tool_mode = "OpenAI"
                    tools = self._tool_manager.get_all_tools(mode=tool_mode)
                    llm_supports_native_tools = (
                        True
                    )

                if llm_supports_native_tools and not tools:
                    logger.warning(
                        f"Tool Manager returned no tools for mode '{tool_mode}'. Tool usage might be limited."
                    )

            if self._use_mcpp and tool_mode == "Claude":
                async for output in self._claude_tool_interaction_loop(
                    messages, tools if tools else []
                ):
                    yield output
                return
            elif self._use_mcpp and tool_mode == "OpenAI":
                async for output in self._openai_tool_interaction_loop(
                    messages, tools if tools else []
                ):
                    yield output
                return
            else:
                logger.info(
                    "Starting simple chat completion (MCP disabled, unknown LLM, or no tools)."
                )
                token_stream = self._llm.chat_completion(messages, self._system)
                complete_response = ""
                async for event in token_stream:
                    text_chunk = ""
                    if isinstance(event, dict) and event.get("type") == "text_delta":
                        text_chunk = event.get("text", "")
                    elif isinstance(event, str):
                        text_chunk = event
                    else:
                        logger.warning(
                            f"Received unexpected event type in simple stream: {type(event)}"
                        )
                        logger.warning(f"Event content: {event}")
                        continue
                    if text_chunk:
                        yield text_chunk
                        complete_response += text_chunk
                if complete_response:
                    self._add_message(complete_response, "assistant")
                else:
                    logger.info("No text generated in simple chat completion.")

        return chat_with_memory

    async def chat(
        self,
        input_data: BatchInput,
    ) -> AsyncIterator[Union[SentenceOutput, Dict[str, Any]]]:
        """Runs the chat pipeline, yielding SentenceOutput or tool status events."""
        chat_func_decorated = self._chat_function_factory()
        async for output in chat_func_decorated(input_data):
            yield output

    def reset_interrupt(self) -> None:
        """
        Reset the interrupt handled flag for a new conversation.
        """
        logger.debug("Resetting interrupt flag.")
        self._interrupt_handled = False

    def start_group_conversation(
        self, human_name: str, ai_participants: List[str]
    ) -> None:
        """
        Start a group conversation by adding a system message that informs the AI about
        the conversation participants.

        Args:
            human_name: str - Name of the human participant
            ai_participants: List[str] - Names of other AI participants in the conversation
        """
        if not self._tool_prompts:
            logger.warning(
                "Tool prompts dictionary is not set. Cannot load group conversation prompt."
            )
            return

        other_ais = ", ".join(name for name in ai_participants)

        prompt_name = self._tool_prompts.get("group_conversation_prompt", "")

        if not prompt_name:
            logger.warning(
                "No group conversation prompt name found in tool_prompts. Continuing without group context."
            )
            return

        try:
            group_context = prompt_loader.load_util(prompt_name).format(
                human_name=human_name, other_ais=other_ais
            )
            self._memory.append({"role": "user", "content": group_context})
            logger.debug(
                f"Added group conversation context message to memory: '''{group_context}'''"
            )
        except FileNotFoundError:
            logger.error(f"Group conversation prompt file not found: {prompt_name}")
        except KeyError as e:
            logger.error(
                f"Missing formatting key in group conversation prompt '{prompt_name}': {e}"
            )
        except Exception as e:
            logger.error(
                f"Failed to load or format group conversation prompt '{prompt_name}': {e}"
            )
