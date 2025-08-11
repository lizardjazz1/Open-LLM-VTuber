"""Constructs prompts for servers and tools, formats tool information for OpenAI API."""

from typing import Dict, Optional, List, Tuple, Any
from loguru import logger

from .types import FormattedTool
from .mcp_client import MCPClient
from .server_registry import ServerRegistry


class ToolAdapter:
    """Dynamically fetches tool information from enabled MCP servers and formats it."""

    def __init__(self, server_registery: Optional[ServerRegistry] = None) -> None:
        """Initialize with an ServerRegistry."""
        self.server_registery = server_registery or ServerRegistry()

    async def get_server_and_tool_info(
        self, enabled_servers: List[str]
    ) -> Tuple[Dict[str, Dict[str, str]], Dict[str, FormattedTool]]:
        """Fetch tool information from specified enabled MCP servers."""
        servers_info: Dict[str, Dict[str, str]] = {}
        formatted_tools: Dict[str, FormattedTool] = {}

        if not enabled_servers:
            logger.warning(
                "MC: No enabled MCP servers specified. Cannot fetch tool info."
            )
            return servers_info, formatted_tools

        logger.debug(f"MC: Fetching tool info for enabled servers: {enabled_servers}")

        # Create a new MCPClient for tool fetching
        async with MCPClient(self.server_registery) as client:
            for server_name in enabled_servers:
                try:
                    logger.debug(f"MC: Fetching tools from server '{server_name}'...")
                    tools = await client.list_tools(server_name)
                    
                    # Initialize server info
                    servers_info[server_name] = {}
                    
                    # Format tools for this server
                    for tool in tools:
                        formatted_tool = self._format_tool(tool, server_name)
                        # Use tool.name as the key
                        formatted_tools[tool.name] = formatted_tool
                        
                        # Add tool info to servers_info
                        servers_info[server_name][tool.name] = {
                            "description": tool.description,
                            "parameters": tool.inputSchema.get("properties", {}),
                            "required": tool.inputSchema.get("required", [])
                        }
                        
                    logger.debug(f"MC: Successfully fetched {len(tools)} tools from server '{server_name}'")
                    
                except Exception as e:
                    logger.error(f"MC: Failed to fetch tools from server '{server_name}': {e}")
                    continue

        logger.info(f"MC: Total tools fetched: {len(formatted_tools)}")
        return servers_info, formatted_tools

    def _format_tool(self, tool: Any, server_name: str) -> FormattedTool:
        """Formats a single tool into a FormattedTool object."""
        return FormattedTool(
            input_schema=tool.inputSchema,
            related_server=server_name,
            description=tool.description,
            # Generic schema will be generated later if needed
            generic_schema=None,
        )

    def construct_mcp_prompt_string(
        self, servers_info: Dict[str, Dict[str, str]]
    ) -> str:
        """Build a single prompt string describing enabled servers and their tools."""
        full_prompt_content = ""
        if not servers_info:
            logger.warning(
                "MC: Cannot construct MCP prompt string, servers_info is empty."
            )
            return full_prompt_content

        logger.debug(
            f"MC: Constructing MCP prompt string for {len(servers_info)} server(s)."
        )

        for server_name, tools in servers_info.items():
            if not tools:  # Skip servers where info couldn't be fetched
                logger.warning(
                    f"MC: No tool info available for server '{server_name}', skipping in prompt."
                )
                continue

            prompt_content = f"Server: {server_name}\n"
            prompt_content += "    Tools:\n"
            for tool_name, tool_info in tools.items():
                prompt_content += f"        {tool_name}:\n"
                # Ensure description is handled correctly (might be None)
                description = tool_info.get("description", "No description available.")
                prompt_content += f"            Description: {description}\n"
                parameters = tool_info.get("parameters", {})
                if parameters:
                    prompt_content += "            Parameters:\n"
                    for param_name, param_info in parameters.items():
                        param_desc = param_info.get("description") or param_info.get(
                            "title", "No description provided."
                        )
                        param_type = param_info.get(
                            "type", "string"
                        )  # Default to string if type missing
                        prompt_content += f"                {param_name}:\n"
                        prompt_content += f"                    Type: {param_type}\n"
                        prompt_content += (
                            f"                    Description: {param_desc}\n"
                        )
                required = tool_info.get("required", [])
                if required:
                    prompt_content += f"            Required: {', '.join(required)}\n"
            full_prompt_content += prompt_content + "\n"  # Add newline between servers

        logger.debug("MC: Finished constructing MCP prompt string.")
        return full_prompt_content.strip()  # Remove trailing newline

    def format_tools_for_api(
        self, formatted_tools_dict: Dict[str, FormattedTool]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Format tools to OpenAI and Claude function-calling compatible schemas."""
        openai_tools = []
        claude_tools = []

        if not formatted_tools_dict:
            logger.warning(
                "MC: Cannot format tools for API, input dictionary is empty."
            )
            return openai_tools, claude_tools

        logger.debug(f"MC: Formatting {len(formatted_tools_dict)} tools for API usage.")

        for tool_name, data_object in formatted_tools_dict.items():
            if not isinstance(data_object, FormattedTool):
                logger.warning(f"MC: Skipping invalid tool format for '{tool_name}'")
                continue

            input_schema = data_object.input_schema
            properties: Dict[str, Dict[str, str]] = input_schema.get("properties", {})
            tool_description = data_object.description or "No description provided."
            required_params = input_schema.get("required", [])

            # Format for OpenAI
            openai_function_params = {
                "type": "object",
                "properties": {},
                "required": required_params,
                "additionalProperties": False,  # Disallow extra properties
            }
            for param_name, param_info in properties.items():
                param_schema = {
                    "type": param_info.get("type", "string"),
                    "description": param_info.get("description")
                    or param_info.get("title", "No description provided."),
                }
                # Add enum if present
                if "enum" in param_info:
                    param_schema["enum"] = param_info["enum"]
                # Handle array type correctly
                if param_schema["type"] == "array" and "items" in param_info:
                    param_schema["items"] = param_info["items"]
                elif param_schema["type"] == "array" and "items" not in param_info:
                    logger.warning(
                        f"MC: Array parameter '{param_name}' in tool '{tool_name}' is missing 'items' definition. Assuming items are strings."
                    )
                    param_schema["items"] = {"type": "string"}  # Default or log warning

                openai_function_params["properties"][param_name] = param_schema

            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool_description,
                        "parameters": openai_function_params,
                    },
                }
            )

            # Format for Claude
            claude_input_schema = {
                "type": "object",
                "properties": properties,
                "required": required_params,
            }
            claude_tools.append(
                {
                    "name": tool_name,
                    "description": tool_description,
                    "input_schema": claude_input_schema,
                }
            )

        logger.debug(
            f"MC: Finished formatting tools. OpenAI: {len(openai_tools)}, Claude: {len(claude_tools)}."
        )
        return openai_tools, claude_tools

    async def get_tools(
        self, enabled_servers: List[str]
    ) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Run the dynamic fetching and formatting process."""
        logger.info(
            f"MC: Running dynamic tool construction for servers: {enabled_servers}"
        )
        servers_info, formatted_tools_dict = await self.get_server_and_tool_info(
            enabled_servers
        )
        
        # Добавляем детальное логирование серверов и инструментов
        logger.info(f"MC: Servers info: {list(servers_info.keys())}")
        if formatted_tools_dict:
            tool_names = list(formatted_tools_dict.keys())
            logger.info(f"MC: Available tools: {tool_names}")
        else:
            logger.warning("MC: No tools found from any enabled servers.")

        # Construct MCP prompt string using servers_info
        mcp_prompt_string = self.construct_mcp_prompt_string(servers_info)

        # Format tools for OpenAI and Claude
        openai_tools = []
        claude_tools = []

        for tool_name, formatted_tool in formatted_tools_dict.items():
            # OpenAI format
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": formatted_tool.description,
                    "parameters": formatted_tool.input_schema,
                },
            }
            openai_tools.append(openai_tool)

            # Claude format (same as OpenAI for now)
            claude_tool = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": formatted_tool.description,
                    "parameters": formatted_tool.input_schema,
                },
            }
            claude_tools.append(claude_tool)

        logger.info(
            f"MC: Successfully formatted {len(openai_tools)} tools for OpenAI and {len(claude_tools)} for Claude."
        )

        return mcp_prompt_string, openai_tools, claude_tools
