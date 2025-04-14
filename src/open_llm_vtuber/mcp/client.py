"""MCP Client for Open-LLM-Vtuber.

Reference: https://modelcontextprotocol.io/quickstart/client
"""

from contextlib import AsyncExitStack
from typing import Optional, Dict, Any, List
from loguru import logger
from datetime import timedelta

from mcp import ClientSession, StdioServerParameters
from mcp.types import Tool
from mcp.client.stdio import stdio_client

from .server_manager import MCPServerManager


DEFAULT_TIMEOUT = timedelta(seconds=10)

class MCPClient:
    
    """MCP Client for Open-LLM-Vtuber.
    
    Usage:
    1. By async context manager:
    ```python
    async with MCPClient(auto_search=True) as client:
        await client.connect_to_server("example")
    ```
    2. By manual:
    ```python
    client = MCPClient(auto_search=True)
    await client.connect_to_server("example")
    await client.close()
    ```
    """
    
    def __init__(self, server_manager: MCPServerManager) -> None:
        """Initialize the MCP Client.
        
        Args:
            server_manager (MCPServerManager): The server manager to use for managing servers.
        
        Raises:
            ValueError: If the server manager is not an instance of MCPServerManager.
        """
        self.session: Optional[ClientSession] = None
        self.exit_stack: AsyncExitStack = AsyncExitStack()
        self.stdio, self.write = None, None
        
        if isinstance(server_manager, MCPServerManager):
            self.server_manager = server_manager
        else:
            raise ValueError("MCPC: Invalid server manager. Must be an instance of MCPServerManager.")
    
    
    async def connect_to_server(self, server_name: str, timeout: timedelta = DEFAULT_TIMEOUT) -> None:
        """Connect to the specified server.
        
        Args:
            server_name (str): The name of the server to connect to.
            timeout (datetime.timedelta): The timeout for the connection attempt. Default is 10 seconds.
        
        Raises:
            ValueError: If the server is not found in the available servers or not supported.
            RuntimeError: If node.js is not found in PATH and the server is a JavaScript server.
        """
        logger.debug(f"MCPC: Attempting to connect to server '{server_name}'...")
        # Initialize the server parameters.
        server = self.server_manager.get_server(server_name)
        if not server:
            logger.error(f"MCPC: Server '{server_name}' not found in available servers.")
            raise ValueError(f"MCPC: Server '{server_name}' not found in available servers.")
        
        executable = server["executable"]
        args = server["args"]
        env = server.get("env", None)
            
        server_params = StdioServerParameters(
            command=executable,
            args=args,
            env=env,
        )
        
        # Intialize the session.
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(
                self.stdio, self.write,
                read_timeout_seconds=timeout
            )
        )
        await self.session.initialize()

        logger.info(f"MCPC: Connected to server '{server_name}'.")
        
    
    async def list_tools(self) -> List[Tool]:
        """List all available tools on the connected server.
        
        Returns:
            List[mcp.types.Tool]: A list of tools available on the server.
        
        Raises:
            RuntimeError: If not connected to a server.
        """
        if self.session is None:
            raise RuntimeError("MCPC: Not connected to any server.")
        
        logger.debug("MCPC: Listing tools...")
        response = await self.session.list_tools()
        logger.debug(f"MCPC: Response from server: {response}")
        return response.tools
        
    
    async def call_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Call a tool on the connected server.
        
        Args:
            tool_name (str): The name of the tool to call.
            args (Dict[str, Any]): The arguments to pass to the tool.
            
        Returns:
            str: The result of the tool call.
        
        Raises:
            RuntimeError: If not connected to a server.
        """
        if self.session is None:
            raise RuntimeError("MCPC: Not connected to any server.")
        
        logger.info(f"MCPC: Calling tool '{tool_name}'...")
        logger.debug(f"MCPC: Tool args: {args}")
        
        response = await self.session.call_tool(tool_name, args)
        logger.debug(f"MCPC: Response from server: {response}")
        
        if response.isError:
            logger.error(f"MCPC: Error calling tool '{tool_name}': {response.content[0].text}")
            
        return response.content[0].text
    
    
    async def close(self) -> None:
        """Clean up the MCP client."""
        await self.exit_stack.aclose()
        self.session = None
        self.stdio, self.write = None, None
        logger.info("MCPC: Closed the client session.")
    
    
    async def __aenter__(self) -> "MCPClient":
        """Enter the async context manager."""
        return self

    
    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        """Exit the async context manager."""
        await self.close()
        if exc_type:
            logger.error(f"MCPC: Exception occurred: {exc_value}")
        if traceback:
            logger.error(f"MCPC: Traceback: {traceback}")


# if __name__ == "__main__":
#     # Test the MCPClient.
#     import asyncio
    
#     server_manager = MCPServerManager()
    
#     async def main():
#         async with MCPClient(server_manager) as client:
#             await client.connect_to_server("example")
            
#             # Test error handling by calling a non-existent tool.
#             await client.call_tool("example_tool", {"arg1": "value1"})
            
#     asyncio.run(main())