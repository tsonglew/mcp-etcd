import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters, Tool, stdio_client
from openai import OpenAI

load_dotenv()


class MCPClient:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.base_url = os.getenv("BASE_URL", "http://localhost:8000")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.model = os.getenv("MODEL", "deepseek-chat")
        self.session: ClientSession | None = None

        self.client = OpenAI(api_key=self.openai_api_key, base_url=self.base_url)

        self.tools: list[Tool] = []

    async def connect_to_server(self, server_script_path: str):
        is_python_script = server_script_path.endswith(".py")
        is_javascript_script = server_script_path.endswith(".js")
        if not is_python_script and not is_javascript_script:
            raise ValueError("Unsupported script type: {server_script_path}")

        command = "python" if is_python_script else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None,
        )

        # start mcp server
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.writer = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.writer)
        )
        await self.session.initialize()


        self.tools = (await self.session.list_tools()).tools
        print("mcp server tools: ", self.tools)

    async def process_query(self, query: str) -> str | None:
        available_tools = [
            {"type": "function", "function": {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }} for tool in (self.tools or [])
        ]
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": query}
        ]
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.client.chat.completions.create(
                model=self.model,
                tools=available_tools,
                messages=messages,
            )
        )
        content = response.choices[0]
        if content.finish_reason == "tool_calls":
            tool_call = content.message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            result = await self.session.call_tool(
                tool_name,
                tool_args,
            )
            print(f"Tool call: {tool_name} with args: {tool_args}, result: {result}")

            messages.append(content.message.model_dump())
            messages.append({
                "role": "tool",
                "content": result.content[0].text,
                "tool_call_id": tool_call.id,
            })

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                )
            )
            return response.choices[0].message.content

        return content.message.content

    async def chat_loop(self):
        while True:
            user_input = input("Enter a command: ")
            response = await self.process_query(user_input)
            print(response)

    async def cleanup(self):
        await self.exit_stack.aclose()

async def main():
    server_script_path = sys.argv[1]
    client = MCPClient()
    await client.connect_to_server(server_script_path)
    try:
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())