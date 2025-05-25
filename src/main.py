from typing import Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP()

@mcp.tool()
async def query_etcd(cmd: str, params: Any) -> str:
    print(f"query_etcd: {cmd}, params: {params}")
    return "Hello, world!"

if __name__ == "__main__":
    mcp.run(transport='stdio')