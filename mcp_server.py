"""Absolute-path entry point for desktop MCP clients; no working-directory dependency."""
from marketcanvas.server import mcp

if __name__ == "__main__":
    mcp.run(transport="stdio")
