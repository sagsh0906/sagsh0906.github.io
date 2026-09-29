"""Mat-MCP: the materials-science tool ecosystem exposed over the Model Context Protocol."""

from matbrain.mcp.registry import REGISTRY, ToolRegistry, ToolResult


def load_registry() -> ToolRegistry:
    import matbrain.mcp.tools  # noqa: F401  (registers tools)

    return REGISTRY


__all__ = ["REGISTRY", "ToolRegistry", "ToolResult", "load_registry"]
