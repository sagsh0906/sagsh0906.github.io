"""Mat-MCP server.

    python -m matbrain.mcp.server --transport streamable-http --port 8000
    python -m matbrain.mcp.server --disable-target-db          # benchmark tool pool

Every registered tool is exposed with a JSON schema derived from its Pydantic
argument model; calls are routed through ``ToolRegistry.acall`` so the MCP
path and the in-process path (RL rollouts, tests) behave identically.
"""

from __future__ import annotations

import argparse
import inspect
from typing import Annotated, Any

from matbrain.mcp import load_registry
from matbrain.mcp.registry import ToolRegistry, ToolSpec

try:  # mcp >= 2
    from mcp.server.mcpserver import MCPServer as _Server

    _MCP_V2 = True
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server  # type: ignore

    _MCP_V2 = False


def _signature_for(spec: ToolSpec) -> inspect.Signature:
    params = []
    for fname, finfo in spec.args_model.model_fields.items():
        default = inspect.Parameter.empty if finfo.is_required() else finfo.default
        params.append(
            inspect.Parameter(fname, inspect.Parameter.KEYWORD_ONLY, default=default, annotation=Annotated[finfo.annotation, finfo])
        )
    return inspect.Signature(params, return_annotation=str)


def _make_handler(registry: ToolRegistry, spec: ToolSpec, max_chars: int):
    async def handler(**kwargs: Any) -> str:
        # Drop unset optional fields so model defaults/validators apply exactly once.
        args = {k: v for k, v in kwargs.items() if v is not None or spec.args_model.model_fields[k].is_required()}
        result = await registry.acall(spec.name, args, strict_handles=True)
        return result.to_observation(max_chars=max_chars)

    handler.__name__ = spec.name
    handler.__doc__ = spec.description
    handler.__signature__ = _signature_for(spec)  # type: ignore[attr-defined]
    return handler


def build_server(registry: ToolRegistry | None = None, name: str = "Mat-MCP", max_chars: int = 8000, **server_kwargs):
    registry = registry or load_registry()
    server = _Server(name, **server_kwargs)
    for spec in registry:
        server.add_tool(_make_handler(registry, spec, max_chars), name=spec.name, description=spec.description)
    return server


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transport", default="streamable-http", choices=["stdio", "sse", "streamable-http"])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--disable-target-db", action="store_true", help="exclude MP/OQMD lookup tools (leakage-controlled benchmark)")
    ap.add_argument("--tools", nargs="*", help="explicit subset of tool names")
    ap.add_argument("--max-observation-chars", type=int, default=8000)
    args = ap.parse_args(argv)

    registry = load_registry().subset(include=args.tools, disable_target_db=args.disable_target_db)
    if _MCP_V2:
        server = build_server(registry, max_chars=args.max_observation_chars)
        if args.transport == "stdio":
            server.run("stdio")
        else:
            server.run(args.transport, host=args.host, port=args.port)
    else:
        server = build_server(registry, max_chars=args.max_observation_chars, host=args.host, port=args.port)
        server.run(args.transport)


if __name__ == "__main__":
    main()
