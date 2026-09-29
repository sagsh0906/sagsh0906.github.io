"""In-process Mat-MCP tools for verl multi-turn rollouts.

Two ways to expose Mat-MCP to verl (v0.6.1) are provided:

* remote (paper setup): run ``python -m matbrain.mcp.server`` (Docker/K8s)
  and use ``verl.tools.mcp_base_tool.MCPBaseTool`` with
  ``training/rl/tool_config/mat_mcp_remote.yaml``;
* local: this class, which calls the same ``ToolRegistry`` inside the
  agent-loop worker, avoiding network hops for CPU-only tools. Generate the
  config with ``python training/rl/make_tool_config.py --mode local``.
"""

from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from matbrain.mcp import load_registry


class MatMCPNativeTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.registry = load_registry()
        if self.name not in self.registry:
            raise ValueError(f"{self.name} is not a registered Mat-MCP tool")
        self.max_chars = int(config.get("max_observation_chars", 4000))
        self.timeout = float(config.get("timeout", 600))
        self._instances: dict[str, list[bool]] = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        instance_id = instance_id or str(uuid4())
        self._instances[instance_id] = []
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        result = await self.registry.acall(self.name, parameters, strict_handles=True, timeout=self.timeout)
        self._instances.setdefault(instance_id, []).append(result.ok)
        # Step reward 0.0: the composite process reward is computed on the full trajectory.
        return ToolResponse(text=result.to_observation(self.max_chars)), 0.0, {"tool_ok": float(result.ok), "tool_seconds": result.elapsed}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        calls = self._instances.get(instance_id, [])
        return sum(calls) / len(calls) if calls else 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)
