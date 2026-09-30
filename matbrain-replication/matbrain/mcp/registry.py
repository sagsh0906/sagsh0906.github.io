"""Tool registry shared by the Mat-MCP server, the MatBrain runtime validation
layer and the Mat-T1 syntax reward.

A single source of truth matters here: the syntax reward (paper Eq. 5) checks
(1) registry membership in the Mat-MCP namespace and (2) parameter integrity
against the Pydantic schema, and the runtime layer of the execution node
applies exactly the same checks before dispatching a call.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
import traceback
from dataclasses import dataclass, field
from typing import Annotated, Any, Callable

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError, ValidationInfo

from matbrain.chem import StructureParseError, parse_structure
from matbrain.mcp.artifacts import ArtifactStore, get_store, is_handle

# Tool categories (Supplementary Tables 1-2 of the paper group tools this way).
SEARCH = "search"
RETRIEVAL = "retrieval"  # direct target-database lookup (MP / OQMD)
GENERATION = "generation"
VALIDATION = "validation"
ANALYSIS = "analysis"
COMPUTATION = "computation"
CATEGORIES = (SEARCH, RETRIEVAL, GENERATION, VALIDATION, ANALYSIS, COMPUTATION)


# --------------------------------------------------------------------------- #
# Reusable argument types
# --------------------------------------------------------------------------- #
def _check_cif(value: str, info: ValidationInfo) -> str:
    value = value.strip()
    ctx = info.context or {}
    if is_handle(value):
        if ctx.get("strict_handles"):
            store: ArtifactStore = ctx.get("store") or get_store()
            if not store.exists(value):
                raise ValueError(f"artifact handle {value} does not exist")
        return value
    try:
        parse_structure(value)
    except StructureParseError as exc:
        raise ValueError(f"not a parseable CIF or artifact handle: {exc}") from exc
    return value


def _check_formula(value: str) -> str:
    from pymatgen.core import Composition

    value = value.strip()
    try:
        comp = Composition(value)
    except Exception as exc:
        raise ValueError(f"invalid chemical formula {value!r}: {exc}") from exc
    if comp.num_atoms <= 0:
        raise ValueError("formula has no atoms")
    return value


def _check_chemsys(value: str) -> str:
    from pymatgen.core import Element

    parts = [p.strip() for p in value.replace(",", "-").split("-") if p.strip()]
    if not parts:
        raise ValueError("empty chemical system")
    for p in parts:
        if not Element.is_valid_symbol(p):
            raise ValueError(f"invalid element symbol {p!r}")
    return "-".join(parts)


CifInput = Annotated[
    str,
    AfterValidator(_check_cif),
    Field(description="CIF text or an artifact handle of the form cif://<12 hex chars>"),
]
Formula = Annotated[str, AfterValidator(_check_formula), Field(description="Chemical formula, e.g. Li2ZrCl6")]
ChemSys = Annotated[str, AfterValidator(_check_chemsys), Field(description="Chemical system, e.g. Co-V-S")]


class ToolArgs(BaseModel):
    """Base class for tool argument schemas: unknown arguments are rejected."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
@dataclass
class ToolResult:
    name: str
    ok: bool
    data: Any = None
    error: str | None = None
    stdout: str = ""
    stderr: str = ""
    elapsed: float = 0.0

    def to_observation(self, max_chars: int | None = 4000) -> str:
        """Standardised observation text fed back into the model context."""
        if self.ok:
            payload = {"tool": self.name, "status": "success", "result": self.data}
        else:
            payload = {"tool": self.name, "status": "error", "error": self.error}
        if self.stderr:
            payload["stderr"] = self.stderr[-1000:]
        text = json.dumps(payload, ensure_ascii=False, default=str)
        if max_chars and len(text) > max_chars:
            half = max_chars // 2
            text = text[:half] + " ...[truncated]... " + text[-half:]
        return text


@dataclass
class CallValidation:
    ok: bool
    name: str
    args: BaseModel | None = None
    error: str | None = None


@dataclass
class ToolSpec:
    name: str
    description: str
    args_model: type[ToolArgs]
    func: Callable[..., Any]
    category: str
    target_db_lookup: bool = False
    optional_deps: tuple[str, ...] = ()
    timeout: float = 600.0
    extra: dict[str, Any] = field(default_factory=dict)

    def json_schema(self) -> dict[str, Any]:
        return self.args_model.model_json_schema()

    def openai_schema(self, flat: bool = False) -> dict[str, Any]:
        params = self.json_schema()
        if flat:
            params = flatten_schema(params)
        else:
            params.pop("title", None)
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": params}}


def flatten_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Reduce a Pydantic JSON schema to {type, description, enum} per property.

    verl's OpenAIFunctionPropertySchema only accepts these keys, so this is the
    representation rendered into the chat template during RL rollouts.
    """
    props = {}
    for key, prop in schema.get("properties", {}).items():
        typ = prop.get("type")
        if typ is None and "anyOf" in prop:
            types = [p.get("type") for p in prop["anyOf"] if p.get("type") and p.get("type") != "null"]
            typ = types[0] if types else "string"
        if typ is None and "$ref" in prop:
            typ = "object"
        entry: dict[str, Any] = {"type": typ or "string"}
        desc = prop.get("description") or next((p.get("description") for p in prop.get("anyOf", []) if p.get("description")), None)
        if "default" in prop and prop["default"] is not None:
            desc = f"{desc or ''} (default: {prop['default']})".strip()
        if desc:
            entry["description"] = desc
        if "enum" in prop:
            entry["enum"] = [str(e) for e in prop["enum"]]
        props[key] = entry
    return {"type": "object", "properties": props, "required": list(schema.get("required", []))}


class ToolRegistry:
    def __init__(self, specs: dict[str, ToolSpec] | None = None):
        self._specs: dict[str, ToolSpec] = dict(specs or {})

    # ----------------------------------------------------------------- build
    def tool(
        self,
        name: str | None = None,
        *,
        category: str,
        target_db_lookup: bool = False,
        optional_deps: tuple[str, ...] = (),
        timeout: float = 600.0,
    ):
        """Decorator. The function takes a single argument: its Pydantic args model."""
        if category not in CATEGORIES:
            raise ValueError(f"unknown category {category}")

        def deco(func: Callable[..., Any]):
            params = list(inspect.signature(func).parameters.values())
            if len(params) != 1:
                raise TypeError(f"tool {func.__name__} must take exactly one args-model parameter")
            model = func.__annotations__.get(params[0].name)
            if isinstance(model, str):
                model = func.__globals__[model]
            if not (isinstance(model, type) and issubclass(model, BaseModel)):
                raise TypeError(f"tool {func.__name__}: parameter must be annotated with a pydantic model")
            tool_name = name or func.__name__
            doc = inspect.cleandoc(func.__doc__ or tool_name)
            self._specs[tool_name] = ToolSpec(
                name=tool_name,
                description=doc,
                args_model=model,
                func=func,
                category=category,
                target_db_lookup=target_db_lookup,
                optional_deps=optional_deps,
                timeout=timeout,
            )
            return func

        return deco

    # ------------------------------------------------------------- queries
    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    def __iter__(self):
        return iter(self._specs.values())

    def names(self) -> list[str]:
        return sorted(self._specs)

    def get(self, name: str) -> ToolSpec:
        return self._specs[name]

    def subset(
        self,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        categories: list[str] | None = None,
        disable_target_db: bool = False,
    ) -> "ToolRegistry":
        """Shared tool pool used in the benchmark (target DB lookup disabled)."""
        out = {}
        for name, spec in self._specs.items():
            if include is not None and name not in include:
                continue
            if exclude and name in exclude:
                continue
            if categories is not None and spec.category not in categories:
                continue
            if disable_target_db and spec.target_db_lookup:
                continue
            out[name] = spec
        return ToolRegistry(out)

    def openai_tools(self, flat: bool = False) -> list[dict[str, Any]]:
        return [self._specs[n].openai_schema(flat=flat) for n in self.names()]

    # ---------------------------------------------------------- validation
    def validate(self, name: str, arguments: Any, *, strict_handles: bool = False, store: ArtifactStore | None = None) -> CallValidation:
        """Registry verification + parameter integrity (paper, syntax reward)."""
        if not isinstance(name, str) or name not in self._specs:
            return CallValidation(False, str(name), error=f"tool {name!r} is not registered in the Mat-MCP namespace")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as exc:
                return CallValidation(False, name, error=f"arguments are not valid JSON: {exc}")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return CallValidation(False, name, error="arguments must be a JSON object")
        spec = self._specs[name]
        try:
            args = spec.args_model.model_validate(arguments, context={"strict_handles": strict_handles, "store": store})
        except ValidationError as exc:
            msgs = "; ".join(f"{'.'.join(map(str, e['loc'])) or '<root>'}: {e['msg']}" for e in exc.errors())
            return CallValidation(False, name, error=f"invalid arguments for {name}: {msgs}")
        return CallValidation(True, name, args=args)

    # ----------------------------------------------------------- execution
    async def acall(self, name: str, arguments: Any, *, strict_handles: bool = True, timeout: float | None = None) -> ToolResult:
        t0 = time.perf_counter()
        check = self.validate(name, arguments, strict_handles=strict_handles)
        if not check.ok:
            return ToolResult(name=str(name), ok=False, error=check.error, elapsed=time.perf_counter() - t0)
        spec = self._specs[name]
        try:
            if inspect.iscoroutinefunction(spec.func):
                coro = spec.func(check.args)
            else:
                coro = asyncio.to_thread(spec.func, check.args)
            data = await asyncio.wait_for(coro, timeout=timeout or spec.timeout)
            stdout = stderr = ""
            if isinstance(data, ToolOutput):
                stdout, stderr, data = data.stdout, data.stderr, data.data
            return ToolResult(name=name, ok=True, data=data, stdout=stdout, stderr=stderr, elapsed=time.perf_counter() - t0)
        except asyncio.TimeoutError:
            return ToolResult(name=name, ok=False, error=f"tool timed out after {timeout or spec.timeout}s", elapsed=time.perf_counter() - t0)
        except ToolError as exc:
            return ToolResult(name=name, ok=False, error=str(exc), stderr=exc.stderr, elapsed=time.perf_counter() - t0)
        except Exception as exc:  # surface any backend failure as an observation
            tb = traceback.format_exc(limit=3)
            return ToolResult(name=name, ok=False, error=f"{type(exc).__name__}: {exc}", stderr=tb, elapsed=time.perf_counter() - t0)

    def call(self, name: str, arguments: Any, **kwargs) -> ToolResult:
        return asyncio.run(self.acall(name, arguments, **kwargs))


class ToolError(RuntimeError):
    """Expected, user-facing tool failure (e.g. unsupported non-stoichiometric formula)."""

    def __init__(self, message: str, stderr: str = ""):
        super().__init__(message)
        self.stderr = stderr


@dataclass
class ToolOutput:
    """Return this from a tool to attach captured stdout/stderr (e.g. CLI tools)."""

    data: Any
    stdout: str = ""
    stderr: str = ""


def require(module: str, extra: str = "tools"):
    """Import an optional backend or raise a ToolError explaining how to install it."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ToolError(
            f"backend '{module}' is not installed in the Mat-MCP container "
            f"(pip install 'matbrain-replication[{extra}]' or see deploy/Dockerfile): {exc}"
        ) from exc


# Global registry populated by matbrain.mcp.tools.*
REGISTRY = ToolRegistry()
