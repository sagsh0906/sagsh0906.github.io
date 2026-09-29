"""MatBrain: graph-based state machine coupling Mat-T1 (execution) and Mat-R1 (reasoning).

    execute (Mat-T1) --> reason (Mat-R1) --FINISH--> finalize --> END
         ^                    |
         +----CONTINUE--------+   (iterative rollback with Mat-R1's next instruction)
                              +--iteration >= max_iterations--> force_answer --> finalize

Mat-R1 never touches tools: it ingests the cumulative execution history from
the shared state, assesses physical plausibility and outputs an
interpretation plus a decision intent. ``max_iterations`` (default 6) bounds
the recursion; on reaching it Mat-R1 must give a best-effort answer.
"""

from __future__ import annotations

import operator
import re
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from matbrain.agent.executor import ToolExecutor
from matbrain.agent.llm import ChatBackend
from matbrain.agent.parsing import ANSWER_RE, HANDLE_RE
from matbrain.mcp.registry import ToolRegistry
from matbrain.prompts import MAT_R1_SYSTEM, MATBRAIN_FORCE_ANSWER, MATBRAIN_INITIAL_ANALYSIS, MATBRAIN_REASONER_INSTRUCTIONS


def _tag(text: str, tag: str) -> str | None:
    m = re.findall(rf"<{tag}>(.*?)</{tag}>", text, re.S)
    return m[-1].strip() if m else None


@dataclass
class Decision:
    decision: str
    interpretation: str
    next_instruction: str | None
    answer: str | None
    raw: str


def parse_decision(text: str) -> Decision:
    body = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    decision = (_tag(body, "decision") or "").upper()
    interpretation = _tag(body, "interpretation") or ""
    nxt = _tag(body, "next_instruction")
    m = ANSWER_RE.findall(body)
    answer = m[-1].strip() if m else None
    if decision not in ("CONTINUE", "FINISH"):
        decision = "FINISH" if answer is not None or not nxt else "CONTINUE"
    if decision == "CONTINUE" and not nxt:
        nxt = interpretation or "Gather the missing evidence required to answer the task."
    if decision == "FINISH" and answer is None:
        answer = body.strip()
    return Decision(decision, interpretation, nxt, answer, text)


class MatBrainState(TypedDict, total=False):
    query: str
    instruction: str
    iteration: int
    executions: Annotated[list[dict[str, Any]], operator.add]
    analyses: Annotated[list[dict[str, Any]], operator.add]
    decision: str
    final_answer: str
    forced: bool


@dataclass
class MatBrainResult:
    query: str
    answer: str
    iterations: int
    forced: bool
    executions: list[dict[str, Any]] = field(default_factory=list)
    analyses: list[dict[str, Any]] = field(default_factory=list)


class MatBrain:
    def __init__(
        self,
        executor_backend: ChatBackend,
        reasoner_backend: ChatBackend,
        registry: ToolRegistry,
        max_iterations: int = 6,
        max_tool_calls: int = 12,
        start_with_analysis: bool = False,
        dispatcher: Any = None,
        resolve_handles: bool = True,
        reasoner_temperature: float = 0.0,
    ):
        self.executor = ToolExecutor(executor_backend, registry, max_tool_calls=max_tool_calls, dispatcher=dispatcher)
        self.reasoner = reasoner_backend
        self.dispatcher = dispatcher or registry
        self.max_iterations = max_iterations
        self.start_with_analysis = start_with_analysis
        self.resolve_handles = resolve_handles
        self.reasoner_temperature = reasoner_temperature
        self.graph = self._build()

    # ------------------------------------------------------------------ nodes
    def _history(self, state: MatBrainState) -> str:
        parts = []
        for i, ex in enumerate(state.get("executions", []), 1):
            parts.append(f"### Iteration {i}\n{ex['summary']}")
        for i, an in enumerate(state.get("analyses", []), 1):
            if an.get("interpretation"):
                parts.append(f"### Mat-R1 analysis {i}\n{an['interpretation']}")
        return "\n\n".join(parts) or "(no tool execution yet)"

    async def _ask_reasoner(self, state: MatBrainState, extra: str) -> Decision:
        messages = [
            {"role": "system", "content": f"{MAT_R1_SYSTEM}\n\n{MATBRAIN_REASONER_INSTRUCTIONS}"},
            {"role": "user", "content": f"Task:\n{state['query']}\n\nExecution history:\n{self._history(state)}\n\n{extra}".strip()},
        ]
        reply = await self.reasoner.chat(messages, temperature=self.reasoner_temperature)
        text = reply.content if not reply.reasoning else f"<think>{reply.reasoning}</think>{reply.content}"
        return parse_decision(text)

    async def analyze(self, state: MatBrainState) -> dict:
        d = await self._ask_reasoner(state, MATBRAIN_INITIAL_ANALYSIS)
        out = {"analyses": [d.__dict__], "decision": d.decision}
        if d.decision == "FINISH":
            out["final_answer"] = d.answer
        else:
            out["instruction"] = d.next_instruction
        return out

    async def execute(self, state: MatBrainState) -> dict:
        iteration = state.get("iteration", 0) + 1
        context = None
        if state.get("executions"):
            context = f"Original task: {state['query']}\n\n{self._history(state)}"
        trace = await self.executor.run(state.get("instruction") or state["query"], context=context)
        record = {"iteration": iteration, "summary": trace.summary(), "trace": trace.to_dict()}
        return {"executions": [record], "iteration": iteration}

    async def reason(self, state: MatBrainState) -> dict:
        d = await self._ask_reasoner(state, "")
        out: dict[str, Any] = {"analyses": [d.__dict__], "decision": d.decision}
        if d.decision == "FINISH":
            out["final_answer"] = d.answer
        else:
            out["instruction"] = d.next_instruction
        return out

    async def force_answer(self, state: MatBrainState) -> dict:
        d = await self._ask_reasoner(state, MATBRAIN_FORCE_ANSWER)
        return {"analyses": [d.__dict__], "decision": "FINISH", "final_answer": d.answer or d.interpretation, "forced": True}

    async def finalize(self, state: MatBrainState) -> dict:
        answer = state.get("final_answer") or ""
        if self.resolve_handles:
            answer = await self._resolve(answer)
        return {"final_answer": answer}

    async def _resolve(self, answer: str) -> str:
        """Replace an answer that is just an artifact handle by the full CIF."""
        handles = HANDLE_RE.findall(answer)
        if len(handles) != 1 or len(answer.strip()) > len(handles[0]) + 200:
            return answer
        res = await self.dispatcher.acall("get_structure_cif", {"handle": handles[0]}, strict_handles=True)
        if res.ok and isinstance(res.data, dict) and res.data.get("cif"):
            return answer.replace(handles[0], "\n" + res.data["cif"])
        return answer

    # ---------------------------------------------------------------- routing
    def _after_reason(self, state: MatBrainState) -> str:
        if state.get("decision") == "FINISH":
            return "finalize"
        if state.get("iteration", 0) >= self.max_iterations:
            return "force_answer"
        return "execute"

    def _after_analyze(self, state: MatBrainState) -> str:
        return "finalize" if state.get("decision") == "FINISH" else "execute"

    def _build(self):
        g = StateGraph(MatBrainState)
        g.add_node("execute", self.execute)
        g.add_node("reason", self.reason)
        g.add_node("force_answer", self.force_answer)
        g.add_node("finalize", self.finalize)
        if self.start_with_analysis:
            g.add_node("analyze", self.analyze)
            g.add_edge(START, "analyze")
            g.add_conditional_edges("analyze", self._after_analyze, {"execute": "execute", "finalize": "finalize"})
        else:
            g.add_edge(START, "execute")
        g.add_edge("execute", "reason")
        g.add_conditional_edges("reason", self._after_reason, {"execute": "execute", "force_answer": "force_answer", "finalize": "finalize"})
        g.add_edge("force_answer", "finalize")
        g.add_edge("finalize", END)
        return g.compile()

    async def arun(self, query: str) -> MatBrainResult:
        state = await self.graph.ainvoke(
            {"query": query, "instruction": query, "iteration": 0, "executions": [], "analyses": [], "forced": False},
            config={"recursion_limit": 4 * self.max_iterations + 10},
        )
        return MatBrainResult(
            query=query,
            answer=state.get("final_answer", ""),
            iterations=state.get("iteration", 0),
            forced=bool(state.get("forced")),
            executions=state.get("executions", []),
            analyses=state.get("analyses", []),
        )
