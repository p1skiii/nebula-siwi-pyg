from typing import Dict, List

from siwi.agent.sdk import AgentRequest, AgentResult, AgentStep
from siwi.agent.tools import GraphTool, TextRagTool


GRAPH_KEYWORDS = {"graph", "subgraph", "neighbor", "path"}


def detect_intent(message: str) -> str:
    lowered = message.lower()
    if any(keyword in lowered for keyword in GRAPH_KEYWORDS):
        return "graph"
    return "text"


def run_agent(request: AgentRequest, deps: Dict) -> AgentResult:
    steps: List[AgentStep] = []

    intent = detect_intent(request.message)
    steps.append(AgentStep(name="classify_intent", input={"message": request.message}, output={"intent": intent}))

    pipeline = deps["pipeline"]
    settings = deps.get("settings")
    graph_enabled = bool(getattr(settings, "enable_graph", False))

    text_tool = TextRagTool(pipeline=pipeline)
    graph_tool = GraphTool(graph_enabled=graph_enabled)

    if intent == "graph" and graph_enabled:
        answer, sources, tool_meta = graph_tool.run(request)
        steps.append(
            AgentStep(
                name="call_graph_rag",
                input={"graph_enabled": graph_enabled},
                output={"answer_preview": answer[:160], "sources_count": len(sources), "meta": tool_meta},
            )
        )
        meta = {
            "mode": "graph_rag",
            "intent": intent,
            "graph_enabled": graph_enabled,
            "llm_provider": pipeline.llm_client.__class__.__name__,
            "agent_version": "lite-1",
        }
        meta.update(tool_meta)
        return AgentResult(answer=answer, sources=sources, trace=steps, meta=meta)

    # Fallback to text RAG
    answer, sources, tool_meta = text_tool.run(request)
    steps.append(
        AgentStep(
            name="call_text_rag",
            input={"top_k": pipeline.top_k},
            output={"answer_preview": answer[:160], "sources_count": len(sources), "meta": tool_meta},
        )
    )
    meta = {
        "mode": "text_rag",
        "intent": intent,
        "graph_enabled": graph_enabled,
        "llm_provider": pipeline.llm_client.__class__.__name__,
        "agent_version": "lite-1",
    }
    meta.update(tool_meta)
    return AgentResult(answer=answer, sources=sources, trace=steps, meta=meta)
