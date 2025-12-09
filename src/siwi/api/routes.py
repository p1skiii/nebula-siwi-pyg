from flask import Blueprint, current_app, jsonify, request

from siwi.agent.runner import run_agent
from siwi.agent.sdk import AgentRequest
from siwi.api.errors import APIError
from siwi.agent.tools import TextRagTool

bp = Blueprint("api", __name__)


@bp.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"status": "ok"})


@bp.route("/api/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    message = payload.get("message")
    session_id = payload.get("session_id", "")

    if not message or not isinstance(message, str):
        raise APIError(status_code=400, message="message is required")

    deps = current_app.config["deps"]
    settings = deps["settings"]

    if deps.get("llm_client") is None:
        hint = (
            "Configure a real LLM provider (e.g., LLM_PROVIDER=gemini with GEMINI_API_KEY) "
            "or enable demo mode with LLM_PROVIDER=mock and LLM_ALLOW_MOCK=1."
        )
        return (
            jsonify({"error": "LLM provider not configured", "hint": hint}),
            503,
        )

    # LLM-only path (default): RAG/Graph off unless explicitly enabled
    if not getattr(settings, "enable_rag", False):
        try:
            answer = deps["llm_client"].generate(query=message, context=[])
            meta = {
                "agent_enabled": False,
                "mode": "llm_only",
                "llm_provider": deps["llm_client"].__class__.__name__,
                "rag_enabled": False,
                "graph_enabled": settings.enable_graph,
            }
            return jsonify({"answer": answer, "sources": [], "meta": meta, "trace": []})
        except Exception as exc:
            return jsonify({"error": "LLM provider error", "detail": str(exc)}), 502

    if not settings.agent_experimental:
        pipeline = deps["pipeline"]
        text_tool = TextRagTool(pipeline=pipeline)
        try:
            answer, sources, tool_meta = text_tool.run(AgentRequest(session_id=session_id, message=message))
            meta = {
                "agent_enabled": False,
                "mode": "text_rag",
                "llm_provider": pipeline.llm_client.__class__.__name__,
                "rag_enabled": True,
                "graph_enabled": settings.enable_graph,
            }
            meta.update(tool_meta)
            return jsonify({"answer": answer, "sources": sources, "meta": meta, "trace": []})
        except Exception as exc:
            return jsonify({"error": "LLM provider error", "detail": str(exc)}), 502

    agent_request = AgentRequest(session_id=session_id, message=message)
    try:
        result = run_agent(agent_request, deps)
        return jsonify(
            {
                "answer": result.answer,
                "sources": result.sources,
                "meta": result.meta,
                "trace": [step.__dict__ for step in result.trace],
            }
        )
    except Exception as exc:
        return jsonify({"error": "Agent execution error", "detail": str(exc)}), 502


def register_routes(app) -> None:
    app.register_blueprint(bp)
