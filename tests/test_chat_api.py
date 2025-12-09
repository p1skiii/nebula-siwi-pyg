import json

from siwi.api.app import Flask
from siwi.api.deps import build_dependencies
from siwi.api.errors import register_error_handlers
from siwi.api.routes import register_routes
from siwi.api.settings import get_settings


def test_chat_endpoint_returns_sources():
    # Recreate app with mock LLM allowed
    base_settings = get_settings()
    settings = base_settings.__class__(
        debug=True,
        host=base_settings.host,
        port=base_settings.port,
        rag_top_k=base_settings.rag_top_k,
        rag_data_dir=base_settings.rag_data_dir,
        embed_backend=base_settings.embed_backend,
        llm_provider="mock",
        llm_api_key="",
        llm_allow_mock=True,
        enable_graph=False,
        agent_experimental=False,
    )
    deps = build_dependencies(settings)

    app = Flask(__name__)
    app.config["settings"] = settings
    app.config["deps"] = deps
    register_routes(app)
    register_error_handlers(app)
    client = app.test_client()

    resp = client.post(
        "/api/chat",
        data=json.dumps({"message": "What is this project?"}),
        content_type="application/json",
    )

    assert resp.status_code == 200
    data = resp.get_json()
    assert "answer" in data
    assert isinstance(data.get("sources"), list)
    assert len(data["sources"]) >= 0
