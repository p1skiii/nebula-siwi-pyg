from flask import Flask

from siwi.api.deps import build_dependencies
from siwi.api.errors import register_error_handlers
from siwi.api.routes import register_routes
from siwi.api.settings import get_settings


def create_app() -> Flask:
    app = Flask(__name__)

    settings = get_settings()
    deps = build_dependencies(settings)

    app.config["settings"] = settings
    app.config["deps"] = deps

    register_routes(app)
    register_error_handlers(app)

    return app


app = create_app()


if __name__ == "__main__":
    settings = app.config["settings"]
    app.run(host=settings.host, port=settings.port, debug=settings.debug)
