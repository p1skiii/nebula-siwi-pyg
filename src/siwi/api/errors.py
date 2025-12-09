from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from flask import jsonify


@dataclass
class APIError(Exception):
    status_code: int
    message: str
    payload: Optional[Dict[str, Any]] = None

    def to_response(self) -> Tuple[Any, int]:
        body = {"error": self.message}
        if self.payload:
            body.update(self.payload)
        return jsonify(body), self.status_code


def register_error_handlers(app) -> None:
    @app.errorhandler(APIError)
    def handle_api_error(error: APIError):
        return error.to_response()

    @app.errorhandler(Exception)
    def handle_unexpected_error(error: Exception):
        body = {"error": "Internal server error"}
        return jsonify(body), 500
