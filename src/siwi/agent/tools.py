from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from siwi.agent.sdk import AgentRequest, Tool
from siwi.rag.pipeline import RagPipeline


class GraphAdapter:
    """Thin adapter to optional graph backend."""

    def __init__(self):
        try:
            from siwi.graph_backend.subgraph_sampler import SubgraphSampler  # type: ignore
            from siwi.graph_backend.feature_store import get_nebula_connection_pool  # type: ignore
        except Exception as exc:  # pragma: no cover - optional
            raise RuntimeError(f"Graph backend unavailable: {exc}") from exc

        self.pool = get_nebula_connection_pool()
        self.sampler = SubgraphSampler(self.pool)

    def fetch_subgraph_summary(self, vid: str) -> Tuple[str, List[Dict], Dict]:
        subgraph = self.sampler.sample_subgraph(center_vid=vid, n_hops=1, max_nodes=128)
        nodes = subgraph.get("idx_to_vid", [])
        edges = subgraph.get("edge_index")
        num_edges = int(edges.shape[1]) if edges is not None and hasattr(edges, "shape") else 0
        answer = f"Fetched 1-hop subgraph for {vid} with {len(nodes)} nodes and {num_edges} edges."
        sources = [
            {
                "doc_id": vid,
                "title": "graph_subgraph",
                "snippet": f"nodes={len(nodes)}, edges={num_edges}",
                "score": 1.0,
            }
        ]
        meta = {"graph_vid": vid}
        return answer, sources, meta


@dataclass
class TextRagTool(Tool):
    name: str = "text_rag"
    pipeline: RagPipeline = None  # type: ignore

    def run(self, request: AgentRequest) -> Tuple[str, List[Dict], Dict]:
        answer, sources, meta = self.pipeline.answer(query=request.message)
        tool_meta = {"tool": self.name, **meta}
        return answer, sources, tool_meta


@dataclass
class GraphTool(Tool):
    name: str = "graph_rag"
    graph_enabled: bool = False
    _adapter: Optional[GraphAdapter] = None
    _init_error: Optional[str] = None

    def __post_init__(self):
        if self.graph_enabled:
            try:
                self._adapter = GraphAdapter()
            except Exception as exc:
                self._init_error = str(exc)

    def run(self, request: AgentRequest) -> Tuple[str, List[Dict], Dict]:
        meta = {
            "tool": self.name,
            "graph_enabled": self.graph_enabled,
            "graph_error": self._init_error,
        }

        if not self.graph_enabled:
            return "Graph backend disabled.", [], meta

        if self._init_error:
            return f"Graph backend unavailable: {self._init_error}", [], meta

        vid = self._extract_vid(request.message)
        if not vid:
            meta["graph_error"] = "missing_vid"
            return "Provide a node id (e.g., player142) to fetch a subgraph.", [], meta

        try:
            answer, sources, adapter_meta = self._adapter.fetch_subgraph_summary(vid)
            meta.update(adapter_meta)
            return answer, sources, meta
        except Exception as exc:  # pragma: no cover - depends on Nebula
            meta["graph_error"] = str(exc)
            return f"Graph backend call failed: {exc}", [], meta

    @staticmethod
    def _extract_vid(text: str) -> Optional[str]:
        tokens = text.replace("?", " ").replace(",", " ").split()
        for token in tokens:
            if token.startswith(("player", "team")):
                return token
        return None
