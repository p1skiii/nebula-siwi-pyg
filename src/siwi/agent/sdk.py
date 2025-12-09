from dataclasses import dataclass, field
from typing import Dict, List, Protocol, Tuple


@dataclass
class AgentRequest:
    session_id: str
    message: str


@dataclass
class AgentStep:
    name: str
    input: Dict
    output: Dict


@dataclass
class AgentResult:
    answer: str
    sources: List[Dict]
    trace: List[AgentStep] = field(default_factory=list)
    meta: Dict = field(default_factory=dict)


class Tool(Protocol):
    name: str

    def run(self, request: AgentRequest) -> Tuple[str, List[Dict], Dict]:
        ...
