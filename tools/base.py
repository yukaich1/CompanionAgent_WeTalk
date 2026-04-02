from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    tags: List[str] = field(default_factory=list)


class AgentTool:
    spec: ToolSpec

    def run(self, **kwargs):
        raise NotImplementedError
