from __future__ import annotations

from tools.intent_extractor import IntentExtractionResult
from tools.tool_router import ToolExecutionReport, ToolRouter


class ToolRuntime:
    """薄执行层：只执行结构化工具意图，不再重复做硬编码判断。"""

    def __init__(self, registry):
        self.registry = registry
        self.router = ToolRouter(registry)

    def _ensure_runtime_fields(self) -> None:
        if not hasattr(self, "registry") or self.registry is None:
            from tools import DEFAULT_TOOL_REGISTRY

            self.registry = DEFAULT_TOOL_REGISTRY
        if not hasattr(self, "router") or self.router is None:
            self.router = ToolRouter(self.registry)

    def execute(
        self,
        intent_result: IntentExtractionResult,
        persona_name: str = "",
        **_: object,
    ) -> ToolExecutionReport:
        self._ensure_runtime_fields()
        return self.router.execute_intent(intent_result, persona_name=persona_name)
