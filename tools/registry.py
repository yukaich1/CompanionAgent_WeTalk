from tools.weather import WeatherTool
from tools.web_search import WebSearchTool


class ToolRegistry:
    def __init__(self):
        self._tools = {}

    def register(self, tool):
        self._tools[tool.spec.name] = tool

    def get(self, name):
        return self._tools.get(name)

    def run(self, name, **kwargs):
        tool = self.get(name)
        if tool is None:
            raise KeyError(f"Unknown tool: {name}")
        return tool.run(**kwargs)

    def list_specs(self):
        return [tool.spec for tool in self._tools.values()]


DEFAULT_TOOL_REGISTRY = ToolRegistry()
DEFAULT_TOOL_REGISTRY.register(WebSearchTool())
DEFAULT_TOOL_REGISTRY.register(WeatherTool())
