from __future__ import annotations

import re

from knowledge.knowledge_source import PersonaRecallResult, RouteDecision, RouteType, SearchMode


class QueryRouter:
    RELATIONAL_KEYWORDS = (
        "还记得我们",
        "记得我吗",
        "第一次聊天",
        "我们之间",
        "你对我",
        "你怎么看我",
        "你喜欢我吗",
        "你讨厌我吗",
        "我们的关系",
        "你信任我吗",
        "你会想我吗",
    )
    PERSONA_INTERNAL_KEYWORDS = (
        "故事",
        "经历",
        "设定",
        "背景",
        "过去",
        "原作",
        "剧情",
        "性格",
        "口头禅",
        "口癖",
        "说话方式",
        "语气",
        "称呼",
        "喜欢",
        "讨厌",
        "厌恶",
        "价值观",
        "世界观",
        "身份",
        "外貌",
        "外观",
        "发色",
        "关系",
        "喜好",
        "习惯",
        "经典台词",
        "自我介绍",
        "介绍一下你自己",
        "介绍你自己",
        "你是谁",
        "讲讲你",
        "讲讲你的",
        "你的看法",
        "你的想法",
    )
    REALITY_KEYWORDS = (
        "天气",
        "气温",
        "温度",
        "下雨",
        "晴天",
        "阴天",
        "新闻",
        "比赛",
        "结果",
        "排名",
        "官网",
        "资料",
        "信息",
        "今天",
        "明天",
        "昨天",
        "最新",
        "什么时候",
        "哪里",
        "现实",
        "真实",
        "weather",
        "forecast",
        "news",
        "ranking",
        "result",
        "latest",
        "who is",
        "what is",
    )
    MIXED_KEYWORDS = (
        "你怎么看",
        "你觉得",
        "如何评价",
        "评价一下",
        "好玩吗",
        "值不值得",
        "推荐吗",
        "怎么样",
        "看法",
        "感想",
    )

    def route(self, user_input: str, persona_recall: PersonaRecallResult, is_public: bool) -> RouteDecision:
        info_domain = self._classify_info_domain(user_input)
        coverage = float(persona_recall.coverage_score or 0.0)

        if info_domain == "RELATIONAL":
            return RouteDecision(type=RouteType.E5, web_search_mode=SearchMode.NONE, info_domain=info_domain)
        if info_domain == "REALITY_FACTUAL":
            return RouteDecision(type=RouteType.E4, web_search_mode=SearchMode.REALITY_SEARCH, info_domain=info_domain)
        if info_domain == "MIXED":
            mode = SearchMode.BOTH if is_public else SearchMode.REALITY_SEARCH
            return RouteDecision(type=RouteType.E3, web_search_mode=mode, info_domain=info_domain)
        if coverage >= 0.7:
            return RouteDecision(type=RouteType.E1, web_search_mode=SearchMode.NONE, info_domain="CHARACTER_INTERNAL")
        if is_public:
            return RouteDecision(
                type=RouteType.E2,
                web_search_mode=SearchMode.PERSONA_SEARCH,
                search_hint=persona_recall.activated_features,
                info_domain="CHARACTER_INTERNAL",
            )
        return RouteDecision(
            type=RouteType.E2B,
            web_search_mode=SearchMode.NONE,
            search_hint=persona_recall.activated_features,
            fallback="conservative",
            info_domain="CHARACTER_INTERNAL",
        )

    def _classify_info_domain(self, text: str) -> str:
        raw = (text or "").strip()
        lowered = raw.lower()
        if not raw:
            return "CHARACTER_INTERNAL"
        if self._contains(raw, lowered, self.RELATIONAL_KEYWORDS):
            return "RELATIONAL"
        if self._contains(raw, lowered, self.PERSONA_INTERNAL_KEYWORDS):
            return "CHARACTER_INTERNAL"
        if self._is_reality_factual(raw, lowered):
            return "REALITY_FACTUAL"
        if self._is_mixed(raw, lowered):
            return "MIXED"
        return "CHARACTER_INTERNAL"

    def _is_reality_factual(self, raw: str, lowered: str) -> bool:
        if self._contains(raw, lowered, self.REALITY_KEYWORDS):
            return True
        if re.search(r"\b(weather|forecast|news|ranking|result|score|player|team|game|movie|anime)\b", lowered):
            return True
        if raw.endswith(("？", "?")) and re.search(r"[A-Za-z]{2,}[A-Za-z0-9:_-]*", raw):
            return True
        return False

    def _is_mixed(self, raw: str, lowered: str) -> bool:
        if self._contains(raw, lowered, self.MIXED_KEYWORDS):
            return True
        if re.search(r"[A-Za-z]{2,}[A-Za-z0-9:_-]*", raw) and any(
            phrase in raw for phrase in ("你怎么看", "你觉得", "如何评价", "评价一下", "推荐吗")
        ):
            return True
        return False

    def _contains(self, raw: str, lowered: str, keywords: tuple[str, ...]) -> bool:
        return any(keyword in raw or keyword in lowered for keyword in keywords)
