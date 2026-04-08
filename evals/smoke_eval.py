from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import AISystem


CASES = [
    {
        "name": "casual_greeting",
        "message": "晚上好。",
        "expect": {"intent": {"casual_chat", "emotional_interaction"}, "tool": {"", "none", "None"}},
        "shared_session": False,
    },
    {
        "name": "self_intro",
        "message": "请自我介绍一下。",
        "expect": {"intent": {"character_related"}, "tool": {"", "none", "None"}},
        "shared_session": False,
    },
    {
        "name": "story_request",
        "message": "请讲一个你的故事。",
        "expect": {"intent": {"character_related"}, "tool": {"", "none", "None"}},
        "shared_session": False,
    },
    {
        "name": "weather",
        "message": "请告诉我东京的天气如何？",
        "expect": {"intent": {"weather_query"}, "tool": {"weather"}},
        "shared_session": False,
    },
    {
        "name": "web_search",
        "message": "请介绍一下牛顿第三定律。",
        "expect": {"intent": {"web_search_query", "help_request"}, "tool": {"web_search"}},
        "shared_session": False,
    },
    {
        "name": "emotional",
        "message": "我今天有点难过。",
        "expect": {"intent": {"emotional_interaction", "casual_chat"}, "tool": {"", "none", "None"}},
        "shared_session": False,
    },
    {
        "name": "sports_emotion_no_search",
        "message": "我看比赛气坏了。",
        "expect": {"intent": {"emotional_interaction", "casual_chat"}, "tool": {"", "none", "None"}},
        "shared_session": False,
    },
    {
        "name": "real_person_search",
        "message": "你认识无畏契约职业选手Zmjjkk吗？",
        "expect": {"intent": {"web_search_query", "help_request"}, "tool": {"web_search"}},
        "shared_session": False,
    },
]

PREVIEW_SAMPLE = """
伊蕾娜是灰之魔女，来自和平国罗贝塔。她从小因为《妮可的冒险谭》而向往旅行，后来学习魔法，
并在年纪很轻的时候通过了魔法见习考试。她曾在星尘魔女芙兰门下修行一年，那段经历对她的成长影响很大。
她平时说话礼貌，偶尔带一点自得和克制的调侃，对金钱和旅途中便利很敏感。
""".strip()


def _judge_case(turn_trace: dict, expect: dict) -> dict:
    intent = str(turn_trace.get("intent", "") or "")
    tool = str(turn_trace.get("tool", "") or "")
    expected_intents = set(expect.get("intent", set()))
    expected_tools = set(expect.get("tool", set()))
    return {
        "intent_ok": not expected_intents or intent in expected_intents,
        "tool_ok": not expected_tools or tool in expected_tools,
    }


def _quality_checks(case: dict, result: dict) -> dict:
    reply = str(result.get("reply", "") or "")
    route = str(result.get("route", "") or "")
    checks = {
        "format_ok": all(token not in reply for token in ["锟", "�", "\ufffd"]),
        "not_empty": bool(reply.strip()),
    }
    if case["name"] == "casual_greeting":
        checks["casual_not_too_short"] = len(reply.strip()) >= 6
    if case["name"] == "self_intro":
        checks["self_intro_not_dramatic"] = all(
            token not in reply for token in ("失忆", "连自己是谁", "我是谁都", "说不太清楚自己是谁", "困惑")
        )
    if case["name"] == "story_request":
        checks["story_no_fabrication_hint"] = (
            "某个小镇" not in reply
            and "市集" not in reply
            and "没发生过" not in reply
            and "还没发生" not in reply
        )
    if case["name"] == "emotional":
        checks["emotional_not_detached"] = (
            len(reply.strip()) >= 12
            and "我不明白" not in reply
            and any(token in reply for token in ("陪", "听", "说说", "没关系", "可以", "愿意", "在这儿", "告诉我", "跟我说"))
        )
    if case["name"] == "sports_emotion_no_search":
        checks["no_accidental_search_reply"] = "查到" not in reply and "搜索" not in reply
        checks["no_fake_self_anecdote"] = all(token not in reply for token in ("我上次", "以前我也", "我之前也", "我记得我"))
    if case["name"] == "real_person_search":
        checks["no_unrelated_expansion"] = "英雄联盟" not in reply and "S赛" not in reply
    if route == "RouteType.E4":
        checks["external_first_person_clean"] = (
            "我不明白" not in reply
            and "括号" not in reply
            and "我记得" not in reply
            and "印象里" not in reply
        )
    return checks


def _run_preview_roundtrip() -> dict:
    ai = AISystem()
    preview = ai.persona_system.preview_from_sources(
        persona_name="伊蕾娜",
        work_title="魔女之旅",
        local_text=PREVIEW_SAMPLE,
        local_label="smoke_sample",
        enable_web_search=False,
    )
    preview_summary = ai.persona_system._summary_to_dict(preview.summary)
    result = ai.persona_system.confirm_preview(
        preview.preview_id,
        selected_keywords=list((preview_summary.get("display_keywords", []) or []))[:8],
    )
    summary = ai.persona_system._summary_to_dict(result["preview"]["summary"])
    keywords = list(summary.get("display_keywords", []) or [])
    experiences = list((((summary.get("base_template") or {}).get("00_BACKGROUND") or {}).get("key_experiences", [])) or [])
    experience_quality = bool(experiences) and all(len(item.strip()) >= 8 for item in experiences)
    experience_quality = experience_quality and all(
        not item.startswith(("并", "但", "而", "后来", "于是", "然后", "不过", "只是", "我", "这", "那"))
        for item in experiences
    )
    experience_quality = experience_quality and all(
        not (
            ("来自" in item or "灰之魔女" in item or "伊蕾娜是" in item)
            and all(token not in item for token in ("因为", "通过", "修行", "拜师", "约定", "试炼", "开始", "立志", "察觉"))
        )
        for item in experiences
    )
    return {
        "case": "preview_roundtrip",
        "preview_id": preview.preview_id,
        "committed_count": int(result.get("committed_count", 0) or 0),
        "keywords": keywords,
        "key_experiences": experiences,
        "quality_checks": {
            "keywords_present": bool(keywords),
            "key_experiences_present": bool(experiences),
            "key_experiences_quality": experience_quality,
        },
        "ok": bool(result.get("committed_count", 0) or 0) >= 1 and bool(keywords) and experience_quality,
    }


def run_smoke_eval(output_path: str | None = None) -> list[dict]:
    results: list[dict] = []
    shared_ai = AISystem()
    for case in CASES:
        ai = shared_ai if case.get("shared_session") else AISystem()
        try:
            reply = ai.send_message(case["message"])
            trace = dict(getattr(ai, "last_debug_info", {}) or {})
            turn_trace = dict(trace.get("turnTrace", {}) or {})
            verdict = _judge_case(turn_trace, case.get("expect", {}))
            results.append(
                {
                    "case": case["name"],
                    "message": case["message"],
                    "reply": str(reply or ""),
                    "route": str(turn_trace.get("route", "") or ""),
                    "intent": str(turn_trace.get("intent", "") or ""),
                    "tool": str(turn_trace.get("tool", "") or ""),
                    "coverage": turn_trace.get("coverage", 0.0),
                    "shared_session": bool(case.get("shared_session")),
                    "checks": {**verdict},
                    "quality_checks": {},
                    "ok": all(verdict.values()),
                }
            )
            results[-1]["quality_checks"] = _quality_checks(case, results[-1])
            results[-1]["ok"] = results[-1]["ok"] and all(results[-1]["quality_checks"].values())
        except Exception as exc:
            results.append(
                {
                    "case": case["name"],
                    "message": case["message"],
                    "error": f"{type(exc).__name__}: {exc}",
                    "ok": False,
                }
            )

    try:
        results.append(_run_preview_roundtrip())
    except Exception as exc:
        results.append(
            {
                "case": "preview_roundtrip",
                "error": f"{type(exc).__name__}: {exc}",
                "ok": False,
            }
        )

    if output_path:
        Path(output_path).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    output_file = Path(__file__).resolve().parent / "smoke_eval_output.json"
    data = run_smoke_eval(str(output_file))
    print(json.dumps(data, ensure_ascii=False, indent=2))
