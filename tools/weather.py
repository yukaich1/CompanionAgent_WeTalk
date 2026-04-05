from __future__ import annotations

import re

import requests

from tools.base import AgentTool, ToolSpec


LOCATION_ALIASES = {
    "东京": "Tokyo, Japan",
    "东京都": "Tokyo, Japan",
    "东京日本": "Tokyo, Japan",
    "tokyo": "Tokyo, Japan",
    "北京": "Beijing, China",
    "上海": "Shanghai, China",
    "大阪": "Osaka, Japan",
    "京都": "Kyoto, Japan",
    "横滨": "Yokohama, Japan",
    "福冈": "Fukuoka, Japan",
}


WEATHER_CODES = {
    0: "晴朗",
    1: "大致晴",
    2: "局部多云",
    3: "阴",
    45: "雾",
    48: "冻雾",
    51: "小毛雨",
    53: "毛雨",
    55: "强毛雨",
    61: "小雨",
    63: "降雨",
    65: "大雨",
    71: "小雪",
    73: "降雪",
    75: "大雪",
    80: "阵雨",
    81: "较强阵雨",
    82: "强阵雨",
    95: "雷暴",
}


def _clean_location_query(text: str) -> str:
    value = (text or "").strip()
    if "\n" in value:
        parts = [part.strip() for part in value.splitlines() if part.strip()]
        if parts:
            value = parts[-1]
    value = re.sub(r"(今天|今日|现在|当前|请问|帮我|查一下|看一下|看看|告诉我)", "", value)
    value = re.sub(
        r"(天气|气温|温度|下雨|下雪|晴天|阴天|会下雨吗|会不会下雨|weather|forecast|如何|怎么样|怎样)",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"\s+", " ", value).strip(" ，。！？?；：")
    return value


def _canonical_location_query(query: str) -> str:
    cleaned = _clean_location_query(query)
    if not cleaned:
        return ""
    lowered = cleaned.lower()
    if lowered in LOCATION_ALIASES:
        return LOCATION_ALIASES[lowered]
    if cleaned in LOCATION_ALIASES:
        return LOCATION_ALIASES[cleaned]
    return cleaned


def _pick_best_place(results: list[dict], requested_query: str) -> dict:
    if not results:
        return {}

    query_lower = requested_query.lower()
    if "tokyo" in query_lower or "东京" in requested_query or "东京都" in requested_query:
        for candidate in results:
            country = str(candidate.get("country", "") or "")
            name = str(candidate.get("name", "") or "")
            admin1 = str(candidate.get("admin1", "") or "")
            if country in {"Japan", "日本"} and (
                name == "Tokyo" or "Tokyo" in admin1 or "东京" in admin1 or "东京都" in admin1
            ):
                return candidate
    return results[0]


def _describe_weather(code: int | None) -> str:
    if code is None:
        return "天气未知"
    return WEATHER_CODES.get(int(code), "天气一般")


class WeatherTool(AgentTool):
    spec = ToolSpec(
        name="weather",
        description="Look up today's weather for a location.",
        input_schema={"query": "str"},
        output_schema={"ok": "bool", "location": "str", "summary": "str"},
        tags=["realtime", "weather"],
    )

    def run(self, query):
        location_query = _canonical_location_query(query)
        if not location_query:
            return {"ok": False, "location": "", "summary": "用户没有提供明确地点，无法查询天气。"}

        try:
            geo = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": location_query, "count": 5, "language": "zh", "format": "json"},
                timeout=8,
            )
            geo.raise_for_status()
            geo_data = geo.json()
            results = geo_data.get("results") or []
            if not results:
                return {"ok": False, "location": location_query, "summary": f"没有找到“{location_query}”对应的地点。"}

            place = _pick_best_place(results, location_query)
            latitude = place["latitude"]
            longitude = place["longitude"]
            location = place.get("name", location_query)
            admin1 = place.get("admin1")
            country = place.get("country")
            place_name = " / ".join([part for part in (location, admin1, country) if part])

            weather = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
                    "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                    "timezone": "auto",
                    "forecast_days": 1,
                },
                timeout=8,
            )
            weather.raise_for_status()
            weather_data = weather.json()
            current = weather_data.get("current", {})
            daily = weather_data.get("daily", {})
            max_temp = (daily.get("temperature_2m_max") or [None])[0]
            min_temp = (daily.get("temperature_2m_min") or [None])[0]
            rain = (daily.get("precipitation_probability_max") or [None])[0]
            condition = _describe_weather(current.get("weather_code"))

            summary = (
                f"{place_name}当前{condition}，气温 {current.get('temperature_2m')}°C，"
                f"体感 {current.get('apparent_temperature')}°C，风速 {current.get('wind_speed_10m')} km/h。"
                f"今天最高 {max_temp}°C，最低 {min_temp}°C，降水概率约 {rain}%。"
            )
            return {"ok": True, "location": place_name, "summary": summary}
        except Exception as exc:
            return {"ok": False, "location": location_query, "summary": f"天气查询失败：{exc}"}
