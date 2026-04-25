from __future__ import annotations

import re
from urllib.parse import quote

import requests
import urllib3

from llm import FallbackMistralLLM
from tools.base import AgentTool, ToolSpec


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


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
    value = str(text or "").strip()
    if "\n" in value:
        parts = [part.strip() for part in value.splitlines() if part.strip()]
        if parts:
            value = parts[-1]
    value = re.sub(r"^(请问|帮我|我想知道|我想问|今天|现在|目前)\s*", "", value)
    value = re.sub(
        r"(的)?(天气|气温|温度|下雨|下雪|晴天|阴天|会不会下雨|会不会下雪|weather|forecast|如何|怎么样).*$",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"\s+", " ", value).strip(" ，。！？；：")
    return value


def _normalize_location_with_llm(location_query: str) -> str:
    location_query = str(location_query or "").strip()
    if not location_query:
        return ""
    prompt = (
        "请把下面这个天气查询地点规范成适合地理编码搜索的地点字符串。\n"
        "要求：\n"
        "1. 尽量输出 City, Country 或 City, Region, Country 形式。\n"
        "2. 如果是中国城市，可以直接输出中文城市名，或输出 City, China。\n"
        "3. 如果是国际知名城市，优先用国际通用英文写法。\n"
        "4. 不要解释，只输出一个地点字符串。\n"
        f"地点：{location_query}"
    )
    try:
        value = str(FallbackMistralLLM().generate(prompt, temperature=0.0, max_tokens=40) or "").strip()
    except Exception:
        return location_query
    value = re.sub(r"\s+", " ", value).strip(" ,，。")
    return value or location_query


def _english_alias_with_llm(location_query: str) -> str:
    location_query = str(location_query or "").strip()
    if not location_query:
        return ""
    if not re.search(r"[\u4e00-\u9fff]", location_query):
        return ""
    prompt = (
        "请把下面这个中文地点转成最常见、最适合天气地理编码搜索的英文地名。\n"
        "要求：\n"
        "1. 只输出地点名本身，不要带解释。\n"
        "2. 优先输出国际上最常见的英文城市名。\n"
        "3. 不要带国家名，不要带多余标点。\n"
        f"地点：{location_query}"
    )
    try:
        value = str(FallbackMistralLLM().generate(prompt, temperature=0.0, max_tokens=20) or "").strip()
    except Exception:
        return ""
    value = re.sub(r"\s+", " ", value).strip(" ,，。")
    if re.search(r"[\u4e00-\u9fff]", value):
        return ""
    return value


def _candidate_queries(location_query: str) -> list[str]:
    base = _clean_location_query(location_query)
    normalized = _normalize_location_with_llm(base)
    english_alias = _english_alias_with_llm(base)
    values = [base, normalized, english_alias]
    expanded: list[str] = []
    for item in values:
        clean = str(item or "").strip()
        if not clean:
            continue
        expanded.append(clean)
        if "," in clean:
            expanded.append(clean.split(",")[0].strip())
        if re.search(r"[\u4e00-\u9fff]", clean):
            expanded.append(clean.replace("市", "").replace("省", "").strip())
        else:
            expanded.append(re.sub(r",\s*China$", "", clean, flags=re.IGNORECASE).strip())
            expanded.append(re.sub(r",\s*Japan$", "", clean, flags=re.IGNORECASE).strip())
    deduped: list[str] = []
    seen: set[str] = set()
    for item in expanded:
        if item and item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def _pick_best_place(results: list[dict], requested_query: str) -> dict:
    if not results:
        return {}
    requested = str(requested_query or "").strip().lower()
    normalized_requested = re.sub(r"\s+", "", requested)
    scored: list[tuple[float, dict]] = []
    for candidate in results:
        name = str(candidate.get("name", "") or "")
        admin1 = str(candidate.get("admin1", "") or "")
        country = str(candidate.get("country", "") or "")
        full = " ".join(part for part in (name, admin1, country) if part).lower()
        normalized_full = re.sub(r"\s+", "", full)
        score = 0.0
        if requested and requested in full:
            score += 2.0
        if normalized_requested and normalized_requested in normalized_full:
            score += 2.0
        if name.lower() == requested:
            score += 1.5
        if admin1.lower() == requested:
            score += 1.0
        feature_code = str(candidate.get("feature_code", "") or "").upper()
        if feature_code == "PPLC":
            score += 8.0
        elif feature_code == "PPLA":
            score += 6.5
        elif feature_code == "PPLA2":
            score += 5.0
        elif feature_code == "PPL":
            score += 0.5
        population = candidate.get("population")
        try:
            score += min(float(population or 0.0) / 5_000_000.0, 4.0)
        except Exception:
            pass
        scored.append((score, candidate))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1] if scored else results[0]


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
        location_query = _clean_location_query(query)
        if not location_query:
            return {"ok": False, "location": "", "summary": "用户没有提供明确地点，无法查询天气。"}

        try:
            results = []
            used_query = location_query
            for candidate_query in _candidate_queries(location_query):
                geo = requests.get(
                    "https://geocoding-api.open-meteo.com/v1/search",
                    params={"name": candidate_query, "count": 8, "language": "zh", "format": "json"},
                    timeout=8,
                    verify=False,
                )
                geo.raise_for_status()
                geo_data = geo.json()
                candidate_results = geo_data.get("results") or []
                if candidate_results:
                    results.extend(candidate_results)
                    used_query = candidate_query
            if not results:
                return {"ok": False, "location": location_query, "summary": f"没有找到“{location_query}”对应的地点。"}

            place = _pick_best_place(results, used_query)
            latitude = place["latitude"]
            longitude = place["longitude"]
            location = place.get("name", location_query)
            admin1 = place.get("admin1")
            country = place.get("country")
            place_name = " / ".join([part for part in (location, admin1, country) if part])

            try:
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
                    verify=False,
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
            except Exception:
                fallback_url = f"https://wttr.in/{quote(place_name)}?format=j1"
                fallback = requests.get(fallback_url, timeout=8, verify=False)
                fallback.raise_for_status()
                fallback_data = fallback.json()
                current = (fallback_data.get("current_condition") or [{}])[0]
                today = (fallback_data.get("weather") or [{}])[0]
                desc = ((current.get("weatherDesc") or [{}])[0].get("value") or "天气未知").strip()
                temp = current.get("temp_C")
                feels = current.get("FeelsLikeC")
                wind = current.get("windspeedKmph")
                max_temp = today.get("maxtempC")
                min_temp = today.get("mintempC")
                rain = today.get("daily_chance_of_rain")
                summary = (
                    f"{place_name}当前{desc}，气温 {temp}°C，"
                    f"体感 {feels}°C，风速 {wind} km/h。"
                    f"今天最高 {max_temp}°C，最低 {min_temp}°C，降雨概率约 {rain}%。"
                )
                return {"ok": True, "location": place_name, "summary": summary}
        except Exception as exc:
            return {"ok": False, "location": location_query, "summary": f"天气查询失败：{exc}"}
