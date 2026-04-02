import re

import requests

from tools.base import AgentTool, ToolSpec


def _clean_location_query(text):
    value = (text or "").strip()
    value = re.sub(r"(今天|今日|现在|当前|请问|帮我|查一下|看看|告诉我)", "", value)
    value = re.sub(r"(天气|气温|温度|下雨|会下雨吗|会不会下雨|weather|forecast)", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip(" ，。！？")
    return value


class WeatherTool(AgentTool):
    spec = ToolSpec(
        name="weather",
        description="Look up today's weather for a location.",
        input_schema={"query": "str"},
        output_schema={
            "ok": "bool",
            "location": "str",
            "summary": "str",
        },
        tags=["realtime", "weather"],
    )

    def run(self, query):
        location_query = _clean_location_query(query)
        if not location_query:
            return {
                "ok": False,
                "location": "",
                "summary": "用户没有提供明确地点，无法查询天气。",
            }

        try:
            geo = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": location_query, "count": 1, "language": "zh", "format": "json"},
                timeout=8,
            )
            geo.raise_for_status()
            geo_data = geo.json()
            results = geo_data.get("results") or []
            if not results:
                return {"ok": False, "location": location_query, "summary": f"没有找到“{location_query}”对应的地点。"}
            place = results[0]
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
            summary = (
                f"{place_name}当前 {current.get('temperature_2m')}°C，体感 {current.get('apparent_temperature')}°C，"
                f"风速 {current.get('wind_speed_10m')} km/h。"
                f"今日最高 {max_temp}°C，最低 {min_temp}°C，降水概率约 {rain}%。"
            )
            return {"ok": True, "location": place_name, "summary": summary}
        except Exception as exc:
            return {"ok": False, "location": location_query, "summary": f"天气查询失败：{exc}"}
