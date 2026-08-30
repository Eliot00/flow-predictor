import json
import random
from pathlib import Path

import numpy as np
import requests
import torch

CACHE_FILE = Path.cwd() / "data" / "weather_cache.json"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_weather(
    lat: float,
    lon: float,
    date_str: str,
    cache: dict | None = None,
    force_refresh: bool = False,
) -> dict:
    """
    调用免费天气API，可用的值有：
    {
        'weather_code': int,      # 天气代码
        'temperature': float,     # 平均温度
        'precipitation': float,   # 总降水量 (mm)
        'wind_speed': float,      # 最大风速 (km/h)
        'humidity': float         # 平均相对湿度 (%)
    }
    """
    if cache is None:
        cache = load_cache()

    key = f"{lat:.4f}_{lon:.4f}_{date_str}"
    if not force_refresh and key in cache:
        return cache[key]

    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"
        f"windspeed_10m_max,relative_humidity_2m_mean,weathercode"
        f"&timezone=Asia/Shanghai"
        f"&start_date={date_str}&end_date={date_str}"
    )
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data.get("error"):
            print(f"API error for {date_str}: {data.get('reason', 'Unknown error')}")
            result = {
                "weather_code": 1,
                "temperature": 15.0,
                "precipitation": 0,
                "wind_speed": 5,
                "humidity": 65,
            }
        else:
            daily = data.get("daily", {})

            def safe_get(arr, idx=0, default=0.0):
                return arr[idx] if arr and arr[idx] is not None else default

            result = {
                "weather_code": int(safe_get(daily.get("weathercode", []), 0, 0)),
                "temperature": (
                    safe_get(daily.get("temperature_2m_max", []))
                    + safe_get(daily.get("temperature_2m_min", []))
                )
                / 2,
                "precipitation": safe_get(daily.get("precipitation_sum", [])),
                "wind_speed": safe_get(daily.get("windspeed_10m_max", [])),
                "humidity": safe_get(daily.get("relative_humidity_2m_mean", [])),
            }
        cache[key] = result
        save_cache(cache)
        return result
    except (requests.RequestException, ValueError, KeyError) as e:
        print(f"获取 {date_str} 天气失败: {e}")
        return {
            "weather_code": 1,
            "temperature": 15.0,
            "precipitation": 0,
            "wind_speed": 5,
            "humidity": 65,
        }


def load_cache() -> dict:
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)
