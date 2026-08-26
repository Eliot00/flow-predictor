import numpy as np
import pandas as pd


def fake_all():
    # 模拟10年（chinese_calendar 数据只覆盖到 2026）
    start_date = "2016-01-01"
    end_date = "2026-08-01"
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    # 生成5家店铺，店铺属性信息是固定的
    n_stores = 5
    store_ids = np.arange(1, n_stores + 1)
    areas = np.random.randint(50, 500, size=n_stores)
    longs = np.random.uniform(90.0, 119.0, size=n_stores)
    lats = np.random.uniform(30.0, 45.0, size=n_stores)
    categories = np.random.choice(["化妆品", "手表", "服装"], size=n_stores)

    # 和日期笛卡尔积
    df = pd.DataFrame(
        [
            (date, sid, area, lon, lat, cat)
            for date in dates
            for sid, area, lon, lat, cat in zip(
                store_ids, areas, longs, lats, categories
            )
        ],
        columns=["date", "sid", "area", "longitude", "latitude", "category"],
    )

    # 天气理论上可以用API拿真实的（真实数据还会和地理位置有关），先随机
    weather_choices = ["晴", "阴", "雨"]
    weather_probs = [0.6, 0.25, 0.15]
    df["weather"] = np.random.choice(weather_choices, size=len(df), p=weather_probs)

    # 模拟下四季，国内差不多零下10到40吧
    day_of_year = df["date"].dt.dayofyear
    base_temp = 15 + 25 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    df["temperature"] = base_temp + np.random.normal(0, 3, size=len(df))

    # 周末，节假日，调休（chinese_calendar：is_workday 对调休上班的周末返回 True）
    import chinese_calendar as cn_cal
    df["is_workday"] = [int(cn_cal.is_workday(d.date())) for d in df["date"]]
    df["is_weekend"] = df["date"].dt.weekday.isin([5, 6]).astype(int)
    df["is_holiday"] = [int(not cn_cal.is_workday(d.date())) for d in df["date"]]

    # 生成假的客流，基于一些假设的关系
    # 面积越大、位置越好，客流越大
    base_passby = 200 + 0.8 * df["area"] + 20 * (df["latitude"] - 30)
    # 天气系数，晴天人多，雨天人少
    weather_factor = df["weather"].map({"晴": 1.0, "阴": 0.8, "雨": 0.5})
    # 温度，太冷太热不行
    temp_factor = 1 - 0.004 * (df["temperature"] - 20) ** 2
    temp_factor = temp_factor.clip(0.4, 1.0)
    # 周末人多，法定假日更多；调休上班的周末接近工作日
    weekend_factor = 1 + 0.3 * df["is_weekend"] + 0.25 * df["is_holiday"] - 0.15 * ((df["is_weekend"] == 1) & (df["is_workday"] == 1)).astype(int)
    # 加点随机波动
    noise = np.random.normal(1, 0.12, len(df))
    df["passby_visit"] = (
        (base_passby * weather_factor * temp_factor * weekend_factor * noise)
        .round()
        .astype(int)
    )
    df["passby_visit"] = df["passby_visit"].clip(20, None)  # 加个最小20个路过

    # 基础进店率15%，雨天进店率翻倍（躲雨），服装店进店率高一些
    enter_rate = (
        0.15
        + 0.12 * (df["weather"] == "雨").astype(int)
        + 0.05 * (df["category"] == "服装").astype(int)
    )
    df["entering_people"] = (
        (df["passby_visit"] * enter_rate * np.random.normal(1, 0.10, len(df)))
        .round()
        .astype(int)
    )
    df["entering_people"] = df["entering_people"].clip(0)

    # 基础停留率50%，面积大停留率高，手表类目停留率高
    dwell_rate = (
        0.5
        + 0.15 * (df["area"] / df["area"].max())
        + 0.10 * (df["category"] == "手表").astype(int)
    )
    dwell_rate = dwell_rate.clip(0.3, 0.95)
    df["dwell_people"] = (
        (df["entering_people"] * dwell_rate * np.random.normal(1, 0.10, len(df)))
        .round()
        .astype(int)
    )
    df["dwell_people"] = df["dwell_people"].clip(0)

    # 基础服务率50%，手表类目服务率高
    serve_rate = (
        0.5
        + 0.10 * (df["category"] == "手表").astype(int)
    )
    serve_rate = serve_rate.clip(0.2, 0.85)
    df["served_people"] = (
        (df["dwell_people"] * serve_rate * np.random.normal(1, 0.10, len(df)))
        .round()
        .astype(int)
    )
    df["served_people"] = df["served_people"].clip(0)

    return df
