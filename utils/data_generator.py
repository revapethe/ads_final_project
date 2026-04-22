import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)


def generate_supply_chain_data(n_days=730, disruption_rate=0.04):
    base_date = datetime(2022, 1, 1)
    dates = [base_date + timedelta(days=i) for i in range(n_days)]
    records = []

    for i, date in enumerate(dates):
        day_of_year = date.timetuple().tm_yday
        season_factor = 1 + 0.35 * np.sin((day_of_year / 365) * 2 * np.pi - np.pi / 2)

        shipment_delay = max(0, RNG.normal(8, 3) * season_factor)
        inventory_level = max(5, min(100, RNG.normal(65, 15) / season_factor))
        lead_time = max(1, RNG.normal(7, 2) * season_factor)
        demand_index = max(0, RNG.normal(100, 20) * season_factor)
        transport_stress = max(0, min(100, RNG.normal(40, 15) * season_factor))
        weather_risk = max(0, min(100, RNG.normal(20, 10)))
        supplier_rel = max(0, min(100, RNG.normal(78, 12)))

        disrupted = 0
        if RNG.random() < disruption_rate:
            disrupted = 1
            shipment_delay *= RNG.uniform(2.5, 5.0)
            inventory_level *= RNG.uniform(0.1, 0.4)
            lead_time *= RNG.uniform(2.0, 4.0)
            transport_stress = min(100, transport_stress * RNG.uniform(1.5, 2.5))
            supplier_rel = max(0, supplier_rel * RNG.uniform(0.3, 0.6))

        records.append({
            "date": date,
            "supplier_id": "SUP_" + str(RNG.integers(1, 21)).zfill(2),
            "shipment_delay_hours": round(shipment_delay, 2),
            "inventory_level_pct": round(inventory_level, 2),
            "lead_time_days": round(lead_time, 2),
            "demand_index": round(demand_index, 2),
            "transport_stress": round(transport_stress, 2),
            "weather_risk": round(weather_risk, 2),
            "supplier_reliability": round(supplier_rel, 2),
            "disruption": disrupted,
        })

    df = pd.DataFrame(records)
    df = _add_temporal_features(df)
    return df


def _add_temporal_features(df):
    df = df.sort_values("date").reset_index(drop=True)

    for col in ["shipment_delay_hours", "inventory_level_pct", "lead_time_days"]:
        df[col + "_7d_avg"] = df[col].shift(1).rolling(7, min_periods=1).mean()
        df[col + "_velocity"] = df[col].diff(3)

    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["month"] = pd.to_datetime(df["date"]).dt.month
    df["quarter"] = pd.to_datetime(df["date"]).dt.quarter
    df["is_holiday_qtr"] = (df["quarter"] == 4).astype(int)

    df["composite_risk"] = (
        (df["shipment_delay_hours"] / 50) * 30 +
        ((100 - df["inventory_level_pct"]) / 100) * 25 +
        (df["transport_stress"] / 100) * 25 +
        (df["weather_risk"] / 100) * 10 +
        ((100 - df["supplier_reliability"]) / 100) * 10
    ).clip(0, 100).round(2)

    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df


def get_feature_columns():
    return [
        "shipment_delay_hours", "inventory_level_pct", "lead_time_days",
        "demand_index", "transport_stress", "weather_risk", "supplier_reliability",
        "shipment_delay_hours_7d_avg", "inventory_level_pct_7d_avg",
        "lead_time_days_7d_avg", "shipment_delay_hours_velocity",
        "inventory_level_pct_velocity", "lead_time_days_velocity",
        "day_of_week", "month", "quarter", "is_holiday_qtr",
    ]
