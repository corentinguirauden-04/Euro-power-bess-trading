"""Unit tests for Stage 1 modules (PowerMarketSimulator, BatteryAsset, TradingEnv)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.asset import BatteryAsset
from src.market_sim import MarketConfig, PowerMarketSimulator
from src.trading_env import TradingEnv


def test_market_simulator_schema_and_granularity() -> None:
    """Simulator produces 96 steps/day at 15-minute intervals with expected columns."""
    sim = PowerMarketSimulator(config=MarketConfig(), seed=42)
    df = sim.simulate(n_days=3, start_date="2025-05-12", seed=42)

    expected_cols = [
        "timestamp",
        "demand_mw",
        "solar_mw",
        "wind_mw",
        "net_load_mw",
        "price_eur_mwh",
    ]
    assert list(df.columns) == expected_cols
    assert len(df) == 3 * 96
    assert (df["timestamp"].diff().dropna() == pd.Timedelta(minutes=15)).all()
    assert (df["solar_mw"] >= 0.0).all()
    assert (df["wind_mw"] >= 0.0).all()
    assert np.allclose(
        df["net_load_mw"],
        df["demand_mw"] - df["solar_mw"] - df["wind_mw"],
        atol=0.05,
    )


def test_market_simulator_reproducibility_and_regimes() -> None:
    """Same seed produces identical paths and exhibits both negative prices and spikes."""
    sim = PowerMarketSimulator(seed=42)
    df1 = sim.simulate(n_days=5, seed=42)
    df2 = sim.simulate(n_days=5, seed=42)

    pd.testing.assert_frame_equal(df1, df2)
    assert (df1["price_eur_mwh"] < 0.0).any(), "Expected negative price regime under high RES"
    assert (df1["price_eur_mwh"] > 150.0).any(), "Expected scarcity/peaker price spikes"


def test_battery_power_clipping_and_efficiency() -> None:
    """Battery clips power to max_power_mw and accounts for one-way efficiency."""
    battery = BatteryAsset(
        capacity_mwh=10.0,
        max_power_mw=5.0,
        efficiency_charge=0.92,
        efficiency_discharge=0.92,
        soc_min=0.10,
        soc_max=0.90,
        degradation_cost_eur_mwh=2.0,
        initial_soc=0.50,
    )

    # Request 10 MW discharge -> clipped to 5 MW over 0.25 h = 1.25 MWh exported
    energy_mwh, new_soc = battery.step(power_mw=10.0, dt=0.25)
    assert np.isclose(energy_mwh, 1.25)
    assert np.isclose(new_soc, 0.50 - (1.25 / 0.92) / 10.0)

    # Request -10 MW charge -> clipped to -5 MW over 0.25 h = -1.25 MWh imported
    prev_soc = new_soc
    energy_mwh, new_soc = battery.step(power_mw=-10.0, dt=0.25)
    assert np.isclose(energy_mwh, -1.25)
    assert np.isclose(new_soc, prev_soc + (1.25 * 0.92) / 10.0)


def test_battery_soc_hard_bounds() -> None:
    """Battery never violates [soc_min, soc_max] under sustained charge/discharge."""
    battery = BatteryAsset(initial_soc=0.50)

    for _ in range(30):
        battery.step(power_mw=-5.0, dt=0.25)
    assert np.isclose(battery.soc, battery.soc_max)
    e_full, soc_full = battery.step(power_mw=-5.0, dt=0.25)
    assert np.isclose(e_full, 0.0)
    assert np.isclose(soc_full, battery.soc_max)

    for _ in range(30):
        battery.step(power_mw=5.0, dt=0.25)
    assert np.isclose(battery.soc, battery.soc_min)
    e_empty, soc_empty = battery.step(power_mw=5.0, dt=0.25)
    assert np.isclose(e_empty, 0.0)
    assert np.isclose(soc_empty, battery.soc_min)


def test_trading_env_negative_price_and_pnl_accounting() -> None:
    """Charging during negative prices earns positive cashflow minus degradation."""
    market_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-05-12", periods=2, freq="15min", tz="UTC"),
            "demand_mw": [30_000.0, 48_000.0],
            "solar_mw": [18_000.0, 0.0],
            "wind_mw": [10_000.0, 3_000.0],
            "net_load_mw": [2_000.0, 45_000.0],
            "price_eur_mwh": [-40.0, 200.0],
        }
    )
    battery = BatteryAsset(
        capacity_mwh=10.0,
        max_power_mw=5.0,
        efficiency_charge=0.92,
        efficiency_discharge=0.92,
        degradation_cost_eur_mwh=2.0,
        initial_soc=0.50,
    )
    env = TradingEnv(market_data=market_df, battery=battery)

    # Buy 1 MWh at -40 EUR/MWh -> cashflow = +40 EUR, degradation = 2 EUR -> net = +38 EUR
    res0 = env.step(power_mw=-4.0)
    assert np.isclose(res0.energy_exchanged_mwh, -1.0)
    assert np.isclose(res0.cashflow_eur, 40.0)
    assert np.isclose(res0.degradation_cost_eur, 2.0)
    assert np.isclose(res0.net_pnl_eur, 38.0)
    assert not res0.done

    # Sell 1 MWh at +200 EUR/MWh -> cashflow = +200 EUR, degradation = 2 EUR -> net = +198 EUR
    res1 = env.step(power_mw=4.0)
    assert np.isclose(res1.energy_exchanged_mwh, 1.0)
    assert np.isclose(res1.cashflow_eur, 200.0)
    assert np.isclose(res1.net_pnl_eur, 198.0)
    assert np.isclose(res1.cumulative_pnl_eur, 236.0)
    assert res1.done

    with pytest.raises(RuntimeError):
        env.step(power_mw=0.0)
