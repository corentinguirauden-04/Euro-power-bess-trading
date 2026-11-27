"""Stage 1 validation and visualization script for euro-power-bess-trading.

Simulates 3 days of 15-minute European power market data, executes a simple
quantile-based threshold trading strategy on a 10 MWh / 5 MW BESS asset, prints
financial and physical KPIs, and generates a 3-panel Matplotlib diagnostic chart.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Callable

# Ensure repository root is importable when executed directly as `python scripts/demo_stage1.py`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.asset import BatteryAsset
from src.market_sim import MarketConfig, PowerMarketSimulator
from src.trading_env import TradingEnv


def build_quantile_policy(
    prices: pd.Series,
    max_power_mw: float,
    low_quantile: float = 0.25,
    high_quantile: float = 0.75,
) -> tuple[Callable[[pd.Series, float], float], float, float]:
    """Create a simple threshold arbitrage strategy based on price quantiles.

    Buys (charges at ``-max_power_mw``) when Spot price is below ``q_low``,
    sells (discharges at ``+max_power_mw``) when Spot price is above ``q_high``,
    and remains idle otherwise.

    Args:
        prices: Simulated Spot price series in EUR/MWh.
        max_power_mw: BESS maximum power rating in MW.
        low_quantile: Lower quantile threshold for charging (default 0.25).
        high_quantile: Upper quantile threshold for discharging (default 0.75).

    Returns:
        Tuple ``(policy_fn, q_low_eur_mwh, q_high_eur_mwh)``.
    """
    q_low = float(prices.quantile(low_quantile))
    q_high = float(prices.quantile(high_quantile))

    def policy(market_row: pd.Series, _soc: float) -> float:
        price = float(market_row["price_eur_mwh"])
        if price < q_low:
            return -max_power_mw
        if price > q_high:
            return max_power_mw
        return 0.0

    return policy, q_low, q_high


def plot_stage1_diagnostics(
    results: pd.DataFrame,
    battery: BatteryAsset,
    q_low: float,
    q_high: float,
    summary: dict[str, float],
    save_path: Path,
    show: bool = True,
) -> None:
    """Render a 3-panel diagnostic figure covering Spot price, Mix/Net Load, and BESS state."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 10.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.15, 1.15]},
    )

    ts = results["timestamp"]
    prices = results["price_eur_mwh"]

    # --- Panel 1: Spot Price (EUR/MWh) ---
    ax_price = axes[0]
    ax_price.plot(ts, prices, color="#1f4e79", linewidth=1.8, label="Spot Price (€/MWh)")
    ax_price.axhline(0.0, color="#333333", linewidth=0.9, linestyle="-")
    ax_price.axhline(
        q_low,
        color="#2e8b57",
        linewidth=1.1,
        linestyle="--",
        label=f"Q25 Charge Threshold ({q_low:.1f} €/MWh)",
    )
    ax_price.axhline(
        q_high,
        color="#c0392b",
        linewidth=1.1,
        linestyle="--",
        label=f"Q75 Discharge Threshold ({q_high:.1f} €/MWh)",
    )
    ax_price.fill_between(
        ts,
        prices,
        0.0,
        where=prices < 0.0,
        interpolate=True,
        color="#e74c3c",
        alpha=0.22,
        label="Negative Price Regime",
    )
    price_span = max(float(prices.max() - prices.min()), 50.0)
    ax_price.set_ylim(float(prices.min()) - 0.10 * price_span, float(prices.max()) + 0.24 * price_span)
    ax_price.set_ylabel("Spot Price (€/MWh)", fontweight="bold")
    ax_price.set_title(
        f"European Power Market & BESS Naive Arbitrage (3 Days, 15-min)  |  "
        f"Total Net PnL: €{summary['total_pnl_eur']:,.2f}  "
        f"(Cycles: {summary['equivalent_full_cycles']:.2f})",
        fontsize=12.5,
        fontweight="bold",
        pad=10,
    )
    ax_price.legend(loc="upper left", ncol=2, frameon=True, fontsize=8.8)

    # --- Panel 2: Generation Mix & Net Load (MW) ---
    ax_mix = axes[1]
    ax_mix.plot(ts, results["demand_mw"] / 1_000.0, color="#2c3e50", linewidth=1.8, label="Demand")
    ax_mix.plot(
        ts,
        results["net_load_mw"] / 1_000.0,
        color="#8e44ad",
        linewidth=1.8,
        linestyle="-",
        label="Net Load (Demand - Solar - Wind)",
    )
    ax_mix.fill_between(
        ts,
        0.0,
        results["solar_mw"] / 1_000.0,
        color="#f39c12",
        alpha=0.35,
        label="Solar PV (Duck Curve)",
    )
    ax_mix.fill_between(
        ts,
        0.0,
        results["wind_mw"] / 1_000.0,
        color="#3498db",
        alpha=0.28,
        label="Wind (OU Process)",
    )
    ax_mix.set_ylim(0.0, float(results["demand_mw"].max()) / 1_000.0 * 1.26)
    ax_mix.set_ylabel("Power (GW)", fontweight="bold")
    ax_mix.legend(loc="upper left", ncol=2, frameon=True, fontsize=8.8)

    # --- Panel 3: BESS State of Charge (%) & Dispatch Power (MW) ---
    ax_soc = axes[2]
    ax_pwr = ax_soc.twinx()

    power = results["actual_power_mw"]
    bar_width = 0.25 / 24.0 * 0.85  # 15-min width in matplotlib date units
    discharge_vals = np.where(power > 0.0, power, 0.0)
    charge_vals = np.where(power < 0.0, power, 0.0)

    bars_dis = ax_pwr.bar(
        ts,
        discharge_vals,
        width=bar_width,
        color="#e67e22",
        alpha=0.45,
        label="Discharge / Sell (+MW)",
    )
    bars_chg = ax_pwr.bar(
        ts,
        charge_vals,
        width=bar_width,
        color="#27ae60",
        alpha=0.45,
        label="Charge / Buy (-MW)",
    )
    ax_pwr.axhline(0.0, color="#7f8c8d", linewidth=0.8)
    ax_pwr.set_ylim(-battery.max_power_mw * 1.35, battery.max_power_mw * 1.55)
    ax_pwr.set_ylabel("BESS Power (MW)", fontweight="bold", color="#555555")
    ax_pwr.grid(False)

    ax_soc.plot(
        ts,
        results["soc_after"] * 100.0,
        color="#111111",
        linewidth=2.0,
        zorder=5,
        label="State of Charge (%)",
    )
    ax_soc.axhline(
        battery.soc_min * 100.0,
        color="#7f8c8d",
        linestyle=":",
        linewidth=1.1,
        label=f"SoC Bounds ({battery.soc_min*100:.0f}%–{battery.soc_max*100:.0f}%)",
    )
    ax_soc.axhline(battery.soc_max * 100.0, color="#7f8c8d", linestyle=":", linewidth=1.1)
    ax_soc.set_ylim(0.0, 122.0)
    ax_soc.set_ylabel("State of Charge (%)", fontweight="bold")
    ax_soc.set_xlabel("Timestamp (UTC)", fontweight="bold")
    ax_soc.set_zorder(ax_pwr.get_zorder() + 1)
    ax_soc.patch.set_visible(False)

    # Combine legends from both y-axes cleanly
    handles_soc, labels_soc = ax_soc.get_legend_handles_labels()
    handles_pwr = [bars_dis, bars_chg]
    labels_pwr = ["Discharge / Sell (+MW)", "Charge / Buy (-MW)"]
    ax_soc.legend(
        handles_soc + handles_pwr,
        labels_soc + labels_pwr,
        loc="upper left",
        ncol=2,
        frameon=True,
        fontsize=8.6,
    )

    # Clean time formatting on x-axis
    ax_soc.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
    ax_soc.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    fig.autofmt_xdate(rotation=0, ha="center")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved Stage 1 diagnostic chart to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def print_report(
    market_df: pd.DataFrame,
    summary: dict[str, float],
    q_low: float,
    q_high: float,
) -> None:
    """Print a concise console report of market regimes and BESS PnL."""
    prices = market_df["price_eur_mwh"]
    neg_intervals = int((prices < 0.0).sum())
    spike_intervals = int((prices > 150.0).sum())

    print("=" * 68)
    print("EURO-POWER BESS TRADING — STAGE 1 SIMULATION REPORT")
    print("=" * 68)
    print(
        f"Horizon           : {len(market_df)} intervals "
        f"({len(market_df) * 0.25:.1f} hours / {len(market_df) // 96} days)"
    )
    print(
        f"Spot Price Range  : min {prices.min():7.2f} | "
        f"mean {prices.mean():7.2f} | max {prices.max():7.2f} €/MWh"
    )
    print(
        f"Extreme Regimes   : {neg_intervals} negative-price intervals "
        f"({neg_intervals * 0.25:.2f} h) | "
        f"{spike_intervals} spike (>150 €/MWh) intervals ({spike_intervals * 0.25:.2f} h)"
    )
    print(f"Naive Thresholds  : Charge < {q_low:6.2f} €/MWh | Discharge > {q_high:6.2f} €/MWh")
    print("-" * 68)
    print(f"Gross Cashflow    : €{summary['gross_cashflow_eur']:10,.2f}")
    print(f"Degradation Cost  : €{summary['total_degradation_eur']:10,.2f}")
    print(f"Total Net PnL     : €{summary['total_pnl_eur']:10,.2f}")
    print("-" * 68)
    print(f"Energy Charged    : {summary['energy_charged_mwh']:8.2f} MWh "
          f"(avg price: {summary['avg_charge_price_eur_mwh']:6.2f} €/MWh)")
    print(f"Energy Discharged : {summary['energy_discharged_mwh']:8.2f} MWh "
          f"(avg price: {summary['avg_discharge_price_eur_mwh']:6.2f} €/MWh)")
    print(f"Equivalent Cycles : {summary['equivalent_full_cycles']:8.2f}")
    print(f"Final SoC         : {summary['final_soc'] * 100.0:7.1f}%")
    print("=" * 68)


def parse_args() -> argparse.Namespace:
    """Parse CLI options for the Stage 1 demo script."""
    parser = argparse.ArgumentParser(
        description="Run Stage 1 European Power Market & BESS simulation demo."
    )
    parser.add_argument("--days", type=int, default=3, help="Number of days to simulate (default: 3).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument(
        "--save-path",
        type=Path,
        default=REPO_ROOT / "outputs" / "stage1_demo.png",
        help="Path to save the 3-panel diagnostic figure.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the chart without opening an interactive Matplotlib window.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the Stage 1 simulation, naive arbitrage test, and visualization."""
    args = parse_args()

    simulator = PowerMarketSimulator(config=MarketConfig(), seed=args.seed)
    market_df = simulator.simulate(n_days=args.days, start_date="2025-05-12")

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

    env = TradingEnv(market_data=market_df, battery=battery)
    policy, q_low, q_high = build_quantile_policy(
        prices=market_df["price_eur_mwh"],
        max_power_mw=battery.max_power_mw,
        low_quantile=0.25,
        high_quantile=0.75,
    )

    results = env.run_strategy(policy)
    summary = env.get_summary()

    print_report(market_df, summary, q_low, q_high)
    plot_stage1_diagnostics(
        results=results,
        battery=battery,
        q_low=q_low,
        q_high=q_high,
        summary=summary,
        save_path=args.save_path,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
