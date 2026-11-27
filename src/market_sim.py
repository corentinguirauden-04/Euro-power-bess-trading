"""European power market simulator at 15-minute resolution.

Models fundamental electricity market drivers—diurnal demand profiles, solar
generation (Duck Curve), Ornstein-Uhlenbeck wind dynamics, and unplanned
supply/demand shocks—mapped to Day-Ahead / Spot clearing prices via a
piecewise Merit Order supply stack capable of producing both negative prices
and scarcity spikes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketConfig:
    """Configuration parameters for the European power market simulator.

    All power quantities are expressed in MW and prices in EUR/MWh unless
    otherwise specified.

    Attributes:
        dt_hours: Time step duration in hours (0.25 = 15-minute intervals).
        base_demand_mw: Overnight base electricity demand in MW.
        morning_peak_mw: Incremental demand for the morning ramp peak in MW.
        evening_peak_mw: Incremental demand for the evening peak in MW.
        demand_noise_std_mw: Standard deviation of AR(1) demand forecast noise.
        solar_capacity_mw: Installed solar PV peak generation potential in MW.
        wind_capacity_mw: Installed wind capacity ceiling in MW.
        wind_mean_mw: Long-run mean wind generation level in MW.
        wind_theta: Mean-reversion speed of the Ornstein-Uhlenbeck wind process (1/hour).
        wind_sigma: Volatility parameter of the Ornstein-Uhlenbeck wind process (MW / sqrt(hour)).
        must_run_mw: Inflexible must-run / nuclear baseload floor in MW. When
            net load falls below this threshold, curtailment / negative prices occur.
        nuclear_capacity_mw: Total nuclear / low-marginal-cost baseload capacity in MW.
        mid_merit_capacity_mw: Coal / efficient CCGT mid-merit stack width in MW.
        gas_capacity_mw: Marginal gas CCGT stack width in MW.
        peaker_capacity_mw: OCGT / oil peaker stack width in MW.
        negative_price_floor_eur_mwh: Minimum clearing price cap (EUR/MWh).
        scarcity_price_cap_eur_mwh: Maximum scarcity price cap (EUR/MWh).
        shock_prob_per_step: Probability per 15-minute step of initiating a
            supply outage / scarcity shock event.
        shock_duration_steps: Typical duration (in 15-minute steps) of a shock event.
        shock_magnitude_mw: Equivalent supply loss (or net load jump) during a shock in MW.
        price_noise_std_eur_mwh: Microstructure noise added to the clearing price.
    """

    dt_hours: float = 0.25

    # Demand profile parameters (representative regional European bidding zone)
    base_demand_mw: float = 34_000.0
    morning_peak_mw: float = 11_000.0
    evening_peak_mw: float = 14_500.0
    demand_noise_std_mw: float = 550.0

    # Renewable generation parameters
    solar_capacity_mw: float = 24_000.0
    wind_capacity_mw: float = 22_000.0
    wind_mean_mw: float = 9_500.0
    wind_theta: float = 0.12
    wind_sigma: float = 2_400.0

    # Merit Order supply stack capacities (MW of net load)
    must_run_mw: float = 17_500.0
    nuclear_capacity_mw: float = 23_000.0
    mid_merit_capacity_mw: float = 8_500.0
    gas_capacity_mw: float = 7_000.0
    peaker_capacity_mw: float = 4_500.0

    # Price bounds and stochastic shocks
    negative_price_floor_eur_mwh: float = -80.0
    scarcity_price_cap_eur_mwh: float = 450.0
    shock_prob_per_step: float = 0.008
    shock_duration_steps: int = 8
    shock_magnitude_mw: float = 4_500.0
    price_noise_std_eur_mwh: float = 2.5


class PowerMarketSimulator:
    """Fundamental European Day-Ahead / Spot electricity market simulator.

    Simulates 15-minute intervals (96 steps/day) of system demand, solar PV
    generation, wind output, net load, and resulting Spot clearing prices
    derived from a stylized European Merit Order curve.
    """

    STEPS_PER_DAY: int = 96

    def __init__(
        self,
        config: Optional[MarketConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the power market simulator.

        Args:
            config: Market configuration dataclass. Uses defaults if omitted.
            seed: Optional random seed for reproducible simulations.
        """
        self.config: MarketConfig = config if config is not None else MarketConfig()
        if not np.isclose(self.config.dt_hours, 0.25):
            raise ValueError(
                f"PowerMarketSimulator requires 15-minute granularity (dt_hours=0.25), "
                f"got {self.config.dt_hours}."
            )
        self._seed: Optional[int] = seed
        self._rng: np.random.Generator = np.random.default_rng(seed)

    def simulate(
        self,
        n_days: int = 3,
        start_date: str = "2025-05-12",
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Simulate the electricity market over a multi-day horizon.

        Args:
            n_days: Number of full days to simulate (must be >= 1).
            start_date: ISO date string for the first simulated day (UTC).
            seed: Optional per-call seed override for deterministic replays.

        Returns:
            A pandas DataFrame with 96 * n_days rows and columns:
            ``timestamp``, ``demand_mw``, ``solar_mw``, ``wind_mw``,
            ``net_load_mw``, and ``price_eur_mwh``.
        """
        if n_days < 1:
            raise ValueError(f"n_days must be >= 1, got {n_days}.")

        rng = np.random.default_rng(seed) if seed is not None else self._rng
        n_steps = n_days * self.STEPS_PER_DAY
        timestamps = pd.date_range(
            start=pd.Timestamp(start_date, tz="UTC"),
            periods=n_steps,
            freq="15min",
        )
        hour_of_day = timestamps.hour.to_numpy(dtype=float) + (
            timestamps.minute.to_numpy(dtype=float) / 60.0
        )

        demand_mw = self._generate_demand(timestamps, hour_of_day, rng)
        solar_mw = self._generate_solar(n_days, hour_of_day, rng)
        wind_mw = self._generate_wind(n_steps, rng)
        shock_mw = self._generate_shocks(n_steps, rng)

        # Net load represents residual demand to be served by dispatchable generation
        net_load_mw = demand_mw - solar_mw - wind_mw
        effective_net_load_mw = net_load_mw + shock_mw

        price_eur_mwh = self._compute_merit_order_price(effective_net_load_mw, rng)

        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "demand_mw": np.round(demand_mw, 2),
                "solar_mw": np.round(solar_mw, 2),
                "wind_mw": np.round(wind_mw, 2),
                "net_load_mw": np.round(net_load_mw, 2),
                "price_eur_mwh": np.round(price_eur_mwh, 2),
            }
        )

    def _generate_demand(
        self,
        timestamps: pd.DatetimeIndex,
        hour_of_day: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate realistic diurnal electricity demand with morning and evening peaks."""
        cfg = self.config
        n_steps = len(hour_of_day)

        # Double-peaked Continental European load shape (morning ~09:30, evening ~19:30)
        morning_peak = cfg.morning_peak_mw * np.exp(-0.5 * ((hour_of_day - 9.5) / 2.2) ** 2)
        evening_peak = cfg.evening_peak_mw * np.exp(-0.5 * ((hour_of_day - 19.5) / 2.0) ** 2)
        daytime_plateau = 4_500.0 * np.exp(-0.5 * ((hour_of_day - 14.2) / 3.8) ** 2)
        overnight_trough = -2_800.0 * np.exp(-0.5 * ((hour_of_day - 3.5) / 1.9) ** 2)

        # Mild weekend load reduction (~8%)
        is_weekend = timestamps.day_of_week.to_numpy() >= 5
        weekend_factor = np.where(is_weekend, 0.92, 1.0)

        # Autocorrelated AR(1) demand noise for smooth 15-min transitions
        noise = np.zeros(n_steps, dtype=float)
        phi = 0.85
        innovation_std = cfg.demand_noise_std_mw * np.sqrt(1.0 - phi**2)
        for t in range(1, n_steps):
            noise[t] = phi * noise[t - 1] + rng.normal(0.0, innovation_std)

        demand = (
            cfg.base_demand_mw
            + morning_peak
            + daytime_plateau
            + evening_peak
            + overnight_trough
        ) * weekend_factor + noise

        return np.maximum(demand, 5_000.0)

    def _generate_solar(
        self,
        n_days: int,
        hour_of_day: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate diurnal solar PV production producing the classic Duck Curve."""
        cfg = self.config
        n_steps = len(hour_of_day)

        # Daylight window from 06:00 to 20:30 with peak around 13:15
        sunrise = 6.0
        sunset = 20.5
        daylight_phase = np.clip((hour_of_day - sunrise) / (sunset - sunrise), 0.0, 1.0)
        clear_sky = np.sin(np.pi * daylight_phase) ** 1.45

        # Day-to-day irradiance factor (sunny vs partially cloudy days)
        daily_factors = rng.uniform(0.65, 1.0, size=n_days)
        # Ensure at least one high-irradiance day on multi-day horizons to showcase Duck Curve
        if n_days >= 2:
            daily_factors[0] = max(daily_factors[0], 0.92)
        day_indices = np.arange(n_steps) // self.STEPS_PER_DAY
        day_scale = daily_factors[day_indices]

        # Fast-moving cloud transients via bounded AR(1) multiplier
        cloud_noise = np.zeros(n_steps, dtype=float)
        phi = 0.90
        for t in range(1, n_steps):
            cloud_noise[t] = phi * cloud_noise[t - 1] + rng.normal(0.0, 0.035)
        cloud_multiplier = np.clip(1.0 + cloud_noise, 0.55, 1.08)

        solar = cfg.solar_capacity_mw * clear_sky * day_scale * cloud_multiplier
        return np.clip(solar, 0.0, cfg.solar_capacity_mw)

    def _generate_wind(
        self,
        n_steps: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate wind power using a discretized Ornstein-Uhlenbeck process."""
        cfg = self.config
        dt = cfg.dt_hours

        # Exact Euler-Maruyama / discrete OU update:
        # dW_t = theta * (mu - W_t) * dt + sigma * dB_t
        exp_decay = np.exp(-cfg.wind_theta * dt)
        cond_std = cfg.wind_sigma * np.sqrt(
            (1.0 - np.exp(-2.0 * cfg.wind_theta * dt)) / (2.0 * cfg.wind_theta)
        )

        wind = np.empty(n_steps, dtype=float)
        wind[0] = np.clip(
            rng.normal(cfg.wind_mean_mw, cfg.wind_capacity_mw * 0.18),
            0.05 * cfg.wind_capacity_mw,
            0.90 * cfg.wind_capacity_mw,
        )

        for t in range(1, n_steps):
            next_val = (
                cfg.wind_mean_mw
                + (wind[t - 1] - cfg.wind_mean_mw) * exp_decay
                + rng.normal(0.0, cond_std)
            )
            # Reflective clamping keeps wind physically bounded in [0, capacity]
            if next_val < 0.0:
                next_val = -0.3 * next_val
            elif next_val > cfg.wind_capacity_mw:
                next_val = cfg.wind_capacity_mw - 0.3 * (next_val - cfg.wind_capacity_mw)
            wind[t] = np.clip(next_val, 0.0, cfg.wind_capacity_mw)

        return wind

    def _generate_shocks(
        self,
        n_steps: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate occasional unplanned thermal outage / scarcity shock windows."""
        cfg = self.config
        shocks = np.zeros(n_steps, dtype=float)
        t = 0
        while t < n_steps:
            if rng.random() < cfg.shock_prob_per_step:
                duration = max(
                    2, int(rng.integers(cfg.shock_duration_steps // 2, cfg.shock_duration_steps + 3))
                )
                magnitude = cfg.shock_magnitude_mw * rng.uniform(0.75, 1.35)
                end = min(n_steps, t + duration)
                # Smooth trapezoid-like ramp in/out
                window = np.hanning(end - t + 2)[1:-1]
                shocks[t:end] = np.maximum(shocks[t:end], magnitude * window)
                t = end + 4
            else:
                t += 1
        return shocks

    def _compute_merit_order_price(
        self,
        net_load_mw: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Map residual net load to Spot clearing price via a piecewise Merit Order stack.

        Merit Order layers (ordered by short-run marginal cost):
            1. Oversupply / Must-run floor (net_load < must_run_mw):
               Negative prices driven by renewable curtailment & inflexible baseload.
            2. Nuclear / Baseload (must_run_mw -> nuclear_capacity_mw):
               Low marginal cost (~15 to 32 EUR/MWh).
            3. Coal / Efficient CCGT mid-merit:
               Moderate upward slope (~32 to 78 EUR/MWh).
            4. Marginal Gas CCGT:
               Steeper slope (~78 to 130 EUR/MWh).
            5. OCGT / Oil Peakers:
               High marginal cost (~130 to 235 EUR/MWh).
            6. Scarcity regime (above total dispatchable stack):
               Exponential scarcity / VoLL pricing up to price cap.
        """
        cfg = self.config

        b0 = cfg.must_run_mw
        b1 = cfg.nuclear_capacity_mw
        b2 = b1 + cfg.mid_merit_capacity_mw
        b3 = b2 + cfg.gas_capacity_mw
        b4 = b3 + cfg.peaker_capacity_mw

        price = np.empty_like(net_load_mw, dtype=float)

        # Region 1: Renewable surplus / inflexible must-run -> negative prices
        m_neg = net_load_mw < b0
        if np.any(m_neg):
            surplus_ratio = (b0 - net_load_mw[m_neg]) / 7_500.0
            # Transitions smoothly from +12 EUR/MWh near b0 down into negative territory
            price[m_neg] = 12.0 - 45.0 * surplus_ratio - 18.0 * (surplus_ratio**1.6)

        # Region 2: Nuclear / Baseload marginal
        m_nuc = (net_load_mw >= b0) & (net_load_mw < b1)
        if np.any(m_nuc):
            frac = (net_load_mw[m_nuc] - b0) / max(b1 - b0, 1.0)
            price[m_nuc] = 12.0 + 20.0 * frac

        # Region 3: Mid-merit thermal (lignite / hard coal / high-efficiency CCGT)
        m_mid = (net_load_mw >= b1) & (net_load_mw < b2)
        if np.any(m_mid):
            frac = (net_load_mw[m_mid] - b1) / max(b2 - b1, 1.0)
            price[m_mid] = 32.0 + 46.0 * (frac**1.15)

        # Region 4: Marginal gas CCGT
        m_gas = (net_load_mw >= b2) & (net_load_mw < b3)
        if np.any(m_gas):
            frac = (net_load_mw[m_gas] - b2) / max(b3 - b2, 1.0)
            price[m_gas] = 78.0 + 52.0 * (frac**1.25)

        # Region 5: OCGT / Oil peakers
        m_peak = (net_load_mw >= b3) & (net_load_mw < b4)
        if np.any(m_peak):
            frac = (net_load_mw[m_peak] - b3) / max(b4 - b3, 1.0)
            price[m_peak] = 130.0 + 105.0 * (frac**1.5)

        # Region 6: Scarcity / tight reserve margin spikes
        m_scarce = net_load_mw >= b4
        if np.any(m_scarce):
            excess_gw = (net_load_mw[m_scarce] - b4) / 1_000.0
            price[m_scarce] = 235.0 + 55.0 * excess_gw + 18.0 * (excess_gw**2)

        # Add modest microstructure noise and enforce market price caps
        price = price + rng.normal(0.0, cfg.price_noise_std_eur_mwh, size=len(net_load_mw))
        return np.clip(price, cfg.negative_price_floor_eur_mwh, cfg.scarcity_price_cap_eur_mwh)
