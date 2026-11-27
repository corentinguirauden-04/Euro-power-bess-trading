"""Physical Battery Energy Storage System (BESS) asset model.

Encapsulates power rating limits, State of Charge (SoC) operational bounds,
asymmetric charging/discharging efficiencies, and cycling degradation costs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BatteryConfig:
    """Physical and economic parameters of a grid-scale BESS asset.

    Attributes:
        capacity_mwh: Nominal usable energy capacity in MWh.
        max_power_mw: Maximum symmetric grid charge/discharge power rating in MW.
        efficiency_charge: One-way charging efficiency in (0, 1].
        efficiency_discharge: One-way discharging efficiency in (0, 1].
        soc_min: Minimum permitted State of Charge as a fraction in [0, 1).
        soc_max: Maximum permitted State of Charge as a fraction in (0, 1].
        degradation_cost_eur_mwh: Marginal cycling wear cost per MWh of energy
            exchanged with the grid (EUR/MWh).
        initial_soc: Initial State of Charge fraction in [soc_min, soc_max].
    """

    capacity_mwh: float = 10.0
    max_power_mw: float = 5.0
    efficiency_charge: float = 0.92
    efficiency_discharge: float = 0.92
    soc_min: float = 0.10
    soc_max: float = 0.90
    degradation_cost_eur_mwh: float = 2.0
    initial_soc: float = 0.50

    def __post_init__(self) -> None:
        if self.capacity_mwh <= 0.0:
            raise ValueError(f"capacity_mwh must be positive, got {self.capacity_mwh}.")
        if self.max_power_mw <= 0.0:
            raise ValueError(f"max_power_mw must be positive, got {self.max_power_mw}.")
        if not (0.0 < self.efficiency_charge <= 1.0):
            raise ValueError(
                f"efficiency_charge must be in (0, 1], got {self.efficiency_charge}."
            )
        if not (0.0 < self.efficiency_discharge <= 1.0):
            raise ValueError(
                f"efficiency_discharge must be in (0, 1], got {self.efficiency_discharge}."
            )
        if not (0.0 <= self.soc_min < self.soc_max <= 1.0):
            raise ValueError(
                f"Require 0 <= soc_min < soc_max <= 1, got [{self.soc_min}, {self.soc_max}]."
            )
        if self.degradation_cost_eur_mwh < 0.0:
            raise ValueError(
                f"degradation_cost_eur_mwh must be >= 0, got {self.degradation_cost_eur_mwh}."
            )
        if not (self.soc_min <= self.initial_soc <= self.soc_max):
            raise ValueError(
                f"initial_soc ({self.initial_soc}) must lie within "
                f"[{self.soc_min}, {self.soc_max}]."
            )


class BatteryAsset:
    """Physical model of a grid-connected Battery Energy Storage System (BESS).

    Power and energy sign convention (grid perspective):
        * Positive values (> 0): Discharging / exporting / selling to the grid.
        * Negative values (< 0): Charging / importing / buying from the grid.
    """

    def __init__(
        self,
        capacity_mwh: float = 10.0,
        max_power_mw: float = 5.0,
        efficiency_charge: float = 0.92,
        efficiency_discharge: float = 0.92,
        soc_min: float = 0.10,
        soc_max: float = 0.90,
        degradation_cost_eur_mwh: float = 2.0,
        initial_soc: float = 0.50,
    ) -> None:
        """Initialize the BESS physical model.

        Args:
            capacity_mwh: Nominal energy capacity in MWh.
            max_power_mw: Maximum power rating in MW.
            efficiency_charge: Charging efficiency in (0, 1].
            efficiency_discharge: Discharging efficiency in (0, 1].
            soc_min: Lower State of Charge bound in [0, 1).
            soc_max: Upper State of Charge bound in (0, 1].
            degradation_cost_eur_mwh: Cycling degradation cost in EUR/MWh.
            initial_soc: Starting State of Charge fraction.
        """
        self.config = BatteryConfig(
            capacity_mwh=capacity_mwh,
            max_power_mw=max_power_mw,
            efficiency_charge=efficiency_charge,
            efficiency_discharge=efficiency_discharge,
            soc_min=soc_min,
            soc_max=soc_max,
            degradation_cost_eur_mwh=degradation_cost_eur_mwh,
            initial_soc=initial_soc,
        )
        self._soc: float = self.config.initial_soc

    @classmethod
    def from_config(cls, config: BatteryConfig) -> "BatteryAsset":
        """Instantiate a BatteryAsset from a BatteryConfig object."""
        return cls(
            capacity_mwh=config.capacity_mwh,
            max_power_mw=config.max_power_mw,
            efficiency_charge=config.efficiency_charge,
            efficiency_discharge=config.efficiency_discharge,
            soc_min=config.soc_min,
            soc_max=config.soc_max,
            degradation_cost_eur_mwh=config.degradation_cost_eur_mwh,
            initial_soc=config.initial_soc,
        )

    @property
    def capacity_mwh(self) -> float:
        """Nominal battery capacity in MWh."""
        return self.config.capacity_mwh

    @property
    def max_power_mw(self) -> float:
        """Maximum symmetric power rating in MW."""
        return self.config.max_power_mw

    @property
    def efficiency_charge(self) -> float:
        """One-way charging efficiency."""
        return self.config.efficiency_charge

    @property
    def efficiency_discharge(self) -> float:
        """One-way discharging efficiency."""
        return self.config.efficiency_discharge

    @property
    def round_trip_efficiency(self) -> float:
        """Round-trip energy efficiency (eta_charge * eta_discharge)."""
        return self.config.efficiency_charge * self.config.efficiency_discharge

    @property
    def soc_min(self) -> float:
        """Minimum allowed State of Charge fraction."""
        return self.config.soc_min

    @property
    def soc_max(self) -> float:
        """Maximum allowed State of Charge fraction."""
        return self.config.soc_max

    @property
    def degradation_cost_eur_mwh(self) -> float:
        """Marginal degradation cost in EUR per MWh of grid throughput."""
        return self.config.degradation_cost_eur_mwh

    @property
    def soc(self) -> float:
        """Current State of Charge fraction in [soc_min, soc_max]."""
        return self._soc

    @property
    def energy_mwh(self) -> float:
        """Current stored energy inside the battery in MWh."""
        return self._soc * self.config.capacity_mwh

    def reset(self, initial_soc: Optional[float] = None) -> float:
        """Reset battery State of Charge to initial_soc (or configured default).

        Args:
            initial_soc: Optional override for starting SoC.

        Returns:
            The reset State of Charge fraction.
        """
        target_soc = self.config.initial_soc if initial_soc is None else initial_soc
        if not (self.config.soc_min <= target_soc <= self.config.soc_max):
            raise ValueError(
                f"initial_soc ({target_soc}) must lie within "
                f"[{self.config.soc_min}, {self.config.soc_max}]."
            )
        self._soc = float(target_soc)
        return self._soc

    def step(self, power_mw: float, dt: float = 0.25) -> tuple[float, float]:
        """Execute a dispatch instruction over time interval ``dt``.

        Applies inverter power clipping, State of Charge limits, and one-way
        charging/discharging efficiency losses.

        Args:
            power_mw: Requested grid power in MW. Positive values discharge
                (sell to grid); negative values charge (buy from grid).
            dt: Time step duration in hours (default 0.25 = 15 minutes).

        Returns:
            Tuple ``(energy_exchanged_mwh, new_soc)`` where:
                * ``energy_exchanged_mwh`` is the actual signed energy exchanged
                  at the grid connection point in MWh (> 0 exported, < 0 imported).
                * ``new_soc`` is the updated State of Charge fraction.
        """
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}.")

        # 1. Clip requested power to inverter limits
        clipped_power_mw = max(
            -self.config.max_power_mw, min(self.config.max_power_mw, float(power_mw))
        )

        current_energy = self._soc * self.config.capacity_mwh
        min_energy = self.config.soc_min * self.config.capacity_mwh
        max_energy = self.config.soc_max * self.config.capacity_mwh

        if clipped_power_mw > 0.0:
            # Discharging (exporting energy to grid)
            requested_grid_mwh = clipped_power_mw * dt
            requested_internal_mwh = requested_grid_mwh / self.config.efficiency_discharge
            available_internal_mwh = max(0.0, current_energy - min_energy)

            actual_internal_mwh = min(requested_internal_mwh, available_internal_mwh)
            energy_exchanged_mwh = actual_internal_mwh * self.config.efficiency_discharge
            new_energy = current_energy - actual_internal_mwh

        elif clipped_power_mw < 0.0:
            # Charging (importing energy from grid)
            requested_grid_in_mwh = (-clipped_power_mw) * dt
            requested_internal_mwh = requested_grid_in_mwh * self.config.efficiency_charge
            headroom_internal_mwh = max(0.0, max_energy - current_energy)

            actual_internal_mwh = min(requested_internal_mwh, headroom_internal_mwh)
            actual_grid_in_mwh = actual_internal_mwh / self.config.efficiency_charge
            energy_exchanged_mwh = -actual_grid_in_mwh
            new_energy = current_energy + actual_internal_mwh

        else:
            energy_exchanged_mwh = 0.0
            new_energy = current_energy

        # Guard against tiny floating-point excursions beyond [soc_min, soc_max]
        new_soc = new_energy / self.config.capacity_mwh
        self._soc = max(self.config.soc_min, min(self.config.soc_max, new_soc))

        return energy_exchanged_mwh, self._soc

    def compute_degradation_cost(self, energy_exchanged_mwh: float) -> float:
        """Compute cycling degradation cost in EUR for a given grid energy exchange."""
        return abs(energy_exchanged_mwh) * self.config.degradation_cost_eur_mwh
