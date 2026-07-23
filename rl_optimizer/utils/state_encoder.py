"""State encoding for RL agents - normalization and discretization."""

import numpy as np


class StateEncoder:
    """Encodes continuous production state into discrete or normalized representation."""

    FEATURE_KEYS = [
        ("well", "oil_rate_bopd"),
        ("well", "water_cut"),
        ("well", "gas_oil_ratio"),
        ("well", "bhp_psi"),
        ("well", "thp_psi"),
        ("equipment", "pump_efficiency"),
        ("equipment", "vibration_mm_s"),
        ("equipment", "temperature_c"),
        ("market", "oil_price_usd_bbl"),
        ("market", "gas_price_usd_mcf"),
        ("market", "opex_usd_bbl"),
        ("reservoir", "permeability_md"),
        ("reservoir", "porosity"),
        ("reservoir", "reservoir_pressure_psi"),
        ("reservoir", "reservoir_temperature_f"),
        ("reservoir", "oil_viscosity_cp"),
    ]

    STATUS_MAP = {"active": 0, "shut_in": 1, "maintenance": 2, "workover": 3}
    CONDITION_MAP = {"optimal": 0, "degraded": 1, "critical": 2, "failed": 3}

    # Approximate feature bounds for normalization
    FEATURE_MINS = np.array([
        50, 0.1, 100, 500, 100, 0.3, 0.0, 20, 40, 1.5, 10,
        1, 0.05, 1000, 60, 0.5
    ])
    FEATURE_MAXS = np.array([
        500, 0.95, 2000, 4000, 1500, 1.0, 10.0, 120, 120, 6.0, 50,
        5000, 0.35, 8000, 200, 500
    ])

    def __init__(self, num_bins=10):
        self.num_bins = num_bins

    def extract_numeric_features(self, scenario):
        """Extract numeric feature vector from a scenario dict."""
        features = []
        for group, key in self.FEATURE_KEYS:
            features.append(scenario[group][key])
        return np.array(features, dtype=np.float64)

    def normalize(self, features):
        """Min-max normalize features to [0, 1]."""
        range_vals = self.FEATURE_MAXS - self.FEATURE_MINS
        range_vals[range_vals == 0] = 1.0
        return np.clip((features - self.FEATURE_MINS) / range_vals, 0.0, 1.0)

    def discretize(self, normalized, num_bins=None):
        """Discretize normalized features into bins."""
        if num_bins is None:
            num_bins = self.num_bins
        bins = np.floor(normalized * num_bins).astype(int)
        bins = np.clip(bins, 0, num_bins - 1)
        return bins

    def encode_state(self, scenario, mode="discrete"):
        """Encode a scenario into an RL state.

        mode: 'discrete' -> tuple index for Q-table
              'normalized' -> float array for policy gradient / DQN
        """
        features = self.extract_numeric_features(scenario)
        normalized = self.normalize(features)

        if mode == "discrete":
            bins = self.discretize(normalized)
            status_val = self.STATUS_MAP.get(scenario["well"]["well_status"], 0)
            cond_val = self.CONDITION_MAP.get(scenario["equipment"]["condition"], 0)
            state_tuple = tuple(bins.tolist()) + (status_val, cond_val)
            return state_tuple
        else:
            status_val = self.STATUS_MAP.get(scenario["well"]["well_status"], 0)
            cond_val = self.CONDITION_MAP.get(scenario["equipment"]["condition"], 0)
            extra = np.array([
                status_val / 3.0, cond_val / 3.0
            ])
            return np.concatenate([normalized, extra])

    @property
    def state_size(self):
        """Size of continuous state vector."""
        return len(self.FEATURE_KEYS) + 2

    @property
    def discrete_state_size(self):
        """Total number of discrete states (approximate)."""
        return (self.num_bins ** len(self.FEATURE_KEYS)) * 4 * 4
