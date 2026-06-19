"""Global configuration constants for the spiking-compression suite.

One place for the LIF time constants, the latency calibration constant, the
simulation timestep, default seeds and the rheobase ceiling, so a single change
re-runs the whole suite consistently.

The values are chosen so the calibration exponent
``alpha = lambda / (tau_rc * ln 2) = 1 / ln 2 ~= 1.4427`` reproduces the paper's
worked drive table (R I/theta = 1.58, 2.94, 7.09, 14.0, ... at q = 0.5, 0.75,
0.9, 0.95). See ``unified_spiking_compression.typ`` Theorem 1 and the drive table.

Units note. The paper works in dimensionless time (tau = lambda = theta = 1).
Nengo uses a physical membrane time constant ``tau_rc`` (seconds) and normalizes
the threshold to 1. The map is: physical time = paper-time x tau_rc, so the
time-per-bit constant is ``lambda = tau_rc`` (s/bit) and the calibration exponent
``alpha = lambda/(tau_rc ln2) = 1/ln2`` is dimensionless and independent of tau_rc.
"""

import numpy as np

# --- LIF neuron (Nengo units: firing threshold normalized to 1) ---
TAU_RC = 0.02        # membrane time constant tau (s) -- the paper's tau
TAU_REF = 0.002      # refractory period (s). NOTE: does NOT delay the FIRST spike
                     # from rest (it is post-spike dead time); the per-window-reset
                     # latency code is therefore refractory-immune (see e01).
THETA = 1.0          # firing threshold (Nengo normalizes to 1)

# --- latency calibration ---
LAMBDA = TAU_RC                          # s/bit  (=> alpha = 1/ln2 ~= 1.4427)
ALPHA = LAMBDA / (TAU_RC * np.log(2.0))  # calibration exponent = 1/ln2

# --- simulation ---
DT = 1e-3            # default Nengo timestep (s) for throughput/learning experiments
DT_FINE = 1e-4       # fine grid for latency-resolution-critical experiments

# --- representable-probability band ---
# A biophysical neuron supplies only a few-fold rheobase current; the calibration
# drive R I/theta = 1/(1 - q^alpha) blows up well inside the working range, pinning
# a maximum representable probability q_max (see latency.q_max_for_ceiling).
RHEOBASE_CEILING = 10.0   # max drive R I/theta a "real" neuron supplies (few-fold)
Q_CLIP_LO = 1e-4          # clamp q away from 0 (drive -> rheobase, noise-dominated)
Q_CLIP_HI = 0.999         # clamp q away from 1 (drive -> infinity)

# --- reproducibility ---
SEED = 7             # matches validate.py / learn_validate.py stream seed
