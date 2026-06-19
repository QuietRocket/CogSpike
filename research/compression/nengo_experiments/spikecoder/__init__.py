"""spikecoder — shared library for the Nengo spiking-compression experiment suite.

This package is the *accumulation layer* of the experiment ladder: every
experiment imports its source, information measures, latency calibration,
network builders, metrics and plotting from here, so the suite stays consistent
and later experiments build on earlier ones.

Submodules
----------
config       global constants (LIF time constants, calibration constant, dt, seeds)
source       the momentum-rover Markov source (byte-compatible with ../validate.py)
information  entropy / cross-entropy / KL / softmax + reference constants
latency      the latency calibration bridge (drive map, analytic + Nengo first-spike laws)
networks     reusable Nengo builders (calibrated readout, softmax, WTA, attractor, ...)
metrics      first-spike extraction, decode error, bits/symbol, energy trajectory
plotting     consistent matplotlib figures (Agg backend)

The central identity these experiments embody: a leaky integrate-and-fire readout
driven by ``R I(q) = theta / (1 - q^alpha)`` fires its first spike (from rest) at
``t*(q) = -lambda log2 q`` — the surprisal of symbol q, as a spike *time*.
"""

__version__ = "0.1.0"
