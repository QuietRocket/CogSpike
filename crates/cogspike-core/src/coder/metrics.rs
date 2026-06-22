//! Decode / bits / energy metrics, byte-faithful to the pure (non-Nengo-probe)
//! functions of `spikecoder/metrics.py`.

/// First-spike-takes-all decode: the argmin-latency neuron is the winner.
/// Returns `(decoded, winner_latency)`; an all-`inf` window decodes to `None`.
#[must_use]
pub fn decode_first_spike(latencies: &[f64]) -> (Option<usize>, f64) {
    let mut winner: Option<usize> = None;
    let mut best = f64::INFINITY;
    for (j, &lat) in latencies.iter().enumerate() {
        if lat < best {
            best = lat;
            winner = Some(j);
        }
    }
    (winner, best)
}

/// Mean per-symbol first-spike time converted to bits (`latency / lambda`),
/// averaged over the finite windows (`inf` if none are finite).
#[must_use]
pub fn mean_bits_per_symbol(win_latencies: &[f64], lam: f64) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for &lat in win_latencies.iter().filter(|l| l.is_finite()) {
        sum += lat;
        count += 1;
    }
    if count == 0 {
        f64::INFINITY
    } else {
        sum / count as f64 / lam
    }
}

/// Fraction of windows whose decoded symbol differs from the emitted symbol.
#[must_use]
pub fn decode_error_rate(decoded: &[Option<usize>], emitted: &[usize]) -> f64 {
    let total = decoded.len().min(emitted.len());
    if total == 0 {
        return f64::NAN;
    }
    let mismatches = decoded
        .iter()
        .zip(emitted)
        .filter(|&(d, &e)| *d != Some(e))
        .count();
    mismatches as f64 / total as f64
}

/// Energy `E = sum_i pi_i sum_j P_ij(-log2 q_ij)` per snapshot (`q` clipped to `[1e-12, 1]`).
#[must_use]
pub fn energy_trajectory(q_snapshots: &[Vec<Vec<f64>>], pi: &[f64], p: &[Vec<f64>]) -> Vec<f64> {
    q_snapshots
        .iter()
        .map(|q| {
            pi.iter()
                .zip(p)
                .zip(q)
                .map(|((&pii, prow), qrow)| {
                    pii * prow
                        .iter()
                        .zip(qrow)
                        .map(|(&pij, &qij)| -pij * qij.clamp(1e-12, 1.0).log2())
                        .sum::<f64>()
                })
                .sum::<f64>()
        })
        .collect()
}

/// First-spike latency gap `lambda*(log2 q_top - log2 q_2nd)` between the two most
/// probable symbols. A small margin means a hard winner-take-all decode.
#[must_use]
pub fn latency_gap_margin(q: &[f64], lam: f64) -> f64 {
    let mut sorted = q.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    match (sorted.first(), sorted.get(1)) {
        (Some(&top), Some(&second)) if second > 0.0 => lam * (top.log2() - second.log2()),
        _ => f64::INFINITY,
    }
}
