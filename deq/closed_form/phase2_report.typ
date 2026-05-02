#set page(paper: "a4", margin: 2cm)
#set text(size: 10pt)
#set par(justify: true)

#align(center)[
  #text(size: 14pt, weight: "bold")[Phase 2 report -- Linear-response H(omega) (H2 part A)]
  #v(0.2em)
  Verdict: *PASS*
]

= Hypothesis (H2 part A)

For the negative-loop (E-I) archetype with default FCS weights
($w_("XA") = 11$, $w_("AI") = 11$, $w_("IA") = -11$, $p_("thin") = 0.7$),
the single-pole low-pass approximation of the linear-response transfer
function

$ H_i(omega) = (partial Phi_i / partial mu_i) / (1 + i omega tau_m) $

at the Siegert fixed point, plugged into the closed-loop matrix
$M(omega) = I - H(omega) J$, satisfies the $omega = 0$ self-consistency:
the time-domain Jacobian
$A = (1\/tau_m)(- I + "diag"(g) J)$ has the same spectrum as
$("diag"(g) J - I) \/ tau_m$ obtained from $det M(0) = 0$.

= Operating point

Calibration locked in from Phase 1:

- $alpha = 0.2500$, $beta = 0.004292$,
  $tau_m = 2.3504$, $tau_("ref") = 0.3611$.

Topology and effective Jacobian:

$ J_("FCS") = mat(0, -11; 11, 0) , quad J_("eff") = alpha J_("FCS") = mat(0.0000, -2.7500; 2.7500, 0.0000) $

Self-consistent Siegert fixed point:

- $nu^* = (0.3520, 0.1793)$
- $mu^* = (1.4319, 0.9680)$
- $sigma^* = (0.4307, 0.3442)$

DC gains:

- $partial Phi_A \/ partial mu_A = 0.3163$
- $partial Phi_I \/ partial mu_I = 0.3598$

= Time-domain Jacobian eigenvalues

Spectrum of $A = (1 \/ tau_m) (- I + "diag"(g) J_("eff"))$:

-0.4255 + 0.3947i, -0.4255 + -0.3947i

Maximum real part: $-0.4255$ (negative => stable focus
or node; positive => unstable)$\.$
Maximum |imaginary| part: $0.3947$
(non-zero => oscillatory dynamics).

= Self-consistency

Closed-loop characteristic polynomial $det M(omega) = det(I - H(omega) J)$
at $omega = 0$ has roots $1 - tau_m lambda$ in correspondence with the
Jacobian spectrum. Numerical residual (max difference between sorted
spectra):

#text(size: 14pt)[$|"residual"| = 0.00e+00$]

Gate $<= 10^(-3)$: *PASS*.

Imaginary-axis crossings of $det M(omega)$ in $[0.01, 30]$ rad/time-unit:
none in tested range.

= Bode plot of closed-loop response

#figure(image("results/phase2/bode.pdf", width: 75%),
  caption: [Bode magnitude (top) and phase (bottom) of the closed-loop
  transfer $G(omega) = (I - H(omega) J)^(-1) H(omega)$, entry A -> A.
  The single-pole low-pass shape is consistent with a stable
  configuration where the negative loop adds bandwidth-limited damping.])

#figure(image("results/phase2/characteristic.pdf", width: 95%),
  caption: [Left: real and imaginary parts of $det M(omega)$. Right:
  Nyquist locus of $det M(omega)$. The locus does not encircle the origin,
  consistent with the Jacobian eigenvalues having negative real parts.])

= Overall verdict

*PASS*.

The Phase 2 self-consistency check confirms the closed-loop machinery
correctly reduces to the time-domain Jacobian at $omega = 0$. Phase 3
(H2 part B) extends to non-zero $omega$: comparing predicted spectral peaks
against LI&F-population FFT measurements.
