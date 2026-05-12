Spiking neural networks have the phenomenon of the spike reset which introduces non-linearity. This non-linearity makes it impossible to use classical control theory tools to establish closed form formulae through differential equations. So we thought about working in terms of populations and Wilson–Cowan. But for example for contralateral inhibition, we're talking about populations with an abstract topology. How can we operate with differential equations and closed form formulae at a more finite level (discrete tangible topologies)? Specifically we're working with Leaky Integrate and Fire.

---

The trick is to recognize that the reset nonlinearity doesn't have to be handled at the membrane-potential level — you can push it into a smooth gain function and then keep the *topology* explicit at the population level. Three frameworks let you do this with closed-form objects, and they're complementary rather than competing.

**1. Diffusion approximation + Siegert: the static piece**

For a LIF neuron driven by many small inputs, you replace the synaptic shot noise by an Ornstein–Uhlenbeck input with mean $\mu$ and variance $\sigma^2$. The stationary firing rate is then the Siegert formula:

$$\nu = \Phi(\mu, \sigma) = \left( \tau_{\text{ref}} + \tau_m \sqrt{\pi} \int_{(V_r-\mu)/\sigma}^{(V_{\text{th}}-\mu)/\sigma} e^{u^2}(1+\text{erf}(u))\, du \right)^{-1}$$

This is genuinely closed-form (in erf), and it absorbs the reset into the lower integration limit. You now have an analytic f–I curve per population. The reset's hard nonlinearity has become an algebraic nonlinearity in $\Phi$ — much friendlier.

**2. Linear response / transfer functions: the dynamic piece**

Brunel & Hakim, Lindner, Richardson, and Ostojic worked out the linear response of a LIF population around a stationary state. You linearize the Fokker–Planck operator and get a transfer function $H_i(\omega)$ for each population — closed form in terms of parabolic cylinder functions or, in useful limits, simple low-pass-with-resonance forms. Once you have $H_i(\omega)$ per population, your contralateral-inhibition motif is just a block diagram:

$$\delta\nu_i(\omega) = H_i(\omega)\, \delta\mu_i(\omega), \qquad \delta\mu_i(\omega) = \sum_j J_{ij}\, \delta\nu_j(\omega) + \delta\mu_i^{\text{ext}}(\omega)$$

where $J_{ij}$ is your *finite, tangible* connectivity matrix. Now classical control theory works directly: poles of $(I - H J)^{-1}$ give stability, Nyquist gives oscillation onset, Bode plots tell you bandwidth of the loop. This is exactly the regime where Brunel got his asynchronous/SR/SI/AI phase diagram for E-I networks.

**3. Mesoscopic equations for genuinely finite populations**

If your "populations" are small enough that finite-size fluctuations matter (a few dozen to a few thousand neurons each — typical for column-like motifs), look at Schwalger, Deger & Gerstner, *PLoS Comp Bio* 2017, "Towards a theory of cortical columns." They derive stochastic integral equations for the population activity $A_k(t)$ of each finite group $k$, including refractoriness and reset, in the form

$$A_k(t) = \int_{-\infty}^{t} P_k(t|\hat{t})\, A_k(\hat{t})\, d\hat{t} + \text{finite-size noise}$$

with explicit kernels. This is the cleanest "Wilson–Cowan but with the spike physics actually in it, on a discrete topology" formulation I know. It reduces to Wilson–Cowan in the large-$N$, slow-dynamics limit and to the population density equations in the deterministic limit.

**Adjacent tools worth knowing**

- **Spike Response Model** (Gerstner): rewrite LIF so the reset is a kernel $\eta(t-\hat{t})$ added to a linearly filtered input. The dynamics become formally linear plus a threshold; population-activity equations follow as Volterra integrals.
- **Quasi-renewal / refractory density** (Naud, Gerstner): keeps non-stationary corrections that pure rate models miss.
- **Augustin, Ladenbauer, Baumann, Obermayer** low-dimensional reductions of the Fokker–Planck — useful when you want a 2D or 3D ODE per population that you can stick into a network and analyze with phase-plane tools.

**Practical recipe for your contralateral inhibition case**

Write each population as a node. Compute $\Phi_i$ and $H_i(\omega)$ analytically from LIF parameters. Build the connectivity matrix $J$ from the actual topology (sign and weight of each contralateral projection). Find fixed points by solving $\nu_i^* = \Phi_i(\mu_i(\nu^*))$ — this is a small algebraic system, usually 2–4 equations for the kinds of motifs you're describing. Then linearize and the rest is linear control theory on a small block diagram.

The thing you give up versus full Wilson–Cowan abstraction is mostly elegance; what you gain is that every parameter in the closed-form expressions is something you can actually measure or set in a LIF simulation, and the topology stays concrete throughout.

Key references to pull: Brunel 2000 (J. Comp. Neurosci.) for the E-I rate-equation framework with Siegert, Brunel & Hakim 1999 for the transfer function, Richardson 2007/2008 for cleaner derivations of $H(\omega)$, and Schwalger et al. 2017 for finite-$N$ mesoscopic dynamics.