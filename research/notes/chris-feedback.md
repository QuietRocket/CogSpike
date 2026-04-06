Remarks:

“and applies probabilistic model checking,” -> Probabilistic model checking appears out of nowhere.

In my opinion, you should say something like: to ensure that the DTMC model satisfies the properties expected by biologists, probabilistic model checking was applied to verify that these expected properties hold (cite Elisabetta’s article). This also helps explain why we are interested in these models. The rest is fine.

⸻

“Threshold Preservation_ theorem (completeness) and an Asymptotic Silence theorem (soundness) guaranteeing that the discretization neither loses fireable configurations nor introduces spurious spikes (@sec-proofs).”

Be careful: soundness and completeness have very precise meanings.

What you call Threshold Preservation, if it refers to the transformation toward the quotient model that guarantees the same properties as the original model, would rather correspond to soundness. Also, some explanations are missing here about what exactly you mean. The other result would intuitively be “the reciprocal”, therefore completeness.

Wait -- what you say about threshold preservation might instead mean that, in the quotient model, we ensure that the computed thresholds actually correspond to the behaviour of the SNN model. In that case, this would be completeness. But this is not clear. You need to provide more details here.

You should also provide an outline of the section.

-----

SoTA section

The state of the art is quite good, but the conclusion is not correct:

“no existing tool unifies probabilistic SNN simulation and formal verification.”

because Elisabetta has already done this, for example.

Here, in the SoTA, you should explain that the main limitation of existing approaches is precisely the state explosion problem, and that this is why you are working on this issue.

In fact, the conclusion is not very clear, whereas this is exactly what the previous sentences suggest.

----

Section: model checking and temporal logic

You do not provide the definition of CTL semantics. Since you are doing model checking and will use it, it might be important to give the definition in order to express your properties rigorously.

However, the drawback is that this takes up additional space in the paper. So this is something we should discuss together tomorrow.

Actually, forget my suggestion — I think the formal semantics is not strictly necessary, since the paper is not focused on the theoretical aspects of model checking, but rather on the SNN -> PCTL connection.

Still, it may become problematic if you want to define bisimulation rigorously.

-----

“The resulting quotient is the coarsest PCTL-preserving partition, reducing the per-neuron state space from $|P_{max} - P_{min} + 1|$ values to $k + 1$ classes.”

This is quite a strong claim and no enough explanation is provided.

----

“refractory machine” ???

----

“this section quantifies that promise”

Perhaps a bit too strong? Feels like LLM-style phrasing that should be avoided.

-----

“We derive closed-form formulas for the DTMC”

Closed-form formulas?? What does that mean here?

----

Conclusion section

“requires balancing the biological fidelity of probabilistic models against the compact state spaces demanded by model checking.”

This phrasing feels somewhat empty and overly pompous, in my opinion. Probabilistic SNN models are only approximations of the brain and do not fully represent neuronal functioning.

Here, what we would like to say is that we seek the right trade-off between combinatorial explosion when performing model checking to verify properties, while remaining as faithful as possible to the SNN model ?