# Manuscript refurbishment

Working title:

**Interaction-Resolved Quantum Optimal Control: Single-Pulse Synthesis of Isolated Tripartite Interactions in an NV-Center Spin Register**

## Editing policy

The existing manuscript is already the basis of the revised paper. The refurbishment should remain **surgical**: preserve the section order, equations, NV Hamiltonian, pulse parametrization, numerical results, figures, and appendices, and change only wording where the present claim hierarchy or terminology is misleading.

## Core conceptual distinction

The paper introduces **interaction-resolved quantum optimal control as a control scheme**. It does **not** claim to introduce the Walsh-Hadamard coordinates themselves.

In a diagonalizing frame, the computational-basis phase generator is projected onto the Pauli-$Z$ basis. The resulting coefficients are equivalently the Walsh-Hadamard transform of the phase map. We use these standard coefficients as **interaction coordinates** inside a QOC objective so that desired interaction supports can be targeted, unwanted supports suppressed, and irrelevant/local coordinates left unconstrained.

## Claim hierarchy

1. Interaction-resolved QOC is the methodological contribution.
2. Pauli/Walsh interaction coordinates are the mathematical representation used by the scheme, not a newly introduced invariant construction.
3. The same formulation is applied directly to a realistically parametrized NV-center spin register.
4. The physical case study demonstrates single-pulse isolated tripartite $ZZZ$ and $XZZ$ interactions with the numerical results already reported in the manuscript.

## Terminology

Prefer **interaction coordinates** or **interaction-resolved coordinates** after the initial definition.

Do not present “support-selective phase invariants” as a new mathematical object. Internal LaTeX labels may retain their old names where changing them would add unnecessary churn.

## Scope of changes

- title;
- abstract framing;
- Introduction claim language;
- prose surrounding the Pauli/Walsh projection;
- QOC/result captions and discussion terminology;
- conclusion and author-contribution wording;
- small grammar fixes directly encountered in those passages.

No wholesale restructuring or new technical claims are intended.
