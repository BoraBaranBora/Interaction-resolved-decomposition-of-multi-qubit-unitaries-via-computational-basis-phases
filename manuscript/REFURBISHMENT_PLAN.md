# Manuscript refurbishment

Working title:

**Interaction-Resolved Quantum Optimal Control: Single-Pulse Synthesis of Isolated Tripartite Interactions in an NV-Center Spin Register**

## Core editorial change

The revised paper should no longer present the Walsh/Pauli projection of diagonal computational-basis phases as a new class of “phase invariants.” The parity-weighted coefficients are standard Walsh-Hadamard / Pauli-Z expansion coordinates. Their role in this paper is as **interaction-resolved control coordinates** used to formulate support-selective quantum-optimal-control objectives.

## Revised claim hierarchy

1. Direct synthesis of an isolated higher-body interaction is a selective quantum-control problem: the desired tripartite term should be generated while unwanted pairwise terms are suppressed, whereas local terms may be left free and corrected separately.
2. In a diagonalizing local frame, computational-basis phases can be projected onto the Pauli-Z basis (equivalently Walsh transformed) to obtain interaction coordinates indexed by their qubit support.
3. These coordinates give a simple interaction-resolved QOC objective that fixes the desired support sectors and leaves irrelevant sectors unconstrained.
4. The method is demonstrated as a realistic NV-center spin-register case study, synthesizing isolated ZZZ and XZZ tripartite interactions in single shaped microwave pulses.

## Terminology

Prefer **interaction coordinates**, **Pauli interaction coordinates**, or **support-resolved interaction phases** after the initial definition.

Avoid presenting “support-selective phase invariants” as a new mathematical object. Avoid unnecessary equivalence-class formalism unless it is needed directly for the control objective.

## Structural target

1. Introduction
2. Interaction-resolved quantum optimal control
   - Interaction coordinates in a diagonalizing frame
   - Support-selective control objective
3. NV-center spin-register model
   - System Hamiltonian
   - Pulse parametrization
4. Single-pulse synthesis of isolated tripartite interactions
   - ZZZ
   - XZZ
5. Discussion and conclusion
6. Appendices

The general mathematical construction should support the NV result rather than dominate the paper. The realistic NV application and the isolated tripartite single-pulse result are the central deliverables.
