# Polymer Dynamics and Entanglement
## Description

Monte Carlo polymer random walk simulations in space/time with entanglement dynamics. Used to explore time evolution of entangled/unentangled chains (see 'Context').

## Theory Considerations

Key assumptions are hardcoded as below.

- Noise is drawn from the random uniform distribution. 
- Polymer chains are formed by random walks in space of one mer, initialised at random spatial coordinates. 
- The time evolution of chains are random walks in time (subject to the constraints of adjacent bonds) of each mer with intrinsic random probability of movement.
- Over many iterations, tends towards equal mer selection probability. 
- 1 iteration is the computational time of attempting moves = total number of mers in system (arbitrarily be referred to as the 'time unit'). 
- At termination of each time unit, system (x,t) stored in array - Dimensions: [Snapshot number, Chain number, Bead number, XYZ coordinates]. 
- Toroidal Boundary Conditions implemented for pseudo-lattice structure.

Factors such as inter-chain interactions are ignored, only entanglement is considered and investigated. Can potentially be reused for further/alternative dynamics investigations. 

## Analysis

Statistical analyses are done using a mean squared displacement routine. 2 are provided. 

- Routine 1 uses the maximum number of averages while Routine 2 uses a less computationally expensive number of averages. 
- Either way passed through plotting routine providing diffusion graphs. 
- Gradients are calculated as per the theoretically expected mean squared displacement regimes.

Later data points have inherently more uncertainty so averaging required for meaningful analysis.

## Computation

Time scales exponentially relative to input. Parallel computation recommended for longer simulations, using Routine 1 or averaged runs. Time evolution recommended to be translated to C++ for better computational runtime, though the 'Numba' JIT compiler is employed for this work.

## Context

The motivation for this work is based on the papers below. Specifically, verifying the emergence of a unique regime under an entanglement constraint. 
1. https://pubs.acs.org/doi/abs/10.1021/acs.macromol.9b02428
2. https://aip.scitation.org/doi/10.1063/1.458541
