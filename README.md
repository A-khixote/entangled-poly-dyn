# entangled-poly-dyn

Monte Carlo polymer random walk simulations in space/time with entanglement dynamics. 

- Noise is drawn from the random uniform distribution. 
- Polymer chains are formed by random walks in space of one mer, initialised at random spatial coordinates. 
- The time evolution of chains are random walks in time (subject to the constraints of adjacent bonds) of each mer with intrinsic random probability of movement.
- Over many iterations, tends towards equal mer selection probability. 
- 1 iteration is the computational time of attempting moves = total number of mers in system (arbitrarily be referred to as the 'time unit'). 
- At termination of each time unit, system (x,t) stored in array - Dimensions: [Snapshot number, Chain number, bead number, XYZ coordinates]. 


Statistical analyses are done using a mean squared displacement routine. 2 are provided. 

- Routine 1 uses the maximum number of averages while Routine 2 uses a less computationally expensive number of averages. 
- Either way passed through plotting routine providing diffusion graphs. 
- Gradients are calculated as per the theoretically expected mean squared displacement regimes.

Time scales exponentially relative to input. Parallel computation recommended for longer simulations, using Routine 1 or averaged runs. 
