# Differentiable Simulations

### TODO

### Implementation

- Keep eyes open for PyTorch ForwardDiff

### Analysis & Experiments

### Thoughts

Klaus said that their curling robot first calculated trajectories, then executed them and was trained on reducing the error by comparing simulation with actual trajectory.

We want to cut into that by tracing the execution and training the parameters directly in the dyn sys space.

The task is to constrain the networks to a very tight manifold.

### Temporary

#### Time-Dependent Viz

- each diffeq gets time dependence flag
- each diffeq viz function gets all the data and all the time
- gets path to directory to save animation/figure

### Code Structure

``````