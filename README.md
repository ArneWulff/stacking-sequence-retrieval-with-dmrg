# Research Code Accompanying the Publication: "Quantum Computing and Tensor Networks for Laminate Design: A Novel Approach to Stacking Sequence Retrieval"

**Authors:** Arne Wulff¹, Boyang Chen¹, Matthew Steinberg², Yinglu Tang¹, Matthias Möller², Sebastian Feld²
¹*Faculty of Aerospace Engineering, Delft University of Technology, The Netherlands*
²*Faculty of Electrical Engineering, Mathematics and Computer Science, Delft University of Technology, The Netherlands*

This repository contains the code to reproduce the results of our paper: URL

Please use the following citation:

```
@article{Wulff2024,
  title = {Quantum Computing and Tensor Networks for Laminate Design: A New Approach to Stacking Sequence Retrieval},
  author = {Wulff, Arne and Chen, Boyang and Steinberg, Matthew and Tang,Yinglu and M{\"o}ller, Matthias and Feld, Sebastian},
  journal = {arXiv},
  year = {2024},
  url = "..."
}
```

**Abstract:**
> As with many tasks in engineering, structural design frequently involves navigating complex and computationally expensive problems. A prime example is the weight optimization of laminated composite materials, which to this day remains a formidable task, due to an exponentially large configuration space and non-linear constraints. The rapidly developing field of quantum computation may offer novel approaches for addressing these intricate problems. However, before applying any quantum algorithm to a given problem, it must be translated into a form that is compatible with the underlying operations on a quantum computer.
Our work specifically targets stacking sequence retrieval with lamination parameters, which is typically the second phase in a common bi-level optimization procedure for minimizing the weight of composite structures. To adapt stacking sequence retrieval for quantum computational methods, we map the possible stacking sequences onto a quantum state space. We further derive a linear operator, the Hamiltonian, within this state space that encapsulates the loss function inherent to the stacking sequence retrieval problem. Additionally, we demonstrate the incorporation of manufacturing constraints on stacking sequences as penalty terms in the Hamiltonian. This quantum representation is suitable for a variety of classical and quantum algorithms for finding the ground state of a quantum Hamiltonian. For a practical demonstration, we chose a classical tensor network algorithm, the DMRG algorithm, to numerically validate our approach. For this purpose, we derived a matrix product operator representation of the loss function Hamiltonian and the penalty terms. Numerical trials with this algorithm successfully yielded approximate solutions, while exhibiting a tradeoff between accuracy and runtime. Although this work primarily concentrates on quantum computation, the application of tensor network algorithms presents a novel quantum-inspired approach for stacking sequence retrieval.

**Contact:** Boyang Chen, Arne Wulff

https://www.tudelft.nl/lr/qaims

---

## Disclaimer: 
This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 

## License Notice:
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0) 
This work is licensed under a [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) license.

Some parts of this code are modifications of stucts and functions in [ITensors.jl](https://github.com/ITensor/ITensors.jl) which is published by The Simons Foundation [(License)](https://github.com/ITensor/ITensors.jl/blob/main/LICENSE). These structs and functions are as follows: 
- The structs and functions in `dmrg_costum.jl` are modifications of ITensors `SweepNext`, `sweepnext(N::Int; ncenter::Int=2)`, `Base.iterate(sn::SweepNext, state=(0, 1))` and `dmrg(PH, psi0::MPS, sweeps::Sweeps; kwargs...)`.
- The function `dmrg_experiment_one_try` is a modification of ITensors `dmrg(PH, psi0::MPS, sweeps::Sweeps; kwargs...)`.

## Contents of this repository:

Lamination parameters:

- `laminationparameters.jl`: Function to calculate the lamination parameters from a stacking sequence

Generate a large set of lamination parameters and calculate weights:

- `generate_random_stacks.jl`: Functions to generate random stacking sequences that disobey a disorientation constraint
- `generate_random_lp.jl`: Functions to generate random lamination parameters using the functions from `generate_random_stacks.jl`, calculate weights for the generated points using a kernel density estimation, and generate the according HDF5 files
- `run_generate_random_lp.jl`: Using the functions in the previous two file to generate a particular weighted set of lamination parameters

Draw a small sample from the large set of lamination parameters according to the generated weights

- `sample_lp_from_file.jl`: Functions to draw random samples from files generated with `generate_random_lp.jl` according to the calculated weights, and store the sample in an HDF5 file
- `run_sample_lp_from_file.jl`: Using the functions in the previous file to generate a small set of lamination parameters. 

Running DMRG:

- `mpo.jl`: Implementation of the matrix product operators for the stacking sequence retrieval problem as ITensor `MPO` objects
- `dmrg_custom.jl`: Modification of ITensors `dmrg` implementation to also support arbitrary sequences of sweeping directions
- `dmrg_experiment.jl`: Functions for performing the experiments as described in the paper
- `run_dmrg_experiment.jl`: Using the function to run experiments

License and Read-Me

- `LICENSE`: Apache 2.0 license
- `README.md`: This readme file


