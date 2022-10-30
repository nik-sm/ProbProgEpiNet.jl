Code for **"Probabilistic program inference in network-based epidemiological simulations"**, PLOS Computational Biology 2022, In Press., by Niklas Smedemark-Margulies*, Robin Walters*, Heiko Zimmermann, Lucas Laird, Christian van der Loo, Neela Kaushik, Rajmonda Caceres, Jan-Willem van de Meent (*Equal contribution).

In this work, we apply probabilistic programming methods for inferring the parameters of stochastic agent-based network disease simulators.
This approach allows complex models to be accurately fit to regional disease statistics, resulting in realistic models of disease spread.

# Setup

This project depends on GraphSEIR.jl (repo containing the disease simulator). GraphSEIR.jl is currently undergoing an internal release review process and will be made public soon.

Note that you must have a python3 environment active, with Pandas installed into it (for `Pandas.jl`).

To setup the project, run `bash setup.sh`.

# Contents

After setting up the project, the contents should look as follows:

```shell
.
├── config.json                                           # Topology configuration details
├── config.size_varying.json                              # Size-varying topology configuration
├── COVID-19                                              # Folder containing regional infection statistics
│   └── ...
├── Project.toml
├── ExperimentData                                        # Folder containing regional network topologies
│   └── ...
├── Manifest.toml
├── notebook                                              # Jupyter notebooks for exploration and figure generation
│   └── ...
├── README.md
├── requirements.txt                                      # Python requirements
├── scripts                                               # shell scripts for running experiments and figure generation
│   └── ...
├── setup.sh                                              # shell script for environment setup
├── size_varying                                          # Folder containing size-varying topologies
├── time_varying                                          # Folder containing time-varying topologies
├── src                                                   # Julia source files
│   ├── data.jl
│   ├── graphs.jl
│   ├── inference.jl
│   ├── plots_and_printing.jl
│   ├── run.jl
│   ├── utils.jl
│   └── ProbProgEpiNet.jl
└── validation_cycle                                     # Configuration for experiments on cycle validation
    └── ...
```

# Usage

After setting up project, use: `bash scripts/test.sh` to run a small integration test.
(Should take several minutes to run)

See [scripts](scripts) and [notebook](notebook) for files to run experiments and generate figures.

# Citation

If you use our work, please include the following citation:

```bibtex
@article{10.1371/journal.pcbi.1010591,
    title = {Probabilistic program inference in network-based epidemiological simulations},
    author = {Smedemark-Margulies, Niklas AND Walters, Robin AND Zimmermann, Heiko AND Laird, Lucas AND van der Loo, Christian AND Kaushik, Neela AND Caceres, Rajmonda AND van de Meent, Jan-Willem},
    journal = {PLOS Computational Biology},
    year = {2022},
    month = {11},
    volume = {18},
    pages = {1-40},
    number = {11},
    publisher = {Public Library of Science},
    doi = {10.1371/journal.pcbi.1010591},
    url = {https://doi.org/10.1371/journal.pcbi.1010591},
}
```