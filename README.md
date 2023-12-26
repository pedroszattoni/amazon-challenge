# An Inverse Optimization approach to the Amazon Challenge

This repository contains the source code to reproduce the Inverse Optimization (IO) approach to the [Amazon Last Mile Routing Research Challenge](https://routingchallenge.mit.edu/) from the paper [Inverse Optimization for Routing Problems](https://arxiv.org/abs/2307.07357).

## Usage

To test the code in this repository, you need to follow the steps:
1. Download the Amazon Challenge datasets [here](https://aws.amazon.com/marketplace/pp/prodview-rqkdusd3nz3mw);
2. Run `process_data.py`. This script preprocesses the datasets and splits them per depot (needs to be run only once);
3. Run `main.py`. This script can be executed in the terminal/command line or an IDE.

The following Python packages are required:
- `numpy`;
- [`invopt`](https://github.com/pedroszattoni/invopt): this package is used to solve the IO problem;
- `gurobipy` or `ortools`: our IO approach to the Amazon Challenge requires a Traveling Salesperson Problem (TSP) solver. Two options are available: one using Gurobi, which solves the TSP to optimality (slow), and one using Google OR Tools, which solves the TSP approximately (fast).

## Citing
If you use this repository, please cite the accompanying paper:

```bibtex
@article{zattoniscroccaro2023inverse,
  title={Inverse Optimization for Routing Problems},
  author={Zattoni Scroccaro, Pedro and van Beek, Piet and Mohajerin Esfahani, Peyman and Atasoy, Bilge},
  journal={https://arxiv.org/abs/2307.07357},
  year={2023}
}
```