# An Inverse Optimization approach to the Amazon Challenge

This repository contains the source code to reproduce the Inverse Optimization (IO) appraoch to the [Amazon Last Mile Routing Research Challenge](https://routingchallenge.mit.edu/) from the paper [Learning in Routing Problems: an Inverse Optimization Approach](https://arxiv.org/abs/0000.00000).

## Usage

To test the code in this repository, you need to follow the steps:
1. Download the Amazon Challenge dataset [here](https://aws.amazon.com/marketplace/pp/prodview-rqkdusd3nz3mw);
2. Run the script `process_data.py`. This scrip preprocesses the datasets and slipts them per depot (this script needs to be run only once);
3. Run `main.py`. This sript can be executed in the terminal/command line or in an IDE.

The following Python packages are required:
- `numpy`;
- [`InvOpt`](https://github.com/pedroszattoni/invopt): this package is used to solve the IO problem;
- `gurobipy` or `ortools`: our IO approach to the Amazon Challenge requires a Traveling Salesperson Problem (TSP) solver. Two options are available: one using Gurobi, which solves the TSP to optimality (exact, but slow), and one using Google OR Tools, which solves the TSP approximatly (approximate, but fast).

## Citing
If you use this repository, please cite the accompanying paper:

```bibtex
@article{zattoniscroccaro2023learning,
  title={Learning in Routing Problems: an Inverse Optimization Approach},
  author={Zattoni Scroccaro, Pedro and van Beek, Piet and Mohajerin Esfahani, Peyman and Atasoy, Bilge},
  journal={https://arxiv.org/abs/0000.00000},
  year={2023}
}
```