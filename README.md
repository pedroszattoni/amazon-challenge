# An Inverse Optimization approach to the Amazon Challenge

This repository contains the source code to reproduce the Inverse Optimization (IO) approach to the [Amazon Last Mile Routing Research Challenge](https://routingchallenge.mit.edu/) from the paper [Inverse Optimization for Routing Problems](https://pubsonline.informs.org/doi/abs/10.1287/trsc.2023.0241).

## Usage

To test the code in this repository, you need to follow the steps:
1. Download the Amazon Challenge datasets [here](https://aws.amazon.com/marketplace/pp/prodview-rqkdusd3nz3mw).
2. Run `process_data.py`. This script preprocesses the datasets and splits them per depot (needs to be run only once).
3. Run `main.py`. This script can be executed in the terminal/command line or IDE. **NOTE**: before using the `main.py` file, the variables `path_to_input_data` and `path_to_output_data` need to be correctly defined as the path to the processed data and the path to the location where the results will be saved.

The following Python packages are required:
- `numpy`.
- [`invopt`](https://github.com/pedroszattoni/invopt): this package is used to solve the IO problem.
- `gurobipy`, `ortools` or `LKH-3`: our IO approach to the Amazon Challenge requires a Traveling Salesperson Problem (TSP) solver. Three options are available: a Gurobi-based solver, which solves the TSP to optimality, but is possibly slow for large TSPs; or LKH-3 or Google OR-Tools, which solves the TSP approximately, but are possibly faster for large TSPs. **NOTE**: to use the LKH-3 solver, the variable `solver_path` in the `utils.py` file needs to be correctly defined.

## Citing
If you use this repository, please cite the accompanying paper:

```bibtex
@article{zattoniscroccaro2024inverse,
  title={Inverse Optimization for Routing Problems},
  author={Zattoni Scroccaro, Pedro and van Beek, Piet and Mohajerin Esfahani, Peyman and Atasoy, Bilge},
  journal={Transportation Science},
  year={2024}
}
```
