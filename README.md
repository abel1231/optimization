# Quantum Approximate Optimization Algorithm for Portfolio Optimization Problem

> This is a repository for Portfolio Optimization Problem that use Python code to
> generate their results (though it can be adapted to use other technologies).

## Software implementation

All source code used to generate the results and figures in the paper are in
the `root` directory. Specifically, `portfolio_optimization.py` is used to load and pre-process the portfolio data, 
which can be obtained in `/data` folder. Running the code in `qaoa.py` generates the results in our project report,
where the QAOA circuit is constructed using Pyqpanda framework and the parameter optimization is implemented by the optimizer from Scipy or Qiskit. 
In order to offer a benchmark for our experiment, we construct a Python Script `qaoa_qiskit.py` with the identical function using Qiskit exclusively.
The figure generation is all run inside `plot_result.py`.
Results generated by the code are saved in `/out`.


## Getting the code

You can download a copy of all the files in this repository by cloning the
[git](https://github.com/abel1231/optimization) repository:

    git clone https://github.com/abel1231/optimization.git

or [download a zip archive](https://github.com/abel1231/optimization/archive/refs/heads/master.zip).



## Dependencies

You'll need a working Python environment to run the code. All the experiments we have done are run in the `conda` virtual environment with Python 3.8.

To build and test the software, produce all results and figures, run this in the top level of the repository:

    pip install -r requirements.txt

If all goes well, all dependencies will be installed successfully.

## Examples of how to use the code

To train an 8-layer QAOA model under the setting of 4 budget, 6 assets, 2 bits for representing one asset, 
and we set the maximum number of optimization iterations to 2000, run the code

    python qaoa.py --budget 4 --num_assets 6 --g 2 --layers 8 --maxiter 2000

Running `python qaoa.py --help` for more information about options.

The results of the training will be printed out in shell, and we give some results in the `/out` folder.

Or you can use the same parameter configuration to run the QAOA model based on Qiskit

    python qaoa_qiskit.py --budget 4 --num_assets 6 --g 2 --layers 8 --maxiter 2000

Running `python qaoa_qiskit.py --help` for more information about options.

The default optimizer is `COBYLA` from Qiskit, as we observe experimentally that it 
has a more efficient optimization process and guarantees the optimization performance. Note that you can replace
the optimizer with either [Scipy Methods](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html), 
or [Qiskit Optimizers](https://qiskit.org/documentation/stubs/qiskit.algorithms.optimizers.html).

In order to generate the figures that characterize the probability distribution of the final quantum state relative to the Hamiltonian, 
we provide `plot_result.py`. For example, 

    python plot_result.py --num_diaplay 5 --offset 0.0001 --path b2_g1_ly8.out --save_type svg
