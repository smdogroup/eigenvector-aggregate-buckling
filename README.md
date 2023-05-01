# Eigenvector Derivatives

## Description
This repository contains the code used to generate the results in the paper "Topology Optimization using an Eigenvector Aggregate" by Bao Li, Yicong Fu and Graeme J. Kennedy. 

## Contents
- `tube_opt.py` is the main file that contains the code for the 2D tube optimization example. There are four problems in the tube example as follows:
  - `plot_E` is used to plot the E matrix shown in Figure 2 in the paper.
  - `accuracy_analysis` is used to plot the accuracy of the approximate eigenvector derivatives shown in section 8.1.1 in the paper.
  - `optimization_eigenvalue` is a function that runs the optimization for the frequency minimization with a volume constraint problem.
  - `optimization_displacement` is a function that runs the optimization for the frequency minimization with a volume constraint and displacement constraint problem.
  - `optimization_stress` is a function that runs the optimization for the frequency minimization with a volume constraint and stress constraint problem.
- `topo_opt.py` is the main file that contains the code for the topology optimization for the beam and square plate examples. The function `parse_cmd_args` takes the following arguments:
  - For the problem parameters:
    - `domain` is the domain of the problem, which can be `beam` or `square`.
    - `objf` is the objective function, which is set to be `frequency` by default.
    - `confs` is the constraint function, which is set to `volume` by default.
    - `vol-frac-ub` is the upper bound of the volume fraction constraint.
    - `stress-ub` is the upper bound of the stress constraint.
    - `dis-ub` is the upper bound of the displacement constraint.
  - For the topology optimization parameters:
    - `nx` is the number of elements along the x direction.
    - `filer` is the density filter, which can be `spectral` or `helmholtz`, with corresponding filter radius `r0` which is set to be `2.1` by default.
    - `ptype-K` is the material penalization method for the stiffness matrix K, which can be `SIMP` or `RAMP`, with corresponding penalization parameters `p` and `q`. It is set to `SIMP` with `p=3` by default.
    - `ptype-M` is the material penalization method for the mass matrix M, which can be `MSIMP`, `RAMP`, or `LINEAR`.
    - `optimizer` is the optimizer used to solve the optimization problem, which can be `pmma` or `tr`, where `pmma` is the MMA method and `tr` is the trust region method.
    - `maxit` is the maximum number of iterations.
- `simple_example.py` contains the code for the simple example in paper section 3.1.
- `demo.py` contains the code for a demo to show exact eigenvector derivatives and approximate eigenvector derivatives in paper sections 4 and 5 respectively.
- `output` folder contains the output files for the simple example, tube, beam, and square plate examples.
- `other` folder contains the code for generating the figures in the paper, helper functions, and bash scripts used to run the code on Georgia Tech's PACE cluster.
- `run.sh` is the bash script to run the code.

## Usage
The code is written in Python 3. To run the code, you need to install the following packages:
- [ParOpt](https://github.com/smdogroup/paropt) is a parallel gradient-based optimizer. The dependencies of ParOpt are listed [MPI](https://www.open-mpi.org/), [BLAS](http://www.netlib.org/blas/), [LAPACK](http://www.netlib.org/lapack/), [mpi4py](https://mpi4py.readthedocs.io/en/stable/), [Cython](https://cython.org/), [numpy](https://numpy.org/), [scipy](https://www.scipy.org/)
- [scienceplots](https://github.com/garrettj403/SciencePlots) which is a plotting library
- [matplotlib](https://matplotlib.org/), [numpy](https://numpy.org/), [scipy](https://www.scipy.org/), [mpmath](http://mpmath.org/)
```
./run.sh
```
The code will run the simple example, tube, beam, and square plate examples. The output files will be saved in the `output` folder.

## Citation
If you find this code useful in your research, please consider citing:
```
To be added
```

## Contact
If you have any questions, please contact [Bao Li](libao@gatech.edu). 

