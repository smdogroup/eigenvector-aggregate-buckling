# Eigenvector Aggregate in Buckling Topology Optimization

## Description
Eigenvector aggregate is a method to impose eigenvector constraints used in topology optimization. The method is described in the following paper:
```
B. Li, and G. Kennedy. Topology Optimization for Buckling using an Eigenvector Aggregate. Computer Methods in Applied Mechanics and Engineering, 2024 (Submitted).
```

## Contents
- `demo.py` contains the code for a demo to show exact eigenvector derivatives and approximate eigenvector derivatives described in paper section 3.
- `topo_opt.py` is the main file for the topology optimization for the beam, column, and square plate.
- `settings.py` contains the default parameters and input arguments for the topology optimization.
- `domain.py` contains the beam, column, and square plate domain definitions.
- `jobs.sbatch` is the SLURM script used to run the code on Georgia Tech's PACE cluster.
- `run.sh` is the bash script used to run the code.
- `output` folder used to save the output figures and log files.
- `src` folder contains the source code rewritten using the Kokkos library for parallel computation. The code is written in C++ and can be compiled using the CMakeLists.txt file.
- `other` folder contains the code for generating the figures in the paper, helper functions, 


## Usage
The code is written in Python 3. To run the code, you need to install the following packages:
- [ParOpt](https://github.com/smdogroup/paropt) (version [2.0.2](https://github.com/smdogroup/paropt/tree/v2.0.2) is recommended) is a parallel gradient-based optimizer that integrates the MMA method and the trust region method. The dependencies of ParOpt are listed [MPI](https://www.open-mpi.org/), [BLAS](http://www.netlib.org/blas/), [LAPACK](http://www.netlib.org/lapack/), [mpi4py](https://mpi4py.readthedocs.io/en/stable/), [Cython](https://cython.org/), [numpy](https://numpy.org/), [scipy](https://www.scipy.org/)
- [scienceplots](https://github.com/garrettj403/SciencePlots), [matplotlib](https://matplotlib.org/), [numpy](https://numpy.org/), [scipy](https://www.scipy.org/), [mpmath](http://mpmath.org/), [icecream](https://github.com/gruns/icecream)

To run the code, simply run the bash script `run.sh`:
```
./run.sh
```
by modifying the bash script `run.sh`, the user can run the code with different parameters. The output figures and log files will be saved in the `output` folder.

## Citation
If you find this code useful in your research, please consider citing:
```
To be added
```

## Contact
If you have any questions, please contact [Bao Li](libao@gatech.edu), or [Graeme J. Kennedy](graeme.kennedy@aerospace.gatech.edu).

