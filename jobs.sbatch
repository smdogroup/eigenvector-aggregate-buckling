#!/bin/bash
#SBATCH --job-name=p5-mix
#SBATCH --account=gts-gkennedy9-coda20
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1
#SBATCH --mem-per-cpu=6gb
#SBATCH --time=8:00:00
#SBATCH --output=p5_%a.out
#SBATCH --array=1-11

# stress minimization under frequency and volume constraint

omega0=5.6606
k_start=0.5
k_delta=0.05

# Bash does not support floating point arithmetic, use bc to post-process
k=$(printf "%.2f" $(echo "$k_start + $k_delta * ($SLURM_ARRAY_TASK_ID - 1)" | bc))
omega_lb=$(printf "%.5f" $(echo "$k * $omega0" | bc))

srun python topo_opt.py  --prefix omega_k=$k --maxit 400  --objf stress --stress-scale 1e-3 \
                         --confs volume frequency --omega-lb $omega_lb --lb 1e-6 \
                         --ptype-K ramp --ptype-M msimp
