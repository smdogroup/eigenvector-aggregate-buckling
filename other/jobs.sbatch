#!/bin/bash
#SBATCH --job-name=srun
#SBATCH --account=gts-gkennedy9-coda20
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1
#SBATCH --mem-per-cpu=6gb
#SBATCH --time=12:00:00
#SBATCH -o out/Report-%j.out

# stress minimization under frequency and volume constraint

# omega0=5.6606
# k_start=0.5
# k_delta=0.05

# # Bash does not support floating point arithmetic, use bc to post-process
# k=$(printf "%.2f" $(echo "$k_start + $k_delta * ($SLURM_ARRAY_TASK_ID - 1)" | bc))
# omega_lb=$(printf "%.5f" $(echo "$k * $omega0" | bc))

# srun python3 topo_opt.py  --prefix omega_k=$k --maxit 400  --objf stress --stress-scale 1e-3 --confs volume frequency --omega-lb $omega_lb --lb 1e-6 --ptype-K ramp --ptype-M msimp

# srun python3 topo_opt.py --optimizer pmma --objf compliance --confs stress volume --nx 50 --vol-frac 0.5 --stress-ub 3004 --domain mbbbeam --movelim 0.2 --r0 2.0


###################### square ######################
# srun python3 topo_opt.py --optimizer pmma --objf volume --confs stress frequency --nx 200 --omega-lb 5 --stress-ub 8000 --maxit 300 --r0 4.0

# srun python3 topo_opt.py --optimizer pmma --objf stress --confs frequency volume_lb volume_ub --nx 200 --omega-lb 2.0 --maxit 200 --vol-frac-ub 0.4 --vol-frac-lb 0.2

###################### mbbbeam ######################
# srun python3 topo_opt.py --optimizer pmma --domain mbbbeam --objf frequency --confs volume_ub --nx 200 --vol-frac-ub 0.5003 --maxit 200  --ptype-K simp

# srun python3 topo_opt.py --optimizer pmma --domain mbbbeam --objf frequency --confs volume_ub displacement --nx 200 --vol-frac-ub 0.5003 --maxit 200  --ptype-K simp --dis-ub 0.1

###################### lbracket ######################
# srun python3 topo_opt.py --optimizer pmma --domain lbracket --objf compliance --confs stress volume_ub volume_lb --nx 50 --maxit 100 --stress-ub 150 --vol-frac-ub 0.5 --vol-frac-lb 0.2 --m0-block-frac 0.5


srun python3 beam.py