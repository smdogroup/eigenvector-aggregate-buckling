#!/bin/bash
#SBATCH --job-name=srun
#SBATCH --account=gts-gkennedy9-coda20
#SBATCH -n1                      # Number of cores for the entire job
#SBATCH -N1                      # Number of nodes
#SBATCH --ntasks-per-node=1      # Number of cores per node required
#SBATCH --mem-per-cpu=16gb
#SBATCH --time=36:00:00
#SBATCH -o out/Report-%j.out

# stress minimization under frequency and volume constraint

# omega0=5.6606
# k_start=0.5
# k_delta=0.05

# # Bash does not support floating point arithmetic, use bc to post-process
# k=$(printf "%.2f" $(echo "$k_start + $k_delta * ($SLURM_ARRAY_TASK_ID - 1)" | bc))
# omega_lb=$(printf "%.5f" $(echo "$k * $omega0" | bc))


# srun python3 topo_opt.py --domain building --objf compliance --confs volume_ub --nx 240 --r0 4.0 --vol-frac-ub 0.3 --maxit 100 --proj --note proj=1e-6=0=1=400=0.02

srun python3 topo_opt.py --domain building --objf frequency --confs volume_ub compliance --nx 240 --r0 4.0 --vol-frac-ub 0.3 --compliance-ub-percent 2.5 --maxit 1000 --proj --frequency-scale 50.0 --note proj=1e-6=0=1=0.05freq_scale=50N=12

# srun python3 topo_opt.py --domain building --objf compliance --confs volume_ub frequency stress displacement --nx 50 --r0 1.5 --vol-frac-ub 0.25 --omega-lb 20 --maxit 1000 --stress-ub 5.0e+6  --dis-ub 0.1 --note ks_rho=160

# ###################### building ######################
# srun python3 topo_opt.py --domain building --problem buckling --objf compliance --confs volume_ub --nx 100 --r0 1.0 --vol-frac-ub 0.25 --maxit 300 --compliance-scale 1e7 --note scale=1e7

# srun python3 topo_opt.py --domain building --problem buckling --objf compliance --confs volume_ub frequency --nx 240 --r0 2.1 --vol-frac-ub 0.25 --omega-lb 15 --maxit 500

# srun python3 topo_opt.py --domain building --problem buckling --objf compliance --confs volume_ub frequency stress --nx 240 --r0 2.1 --vol-frac-ub 0.25 --omega-lb 30 --maxit 500 --stress-ub 8.0e+6 --note paropt_2_0_2,ks_rho_stress=10



# srun python3 topo_opt.py --domain building --problem buckling --objf frequency --confs volume_ub --nx 240 --r0 2.1 --vol-frac-ub 0.25 --maxit 300

# srun python3 topo_opt.py --domain building --problem buckling --objf frequency --confs volume_ub compliance --nx 100 --r0 1.0 --vol-frac-ub 0.25 --compliance-ub-percent 1.9 --maxit 1000 

# --frequency-scale 3.5 --note frequency_scale=3.5

# srun python3 topo_opt.py --domain building --problem buckling --objf frequency --confs volume_ub compliance stress --nx 100 --r0 1.0 --vol-frac-ub 0.25 --compliance-ub-percent 1.5 --stress-ub 23 --maxit 1000



# srun python3 topo_opt.py  --prefix omega_k=$k --maxit 400  --objf stress --stress-scale 1e-3 --confs volume frequency --omega-lb $omega_lb --lb 1e-6 --ptype-K ramp --ptype-M msimp

# srun python3 topo_opt.py --optimizer pmma --objf compliance --confs stress volume --nx 50 --vol-frac 0.5 --stress-ub 3004 --domain beam --movelim 0.2 --r0 2.0

###################### square ######################
# srun python3 topo_opt.py --optimizer pmma --objf frequency --confs --nx 400 --maxit 1000 --r0 2.1 --note l=0.1_delta_2_movelim_0.05

# --filter helmholtz
       
# srun python3 topo_opt.py --optimizer pmma --objf frequency --confs volume_ub --ptype-K ramp --filter helmholtz --nx 400 --vol-frac-ub 0.4 --maxit 1000 --r0 0.0 --note l=test_test

# srun python3 topo_opt.py --optimizer pmma --objf frequency --confs volume_ub stress --nx 400 --vol-frac-ub 0.4 --maxit 1000 --stress-ub 6.0e+12 --note no_delta

# srun python3 topo_opt.py --optimizer pmma --objf frequency --confs volume_ub displacement --nx 400 --maxit 1000 --vol-frac-ub 0.4 --dis-ub 0.080 --note no_delta

# srun python3 topo_opt.py --optimizer pmma --objf frequency --confs volume_ub displacement stress --nx 400 --maxit 1000 --vol-frac-ub 0.4 --dis-ub 0.034 --stress-ub 3.6e+12

###################### beam ######################
# srun python3 topo_opt.py --optimizer pmma --domain beam --objf frequency --confs volume_ub --nx 1000 --maxit 1000 --vol-frac-ub 0.5

# srun python3 topo_opt.py --optimizer pmma --domain beam --objf frequency --confs volume_ub displacement --nx 1000 --maxit 1000  --vol-frac-ub 0.5 --dis-ub 0.3

# srun python3 topo_opt.py --optimizer pmma --domain beam --objf frequency --confs volume_ub stress --nx 1000 --maxit 1000 --vol-frac-ub 0.5 --stress-ub 0.5e+12

# srun python3 topo_opt.py --optimizer pmma --domain beam --objf frequency --confs volume_ub displacement stress --nx 200 --maxit 100 --vol-frac-ub 0.5 --dis-ub 0.3 --stress-ub 2.5e+12


# srun python3 beam.py

# srun python3 topo_opt.py --domain beam --prob buckling --objf compliance --confs volume_ub frequency --nx 200 --r0 1.0 --vol-frac-ub 0.5 --omega-lb 1--maxit 500


# srun python3 topo_opt.py --domain building --prob buckling --objf compliance --confs volume_ub frequency --nx 500 --r0 1.0 --vol-frac-ub 0.25 --omega-lb 300 --maxit 500 --note f=100


# srun python3 topo_opt.py --domain building --prob buckling --objf compliance --confs volume_ub --nx 100 --r0 1.0 --vol-frac-ub 0.25 --maxit 200 --note f=100


# ###################### beam ######################
# python3 topo_opt.py --domain beam --problem buckling --objf compliance --confs volume_ub --nx 100 --r0 1.0 --vol-frac-ub 0.5 --maxit 100

# python3 topo_opt.py --domain beam --problem buckling --objf compliance --confs volume_ub frequency --nx 100 --r0 1.0 --vol-frac-ub 0.5 --omega-lb 1000 --maxit 500




# ###################### leg ######################
# srun python3 topo_opt.py --domain leg --problem buckling --objf compliance --confs volume_ub --nx 50 --r0 1.1 --vol-frac-ub 0.2 --maxit 100

# srun python3 topo_opt.py --domain leg --problem buckling --objf compliance --confs volume_ub frequency --nx 50 --r0 1.0 --vol-frac-ub 0.2 --omega-lb 600 --maxit 500 
