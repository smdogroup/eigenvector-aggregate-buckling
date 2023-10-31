#!/bin/bash
#SBATCH --job-name=srun
#SBATCH --account=gts-gkennedy9-coda20
#SBATCH -n1                      # Number of cores for the entire job
#SBATCH -N1                      # Number of nodes
#SBATCH --ntasks-per-node=1      # Number of cores per node required
#SBATCH --mem-per-cpu=16gb
#SBATCH --time=48:00:00
#SBATCH -o out/Report-%j.out

# stress minimization under frequency and volume constraint

# omega0=5.6606
# k_start=0.5
# k_delta=0.05

# # Bash does not support floating point arithmetic, use bc to post-process
# k=$(printf "%.2f" $(echo "$k_start + $k_delta * ($SLURM_ARRAY_TASK_ID - 1)" | bc))
# omega_lb=$(printf "%.5f" $(echo "$k * $omega0" | bc))



# ###################### building ######################
# srun python3 topo_opt.py --domain building --objf compliance --confs volume_ub --nx 240 --r0 4.0 --vol-frac-ub 0.3 --maxit 100 --proj --note proj=1e-6=0=1=400=0.02

# srun python3 topo_opt.py --domain building --objf frequency --confs volume_ub compliance --nx 240 --r0 6.0 --vol-frac-ub 0.3 --compliance-ub-percent 5.0 --proj --min-compliance 7.4e-6 --delta-beta 0.1 --note delta-beta=0.01,new

# srun python3 topo_opt.py --domain building --objf compliance-buckling --confs volume_ub --nx 240 --r0 6.0 --vol-frac-ub 0.3 --c0 7.4e-6 --mu-ks0 0.1 --delta-p 0.01 --weight 1.0 --delta-beta 0.1 --maxit 1000 --note delta-beta=0.01,new

# srun python3 topo_opt.py --domain building --objf frequency --confs volume_ub --nx 240 --r0 3.0 --vol-frac-ub 0.3 --delta-p 0.01 --maxit 1000 --delta-beta 0.1 --proj --note delta-p=0.01,delta-beta=0.2

# srun python3 topo_opt.py --domain building --objf frequency --confs volume_ub compliance stress --nx 240 --r0 6.0 --vol-frac-ub 0.3 --compliance-ub-percent 4.3 --delta-beta 0.1 --proj --min-compliance 7.4e-6 --delta-p 0.01 --stress-ub 4.0 

# srun python3 topo_opt.py --domain building --objf frequency --confs volume_ub compliance displacement --nx 240 --r0 6.0 --vol-frac-ub 0.3 --compliance-ub-percent 4.3  --min-compliance 7.4e-6 --proj --delta-p 0.01 --delta-beta 0.1 --dis-ub 10.0



###################### beam ######################
# srun python3 topo_opt.py --domain beam --objf compliance --confs volume_ub --nx 360 --r0 3.6 --vol-frac-ub 0.45 --maxit 500 --proj --note P-offset

# srun python3 topo_opt.py --domain beam --objf frequency --confs volume_ub compliance --nx 360 --r0 7.2 --vol-frac-ub 0.45 --min-compliance 7.7e-6 --compliance-ub-percent 2.5 --delta-p 0.01 --maxit 1000 --proj --note proj,delta-p=0.01

# srun python3 topo_opt.py --domain beam --objf compliance-buckling --confs volume_ub --nx 360 --r0 5.0 --vol-frac-ub 0.45 --c0 7.7e-6 --mu-ks0 0.115 --weight 0.0 --proj --maxit 1000 --delta-p 0.05 --delta-beta 0.05 --note delta-p=0.05,delta-beta=0.05

# srun python3 topo_opt.py --domain beam --objf frequency --confs volume_ub --nx 360 --r0 5.0 --vol-frac-ub 0.45 --delta-p 0.01 --maxit 1000 --delta-beta 0.2 --proj --note delta-p=0.01,delta-beta=0.2

# srun python3 topo_opt.py --domain beam --objf compliance-buckling --confs volume_ub stress --nx 360 --r0 5.0 --vol-frac-ub 0.45 --c0 7.7e-6 --mu-ks0 0.115 --weight 0.2 --stress-ub 200 --delta-p 0.01 --delta-beta 0.2 --proj --iter-crit-stress 200 --iter-crit 200 --maxit 1000 --note add=200

srun python3 topo_opt.py --domain beam --objf compliance-buckling --confs volume_ub displacement --nx 360 --r0 5.0 --vol-frac-ub 0.45 --c0 7.7e-6 --mu-ks0 0.115 --weight 0.2 --dis-ub 0.0 --delta-p 0.01 --delta-beta 0.2 --proj --maxit 1000 --note newnewnew





































###################### square ######################
# srun python3 topo_opt.py --domain square --objf compliance --confs volume_ub --nx 300 --r0 6.0 --vol-frac-ub 0.3 --maxit 500 --delta-p 0.01 --proj --iter-crit 0 --note P=5e-3,delta-p=0.01,iter_crit=0

# srun python3 topo_opt.py --domain square --objf frequency --confs volume_ub compliance --nx 300 --r0 3.0 --vol-frac-ub 0.4 --maxit 1000 --compliance-ub-percent 3.0 --min-compliance 8e-5 --delta-p 0.01 --iter-crit 200 --proj --note delta-p=0,iter_crit=200


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
