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
# srun python3 topo_opt.py --kokkos --domain building --objf compliance --confs volume_ub --nx 240 --r0 4.0 --vol-frac-ub 0.3 --maxit 100 --proj --note proj=1e-6=0=1=400=0.02

# srun python3 topo_opt.py --kokkos --domain building --objf frequency --confs volume_ub compliance --nx 240 --r0 6.0 --vol-frac-ub 0.3 --compliance-ub-percent 4.3 --proj --min-compliance 7.4e-6 --delta-beta 0.1 --rho0-G 1e-1 --rho0-K 1e-12 --note k_G=1e-1,k_K=1e-12

# srun python3 topo_opt.py --kokkos --domain building --objf compliance-buckling --confs volume_ub --nx 240 --r0 6.0 --vol-frac-ub 0.3 --c0 7.4e-6 --mu-ks0 0.08 --delta-p 0.01 --delta-beta 0.1 --weight 0.2 --maxit 1000 --note new

# srun python3 topo_opt.py --kokkos --domain building --objf frequency --confs volume_ub --nx 240 --r0 3.0 --vol-frac-ub 0.3 --delta-p 0.01 --maxit 1000 --delta-beta 0.1 --proj --note delta-p=0.01,delta-beta=0.2

# srun python3 topo_opt.py --kokkos --domain building --objf frequency --confs volume_ub compliance stress --nx 240 --r0 6.0 --vol-frac-ub 0.3 --compliance-ub-percent 4.3 --delta-beta 0.1 --proj --min-compliance 7.4e-6 --delta-p 0.01 --stress-ub 4.0 

# srun python3 topo_opt.py --kokkos --domain building --objf frequency --confs volume_ub compliance displacement --nx 240 --r0 6.0 --vol-frac-ub 0.3 --compliance-ub-percent 4.3  --min-compliance 7.4e-6 --proj --delta-p 0.01 --delta-beta 0.1 --dis-ub 0.0 --mode 1 --maxit 1000 --note N=6 

# srun python3 topo_opt.py --kokkos --domain building --objf frequency --confs volume_ub compliance displacement --nx 240 --r0 6.0 --vol-frac-ub 0.3 --compliance-ub-percent 4.3  --min-compliance 7.4e-6 --proj --delta-p 0.01 --delta-beta 0.1 --dis-ub 40.0 --mode 3

##### tracking
# srun python3 topo_opt.py --kokkos --domain building --objf frequency --confs volume_ub compliance displacement --nx 240 --r0 6.0 --vol-frac-ub 0.3 --compliance-ub-percent 4.3  --min-compliance 7.4e-6 --proj --delta-p 0.01 --delta-beta 0.1 --dis-ub 10.0 --tracking --iter-crit-dis 101 --maxit 1100 --note tracking,newnewnew --mode 3

##### no tracking
# srun python3 topo_opt.py --kokkos --domain building --objf frequency --confs volume_ub compliance displacement --nx 240 --r0 6.0 --vol-frac-ub 0.3 --compliance-ub-percent 4.3  --min-compliance 7.4e-6 --proj --delta-p 0.01 --delta-beta 0.1 --dis-ub 10.0 --note new --mode 4

##### compliance-buckling no tracking
# srun python3 topo_opt.py --kokkos --domain building --objf compliance-buckling --confs volume_ub displacement --nx 240 --r0 6.0 --vol-frac-ub 0.3 --c0 7.4e-6 --mu-ks0 0.1 --delta-p 0.01 --delta-beta 0.1 --weight 0.3 --maxit 1000 --dis-ub 4.0

##### compliance-buckling tracking
# srun python3 topo_opt.py --kokkos --domain building --objf compliance-buckling --confs volume_ub displacement --nx 240 --r0 6.0 --vol-frac-ub 0.3 --c0 7.4e-6 --mu-ks0 0.1 --delta-p 0.01 --delta-beta 0.1 --weight 0.2 --maxit 1000 --dis-ub 10.0 --tracking --iter-crit-dis 101 --maxit 1100 --note tracking --mode 3


##### compliance-buckling weight0
# srun python3 topo_opt.py --kokkos --domain building --objf compliance-buckling --confs volume_ub --nx 240 --r0 6.0 --vol-frac-ub 0.3 --c0 7.4e-6 --mu-ks0 0.1 --delta-p 0.01 --delta-beta 0.1 --weight 1.0 --maxit 24000 --iter-crit-w 2000 --note wit=2000

##### compliance-buckling weight
# srun python3 topo_opt.py --kokkos --domain building --objf compliance-buckling --confs volume_ub displacement --nx 240 --r0 6.0 --vol-frac-ub 0.3 --c0 7.4e-6 --mu-ks0 0.1 --delta-p 0.01 --delta-beta 0.1 --weight 1.0 --maxit 15000 --dis-ub 7.0 --iter-crit-w 1000 --note wit=1000

##### compliance-buckling displacement
# srun python3 topo_opt.py --kokkos --domain building --objf compliance-buckling --confs volume_ub displacement --nx 240 --r0 6.0 --vol-frac-ub 0.3 --c0 7.4e-6 --mu-ks0 0.1 --delta-p 0.01 --delta-beta 0.1 --weight 0.8 --maxit 8000 --dis-ub 7.0 --iter-crit-d 800 --note dit=800




###################### beam ######################
# srun python3 topo_opt.py --kokkos --domain beam --objf compliance --confs volume_ub --nx 360 --r0 3.6 --vol-frac-ub 0.45 --maxit 500 --proj --note P-offset

# srun python3 topo_opt.py --kokkos --domain beam --objf frequency --confs volume_ub compliance --nx 360 --r0 7.2 --vol-frac-ub 0.45 --min-compliance 7.7e-6 --compliance-ub-percent 2.5 --delta-p 0.01 --maxit 1000 --proj --note proj,delta-p=0.01

# srun python3 topo_opt.py --kokkos --domain beam --objf compliance-buckling --confs volume_ub --nx 360 --r0 5.0 --vol-frac-ub 0.45 --c0 7.7e-6 --mu-ks0 0.115 --weight 0.2 --proj --maxit 1000 --delta-p 0.01 --delta-beta 0.3 --note delta-beta=0.3

# srun python3 topo_opt.py --kokkos --domain beam --objf frequency --confs volume_ub --nx 360 --r0 5.0 --vol-frac-ub 0.45 --delta-p 0.01 --maxit 1000 --delta-beta 0.2 --proj --note delta-p=0.01,delta-beta=0.2

# srun python3 topo_opt.py --kokkos --domain beam --objf compliance-buckling --confs volume_ub displacement --nx 360 --r0 5.0 --vol-frac-ub 0.45 --c0 7.7e-6 --mu-ks0 0.115 --weight 0.2 --dis-ub 0.0 --delta-p 0.01 --delta-beta 0.2 --proj --mode 1

srun python3 topo_opt.py --kokkos --domain beam --objf compliance-buckling --confs volume_ub displacement_frac --nx 360 --r0 5.0 --vol-frac-ub 0.45 --c0 7.7e-6 --mu-ks0 0.115 --weight 0.2 --delta-p 0.01 --delta-beta 0.2 --proj --mode 1 --dis-ub-frac 0.0

##### compliance-buckling weight0
# srun python3 topo_opt.py --kokkos --domain beam --objf compliance-buckling --confs volume_ub --nx 360 --r0 5.0 --vol-frac-ub 0.45 --c0 7.7e-6 --mu-ks0 0.115 --delta-p 0.01 --delta-beta 0.2 --weight 1.0 --proj --maxit 4000 --iter-crit-w 300 --note wit=300


##### tracking
# srun python3 topo_opt.py --kokkos --domain beam --objf compliance-buckling --confs volume_ub displacement --nx 360 --r0 5.0 --vol-frac-ub 0.45 --c0 7.7e-6 --mu-ks0 0.115 --weight 0.2 --dis-ub 6.0 --delta-p 0.01 --delta-beta 0.2 --proj --tracking --iter-crit-dis 101 --maxit 1100 --note tracking --mode 3


##### no tracking
# srun python3 topo_opt.py --kokkos --domain beam --objf compliance-buckling --confs volume_ub displacement --nx 360 --r0 5.0 --vol-frac-ub 0.45 --c0 7.7e-6 --mu-ks0 0.115 --weight 0.6 --dis-ub 10.0 --delta-p 0.01 --delta-beta 0.2 --proj --mode 3




###################### square ######################

# srun python3 topo_opt.py --kokkos --domain square --objf compliance-buckling --confs volume_ub --nx 300 --r0 6.0 --vol-frac-ub 0.25 --proj --maxit 1000  --c0 1e-5 --mu-ks0 0.1 --weight 1.0 --delta-p 0.01 --delta-beta 0.1 --proj --note f=0.25e-3

##### tracking
# srun python3 topo_opt.py --kokkos --domain square --objf compliance-buckling --confs volume_ub displacement --nx 300 --r0 6.0 --vol-frac-ub 0.25 --c0 1e-5 --mu-ks0 0.1 --weight 0.4 --dis-ub 9.0 --delta-p 0.01 --delta-beta 0.1 --proj --tracking --note tracknew --ks-rho-buckling 3000 --mode 4

##### no tracking
# srun python3 topo_opt.py --kokkos --domain square --objf compliance-buckling --confs volume_ub displacement --nx 300 --r0 6.0 --vol-frac-ub 0.25 --c0 1e-5 --mu-ks0 0.1 --weight 0.6 --dis-ub 8.0 --delta-p 0.01 --delta-beta 0.1 --proj --ks-rho-buckling 3000 --mode 4

