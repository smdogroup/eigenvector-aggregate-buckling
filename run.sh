#!/bin/zsh
# create a directory for the results if it does not exist
if [ ! -d output ]; then
  mkdir output
fi





# # ################# simple example #################
# echo "Running simple example, please wait..."
# python3 simple_example.py
# echo "Finished, check the results in output/simple_example"
# echo "Press enter to continue..."
# read ans

# # ###################### demo ######################
# echo "Running demo, please wait..."
# python3 demo.py
# echo "Finished, press enter to continue..."
# read ans

# ###################### tube ######################
# echo "Running tube problem, please wait..."
# python3 tube_opt.py
# echo "Finished, check the results in output/tube"
# echo "Press enter to continue..."
# read ans


# echo "Finished, press enter to continue..."
# read ans

# python3 topo_opt.py --domain beam --objf frequency --confs volume_ub --nx 200 --proj --maxit 100  --vol-frac-ub 0.5 --r0 1.0

# python3 topo_opt.py --domain beam --objf frequency --confs volume_ub displacement --nx 200 --maxit 100  --proj --vol-frac-ub 0.5 --dis-ub 0.3 --r0 1.0 --check-gradient

# python3.11 topo_opt.py --optimizer pmma --domain beam --objf frequency --confs volume_ub displacement --nx 200 --maxit 1000 --ptype-K simp  --vol-frac-ub 0.5 --dis-ub 0.3 --mode 3 --r0 2.1 --proj --check-gradient --kokkos

# python3 topo_opt.py --optimizer pmma --domain beam --objf frequency --confs volume_ub displacement stress --nx 50 --maxit 200 --vol-frac-ub 0.5 --dis-ub 0.3 --stress-ub 1.00 --r0 2.1 --ks-rho 1e+6 --proj --kokkos --check-gradient

# python3 topo_opt.py --domain square --objf compliance --confs volume_ub --nx 100 --r0 2.0 --vol-frac-ub 0.3 --proj --maxit 500 --note P=1e-3,new_square

# python3 topo_opt.py --domain square --objf frequency --confs volume_ub compliance --nx 100 --r0 2.0 --vol-frac-ub 0.3 --proj --maxit 500 --compliance-ub-percent 3 --min-compliance 2.48e-5 --note P=1e-3,new_square

# python3 topo_opt.py --domain square --objf frequency --confs volume_ub compliance --nx 300 --r0 3.0 --vol-frac-ub 0.3 --maxit 1000 --compliance-ub-percent 3.0 --min-compliance 1.1e-4 --proj --note P=2e-3,min-new2 --sigma-scale 2.0

# ###################### square ######################
# echo "Running square problem, check log file in output/square"
# python3 topo_opt.py --domain square --objf frequency --confs volume_ub --nx 100 --r0 1.0 --vol-frac-ub 0.4 --maxit 100 --r0 1.2
# echo "Finished, press enter to continue..."
# read ans

# python3 topo_opt.py --domain square --objf frequency --confs volume_ub stress --nx 100 --vol-frac-ub 0.4 --proj --maxit 100 --stress-ub 6.0e+12 --r0 1.0

# python3 topo_opt.py --domain square --objf frequency --confs volume_ub displacement --nx 100 --maxit 100 --vol-frac-ub 0.4 --dis-ub 0.080 --r0 1.0 --proj

# python3 topo_opt.py --domain square --objf frequency --confs volume_ub displacement stress --nx 50 --maxit 100 --vol-frac-ub 0.4 --dis-ub 0.034 --stress-ub 3.6e+12 --r0 1.0 --proj --kokkos --check-gradient 

# ###################### beam ######################
# python3 topo_opt.py --domain beam --objf compliance --confs volume_ub --nx 100 --r0 1.0 --vol-frac-ub 0.15 --maxit 100

python3 topo_opt.py --domain beam --objf frequency --confs volume_ub compliance --nx 300 --r0 3.0 --vol-frac-ub 0.5 --min-compliance 8e-6 --compliance-ub-percent 3.0 --maxit 1000 --note new

# python3 topo_opt.py --domain beam --objf compliance --confs volume_ub frequency --nx 100 --r0 1.0 --vol-frac-ub 0.5 --omega-lb 1000 --maxit 1000








# ###################### tree ######################
# python3 topo_opt.py --domain building --objf compliance --confs volume_ub --nx 100 --r0 2.0 --vol-frac-ub 0.5 --maxit 100 

# python3 topo_opt.py --domain building --objf frequency --confs volume_ub --nx 100 --r0 2.0 --vol-frac-ub 0.5 --maxit 1000 

# python3 topo_opt.py --domain building --objf compliance --confs volume_ub frequency --nx 100 --r0 2.0 --vol-frac-ub 0.5 --BLF-lb 10 --maxit 1000 ,BLF-lb=10

# python3 topo_opt.py --domain building --objf frequency --confs volume_ub compliance --nx 100 --r0 2.0 --vol-frac-ub 0.5 --compliance-ub-percent 1.5 

# smpirun
# ###################### building ######################
# python3 topo_opt.py --domain building --objf compliance --confs volume_ub --nx 100 --r0 2.1 --vol-frac-ub 0.3 --proj --maxit 100 --note new_building

# python3 topo_opt.py --domain building --objf frequency --confs volume_ub --nx 100 --r0 2.1 --vol-frac-ub 0.3 --proj --maxit 1000 

# python3 topo_opt.py --domain building --objf compliance --confs volume_ub frequency --nx 120 --r0 1.0 --vol-frac-ub 0.3 --BLF-lb 8 --proj --maxit 1000

# python3 topo_opt.py --domain bui>lding --objf frequency --confs volume_ub compliance --nx 240 --r0 4.0 --vol-frac-ub 0.3 --compliance-ub-percent 2.5 --proj --note speedup

# python3 topo_opt.py --domain building --objf frequency --confs volume_ub compliance displacement --nx 240 --r0 4.0 --vol-frac-ub 0.3 --compliance-ub-percent 2.5 --dis-ub 0.1 --proj --note freq_scale=1e6.0 --check-gradient

# python3 topo_opt.py --domain building --objf frequency --confs volume_ub compliance stress --nx 32 --r0 3.0 --vol-frac-ub 0.3 --compliance-ub-percent 2.5 --stress-ub 25 --proj --check-gradient

# python3 topo_opt.py --domain building --objf frequency --confs volume_ub compliance stress displacement --nx 32 --r0 2.0 --vol-frac-ub 0.3  --compliance-ub-percent 2.5 --stress-ub 50.0 --dis-ub 0.1 --frequency-scale 10.0 --proj --maxit 1000 --note freq_scale=1e6.0 --iter-crit 10

# python3 topo_opt.py --domain building --objf compliance --confs volume_ub frequency stress --nx 50 --r0 1.0 --vol-frac-ub 0.25 --omega-lb 20 --proj --maxit 1000 --stress-ub 5.0e+6 --note ks_rho=160 --kokkos

# python3 topo_opt.py --domain building --objf compliance --confs volume_ub frequency stress displacement --nx 50 --r0 1.0 --vol-frac-ub 0.25 --omega-lb 20 --maxit 1000 --stress-ub 5.0e+6  --dis-ub 0.1 --proj --note ks_rho=160 --check-gradient

# ###################### leg ######################
# python3 topo_opt.py --domain leg --objf compliance --confs volume_ub --nx 50 --r0 1.1 --vol-frac-ub 0.2 --maxit 100

# python3 topo_opt.py --domain leg --objf compliance --confs volume_ub frequency --nx 50 --r0 1.0 --vol-frac-ub 0.2 --omega-lb 600 --maxit 1000 


# ###################### rhombus ######################
# python3 topo_opt.py --domain rhombus --objf compliance --confs volume_ub --nx 100 --r0 1.0 --vol-frac-ub 0.3 --maxit 100 --note f=1,v=0.3,f=1000

# python3 topo_opt.py --domain rhombus --objf compliance --confs volume_ub frequency  --nx 50 --r0 1.0 --vol-frac-ub 0.3 --omega-lb 200 --maxit 100 --note f=1,v=0.3,f=1000
