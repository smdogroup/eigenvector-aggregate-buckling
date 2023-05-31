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

# python3 topo_opt.py --domain beam --objf frequency --confs volume_ub --nx 200 --maxit 100  --vol-frac-ub 0.5 --r0 1.0

# python3 topo_opt.py --domain beam --objf frequency --confs volume_ub displacement --nx 200 --maxit 100  --vol-frac-ub 0.5 --dis-ub 0.3 --r0 1.0

# python3 topo_opt.py --optimizer pmma --domain beam --problem buckling --objf frequency --confs volume_ub displacement stress --nx 200 --maxit 100 --vol-frac-ub 0.5 --dis-ub 0.3 --stress-ub 1.00 --r0 2.1 --ks-rho 1e+6 --grad-check

# ###################### square ######################
# echo "Running square problem, check log file in output/square"
# python3 topo_opt.py --domain square --objf frequency --confs volume_ub --nx 100 --r0 1.0 --vol-frac-ub 0.4 --maxit 100 --r0 1.2
# echo "Finished, press enter to continue..."
# read ans

# python3 topo_opt.py --domain square --objf frequency --confs volume_ub stress --nx 100 --vol-frac-ub 0.4 --maxit 100 --stress-ub 6.0e+12 --r0 1.0

# python3 topo_opt.py --domain square --objf frequency --confs volume_ub displacement --nx 100 --maxit 100 --vol-frac-ub 0.4 --dis-ub 0.080 --r0 1.0

# python3 topo_opt.py --domain square --objf frequency --confs volume_ub displacement stress --nx 100 --maxit 100 --vol-frac-ub 0.4 --dis-ub 0.034 --stress-ub 3.6e+12 --r0 1.0

# ###################### beam ######################
# python3 topo_opt.py --domain beam --problem buckling --objf compliance --confs volume_ub --nx 100 --r0 1.0 --vol-frac-ub 0.5 --maxit 100

# python3 topo_opt.py --domain beam --problem buckling --objf compliance --confs volume_ub frequency --nx 100 --r0 1.0 --vol-frac-ub 0.5 --omega-lb 1000 --maxit 500


# ###################### building ######################
# python3 topo_opt.py --domain building --problem buckling --objf compliance --confs volume_ub --nx 200 --r0 1.0 --vol-frac-ub 0.25 --maxit 100 --note speedup

# python3 topo_opt.py --domain building --problem buckling --objf compliance --confs volume_ub frequency --nx 50 --r0 1.0 --vol-frac-ub 0.25 --omega-lb 20 --maxit 500

python3 topo_opt.py --domain building --problem buckling --objf compliance --confs volume_ub frequency stress --nx 50 --r0 1.0 --vol-frac-ub 0.25 --omega-lb 20 --maxit 500 --stress-ub 5.0e+6 --note ks_rho=160

# ###################### leg ######################
# python3 topo_opt.py --domain leg --problem buckling --objf compliance --confs volume_ub --nx 50 --r0 1.1 --vol-frac-ub 0.2 --maxit 100

# python3 topo_opt.py --domain leg --problem buckling --objf compliance --confs volume_ub frequency --nx 50 --r0 1.0 --vol-frac-ub 0.2 --omega-lb 600 --maxit 500 
