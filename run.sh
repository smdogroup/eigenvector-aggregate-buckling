#!/bin/zsh
# create a directory for the results if it does not exist
if [ ! -d output ]; then
  mkdir output
fi

# ################# simple example #################
python3 simple_example.py

# ###################### demo ######################
python3 demo.py

###################### tube ######################
python3 tube_opt.py

# ###################### beam ######################
python3 topo_opt.py --domain beam --objf frequency --confs volume_ub --nx 200 --r0 1.0 --vol-frac-ub 0.5 --maxit 100 --r0 1.0

# python3 topo_opt.py --domain beam --objf frequency --confs volume_ub displacement --nx 200 --maxit 100  --vol-frac-ub 0.5 --dis-ub 0.3 --r0 1.0

# python3 topo_opt.py --domain beam --objf frequency --confs volume_ub displacement stress --nx 200 --maxit 100 --vol-frac-ub 0.5 --dis-ub 0.3 --stress-ub 2.5e+12 --r0 1.0

# ###################### square ######################
python3 topo_opt.py --domain square --objf frequency --confs volume_ub --nx 100 --r0 1.0 --vol-frac-ub 0.4 --maxit 100 --r0 1.0

# python3 topo_opt.py --domain square --objf frequency --confs volume_ub stress --nx 100 --vol-frac-ub 0.4 --maxit 100 --stress-ub 6.0e+12 --r0 1.0

# python3 topo_opt.py --domain square --objf frequency --confs volume_ub displacement --nx 100 --maxit 100 --vol-frac-ub 0.4 --dis-ub 0.080 --r0 1.0

# python3 topo_opt.py --domain square --objf frequency --confs volume_ub displacement stress --nx 100 --maxit 100 --vol-frac-ub 0.4 --dis-ub 0.034 --stress-ub 3.6e+12 --r0 1.0