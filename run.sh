#!/bin/zsh
# create a directory for the results if it does not exist
if [ ! -d output ]; then
  mkdir output
fi

################# simple example #################
python3 simple_example.py

###################### demo ######################
python3 demo.py

###################### tube ######################
python3 tube_opt.py

###################### beam ######################
python3 topo_opt.py --domain beam --objf frequency --confs volume --nx 200 --r0 1.0 --vol-frac-ub 0.5 --maxit 100 

###################### square ######################
python3 topo_opt.py --domain square --objf frequency --confs volume --nx 100 --r0 1.0 --vol-frac-ub 0.4 --maxit 100