#!/bin/zsh
# create a directory for the results if it does not exist
if [ ! -d output ]; then
  mkdir output
fi

# alias python3=python3.11


###################### beam ######################
# python3 topo_opt.py --domain beam --objf compliance-buckling --confs volume_ub --nx 360 --r0 5.0 --vol-frac-ub 0.45 --c0 7.7e-6 --mu-ks0 0.115 --weight 0.2 --proj --maxit 1000 --delta-p 0.01 --delta-beta 0.2

# python3 topo_opt.py --domain beam --objf compliance-buckling --confs volume_ub displacement --nx 360 --r0 5.0 --vol-frac-ub 0.45 --c0 7.7e-6 --mu-ks0 0.115 --weight 0.2 --dis-ub 0.0 --delta-p 0.01 --delta-beta 0.2 --proj --mode 1


###################### building ######################
python3 topo_opt.py --domain building --objf frequency --confs volume_ub compliance --nx 240 --r0 6.0 --vol-frac-ub 0.3 --compliance-ub-percent 4.0 --min-compliance 7.955e-6

# python3 topo_opt.py --domain building --objf frequency --confs volume_ub compliance displacement --nx 240 --r0 6.0 --vol-frac-ub 0.3 --compliance-ub-percent 4.0 --min-compliance 7.955e-6 --proj --delta-p 0.01 --delta-beta 0.1 --dis-ub 0.0 --mode 1 --maxit 1000

# python3 topo_opt.py --domain building --objf compliance-buckling --confs volume_ub displacement --nx 240 --r0 6.0 --vol-frac-ub 0.3 --c0 7.4e-6 --mu-ks0 0.1 --delta-p 0.01 --delta-beta 0.1 --weight 0.3 --maxit 1000 --dis-ub 4.0

# python3 topo_opt.py --domain building --objf compliance-buckling --confs volume_ub displacement --nx 240 --r0 6.0 --vol-frac-ub 0.3 --c0 7.4e-6 --mu-ks0 0.1 --delta-p 0.01 --delta-beta 0.1 --weight 0.8 --maxit 15000 --dis-ub 7.0 --iter-crit-w 1000 --note wit=1000

# python3 topo_opt.py --domain building --objf compliance-buckling --confs volume_ub displacement --nx 240 --r0 6.0 --vol-frac-ub 0.3 --c0 7.4e-6 --mu-ks0 0.1 --delta-p 0.01 --delta-beta 0.1 --weight 0.2 --maxit 1000 --dis-ub 7.0 --N-a 0 --N-b 0 --maxit 1100 --mode 3


###################### square ######################
# python3 topo_opt.py --domain square --objf compliance-buckling --confs volume_ub --nx 300 --r0 6.0 --vol-frac-ub 0.25 --proj --maxit 1000  --c0 1e-5 --mu-ks0 0.1 --weight 0.4 --delta-p 0.01 --delta-beta 0.1

# python3 topo_opt.py --domain square --objf compliance-buckling --confs volume_ub displacement --nx 300 --r0 6.0 --vol-frac-ub 0.25 --c0 1e-5 --mu-ks0 0.1 --weight 0.4 --dis-ub 4.5 --delta-p 0.01 --delta-beta 0.1 --proj --tracking --ks-rho-buckling 3000 --mode 4

