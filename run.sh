#!/bin/zsh
# create a directory for the results if it does not exist
if [ ! -d result ]; then
  mkdir result
fi

# ======================================================================
#                               single run
# ======================================================================

# obj: frequency, conf: volume
# python3.10 topo_opt.py --optimizer mma4py --objf frequency --confs volume --vol-frac 0.5 --nx 1000 --m0-block-frac 0.0 --maxit 300

# obj: volume, conf: frequency
# python3.10 topo_opt.py --optimizer tr --objf volume --confs frequency --nx 48 --m0-block-frac 0.1 --omega-lb 5

# obj: frequency, conf: stress, volume
# python3.10 topo_opt.py --optimizer tr --objf frequency --confs stress volume --nx 50 --m0-block-frac 0.1 --stress-ub 1200 --vol-frac 0.6

# obj: volume, conf: stress, frequency
python topo_opt.py --optimizer mma4py --objf compliance --confs stress volume --nx 100 --vol-frac 0.5 --stress-ub 3000 --domain mbbbeam --movelim 0.2 --r0 2.0

# python3.10 topo_opt.py --optimizer mma4py --objf volume --confs stress --nx 100 --stress-ub 5006 --domain mbbbeam --maxit 300
# python3.10 topo_opt.py --optimizer mma4py --objf volume --confs stress --nx 100 --m0-block-frac 0.0 --vol-frac 0.3 --stress-ub 1502 --domain mbbbeam
# obj: stress, conf: frequency, volume
# python3.10 topo_opt.py --optimizer mma4py --objf stress --confs frequency volume --nx 50 --m0-block-frac 0.1 --omega-lb 5 --vol-frac 0.2


# ======================================================================
#                                   sweep
# ======================================================================

# sweep: obj: volume, conf: frequency
# for w in 4.5; do
#   python3.10 topo_opt.py --optimizer tr --objf volume --confs frequency --omega-lb $w --nx 96 --m0-block-frac 0.1
# done

# sweep: obj: frequency, conf: volume
# for v in 0.4 0.5 0.6 0.7 0.8 0.9; do
#   python3.10 topo_opt.py --optimizer tr --objf frequency --confs volume --vol-frac $v --nx 96 --m0-block-frac 0.1
# done

# sweep: obj: volume, conf: stress, frequency
# for w in 2; do
#   for s in 800 900 1000 1100 1200 1400 1600; do
#     python3.10 topo_opt.py --optimizer tr --objf volume --confs stress frequency --omega-lb $w --stress-ub $s --nx 96 --m0-block-frac 0.1
#   done
# done

# sweep: obj: stress, conf: frequency, volume
# for w in 4.5; do
#   for v in 0.4 0.5 0.6 0.7 0.8 0.9; do
#     python3.10 topo_opt.py --optimizer tr --objf stress --confs frequency volume --omega-lb $w --vol-frac $v --nx 96 --m0-block-frac 0.1
#   done
# done

# sweep: obj: frequency, conf: stress, volume
# for s in 800 1000 1200 1400 1600 1800 2000; do
#   for v in 0.4 0.5 0.6 0.7 0.8 0.9; do
#     python3.10 topo_opt.py --optimizer tr --objf frequency --confs stress volume --stress-ub $s --vol-frac $v --nx 96 --m0-block-frac 0.1
#   done
# done

# Hi, 