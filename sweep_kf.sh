#!/bin/zsh

omega0=7.46799
for kf in $(seq 0.5 0.05 1.5); do
  omega=$(printf "%.5f" $((kf * omega0)))
  python topo_opt.py --prefix stress_kf=$kf --maxit 5 --nx 4 --objf stress --confs volume frequency --omega-lb $omega
done
