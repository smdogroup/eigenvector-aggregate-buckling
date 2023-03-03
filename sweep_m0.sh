#!/bin/zsh

for m0 in 1.0 5.0 10.0 20.0 50.0 100.0; do
  python topo_opt.py --prefix m0=$m0 --maxit 300 --m0 $m0
done
