#!/bin/bash
# Toy battery sweep: one job per (fn, N, seed[, method group]).
cd /coc/flash10/ksingara3/cobweb-forest
SNAP=$(scripts/snapshot.sh toys)
echo "snapshot $SNAP"
for seed in 0 1 2; do
  for fn in sin j0 expsin xy nested4; do
    for N in 250 1000 4000 16000; do
      SNAP=$SNAP sbatch --export=ALL,SNAP=$SNAP -J toy-$fn-$N-$seed --time=8:00:00 scripts/cpu.sbatch \
        $SNAP/experiments/toys.py --fn $fn --N $N --seed $seed \
        --methods $( [ $fn = sin ] || [ $fn = j0 ] && echo single,mlp,kan || echo single,act,label,hybrid,act-split,mlp,kan )
    done
  done
  for N in 250 1000 4000 16000; do
    for m in single act label hybrid act-split mlp kan; do
      SNAP=$SNAP sbatch --export=ALL,SNAP=$SNAP -J toy-exp100-$N-$seed-$m --time=16:00:00 scripts/cpu.sbatch \
        $SNAP/experiments/toys.py --fn exp100 --N $N --seed $seed --methods $m --out results/toys/exp100_parts/$m
    done
  done
done
