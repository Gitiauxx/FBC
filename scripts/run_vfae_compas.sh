#!/bin/bash

_files="$@"

for seed in {0..50..1}
do
	echo $seed
	./fbcenv/bin/python3 eval.py --config_path configs/compas/compas_pareto_vfae.yml --seed $seed --gamma 500.0 --beta 0
  ./fbcenv/bin/python3 eval.py --config_path configs/compas/compas_pareto_vfae.yml --seed $seed --gamma 500.0 --beta 0.1
  ./fbcenv/bin/python3 eval.py --config_path configs/compas/compas_pareto_vfae.yml --seed $seed --gamma 500.0 --beta 0.2
  ./fbcenv/bin/python3 eval.py --config_path configs/compas/compas_pareto_vfae.yml --seed $seed --gamma 500.0 --beta 0.3
  ./fbcenv/bin/python3 eval.py --config_path configs/compas/compas_pareto_vfae.yml --seed $seed --gamma 500.0 --beta 0.4
  ./fbcenv/bin/python3 eval.py --config_path configs/compas/compas_pareto_vfae.yml --seed $seed --gamma 500.0 --beta 0.5
  ./fbcenv/bin/python3 eval.py --config_path configs/compas/compas_pareto_vfae.yml --seed $seed --gamma 500.0 --beta 0.6
  ./fbcenv/bin/python3 eval.py --config_path configs/compas/compas_pareto_vfae.yml --seed $seed --gamma 500.0 --beta 0.7
  ./fbcenv/bin/python3 eval.py --config_path configs/compas/compas_pareto_vfae.yml --seed $seed --gamma 500.0 --beta 0.8
  ./fbcenv/bin/python3 eval.py --config_path configs/compas/compas_pareto_vfae.yml --seed $seed --gamma 500.0 --beta 0.9
  ./fbcenv/bin/python3 eval.py --config_path configs/compas/compas_pareto_vfae.yml --seed $seed --gamma 500.0 --beta 1.0
  ./fbcenv/bin/python3 eval.py --config_path configs/compas/compas_pareto_vfae.yml --seed $seed --gamma 500.0 --beta 1.1
  ./fbcenv/bin/python3 eval.py --config_path configs/compas/compas_pareto_vfae.yml --seed $seed --gamma 500.0 --beta 1.2
  ./fbcenv/bin/python3 eval.py --config_path configs/compas/compas_pareto_vfae.yml --seed $seed --gamma 500.0 --beta 1.3
  ./fbcenv/bin/python3 eval.py --config_path configs/compas/compas_pareto_vfae.yml --seed $seed --gamma 500.0 --beta 1.4
  ./fbcenv/bin/python3 eval.py --config_path configs/compas/compas_pareto_vfae.yml --seed $seed --gamma 500.0 --beta 1.5
done