#!/bin/bash

_files="$@"

for seed in {0..50..1}
do
	echo $seed
	./fbcenv/bin/python3 eval.py --config_path configs/compas/compas_pareto_adv.yml --seed $seed --gamma 0.0 --beta 0
  ./fbcenv/bin/python3 eval.py --config_path configs/compas/compas_pareto_adv.yml --seed $seed --gamma 0.0 --beta 0.025
  ./fbcenv/bin/python3 eval.py --config_path configs/compas/compas_pareto_adv.yml --seed $seed --gamma 0.0 --beta 0.05
  ./fbcenv/bin/python3 eval.py --config_path configs/compas/compas_pareto_adv.yml --seed $seed --gamma 0.0 --beta 0.075
  ./fbcenv/bin/python3 eval.py --config_path configs/compas/compas_pareto_adv.yml --seed $seed --gamma 0.0 --beta 0.1
  ./fbcenv/bin/python3 eval.py --config_path configs/compas/compas_pareto_adv.yml --seed $seed --gamma 0.0 --beta 0.125
  ./fbcenv/bin/python3 eval.py --config_path configs/compas/compas_pareto_adv.yml --seed $seed --gamma 0.0 --beta 0.15
  ./fbcenv/bin/python3 eval.py --config_path configs/compas/compas_pareto_adv.yml --seed $seed --gamma 0.0 --beta 0.175
  ./fbcenv/bin/python3 eval.py --config_path configs/compas/compas_pareto_adv.yml --seed $seed --gamma 0.0 --beta 0.2
  ./fbcenv/bin/python3 eval.py --config_path configs/compas/compas_pareto_adv.yml --seed $seed --gamma 0.0 --beta 0.225
  ./fbcenv/bin/python3 eval.py --config_path configs/compas/compas_pareto_adv.yml --seed $seed --gamma 0.0 --beta 0.25
  ./fbcenv/bin/python3 eval.py --config_path configs/compas/compas_pareto_adv.yml --seed $seed --gamma 0.0 --beta 0.275
  ./fbcenv/bin/python3 eval.py --config_path configs/compas/compas_pareto_adv.yml --seed $seed --gamma 0.0 --beta 0.3
  ./fbcenv/bin/python3 eval.py --config_path configs/compas/compas_pareto_adv.yml --seed $seed --gamma 0.0 --beta 0.325
  ./fbcenv/bin/python3 eval.py --config_path configs/compas/compas_pareto_adv.yml --seed $seed --gamma 0.0 --beta 0.35
  ./fbcenv/bin/python3 eval.py --config_path configs/compas/compas_pareto_adv.yml --seed $seed --gamma 0.0 --beta 0.375
  ./fbcenv/bin/python3 eval.py --config_path configs/compas/compas_pareto_adv.yml --seed $seed --gamma 0.0 --beta 0.4
done