#!/bin/bash

_files="$@"

for seed in {0..50..1}
do
	echo $seed
	./fbcenv/bin/python3 eval.py --config_path configs/adults/adults_pareto_mmd.yml --seed $seed --gamma 0.0 --beta 0.0
  ./fbcenv/bin/python3 eval.py --config_path configs/adults/adults_pareto_mmd.yml --seed $seed --gamma 50.0 --beta 0.0
  ./fbcenv/bin/python3 eval.py --config_path configs/adults/adults_pareto_mmd.yml --seed $seed --gamma 100.0 --beta 0.0
  ./fbcenv/bin/python3 eval.py --config_path configs/adults/adults_pareto_mmd.yml --seed $seed --gamma 200.0 --beta 0.0
  ./fbcenv/bin/python3 eval.py --config_path configs/adults/adults_pareto_mmd.yml --seed $seed --gamma 400.0 --beta 0.0
  ./fbcenv/bin/python3 eval.py --config_path configs/adults/adults_pareto_mmd.yml --seed $seed --gamma 800.0 --beta 0.0
  ./fbcenv/bin/python3 eval.py --config_path configs/adults/adults_pareto_mmd.yml --seed $seed --gamma 1200.0 --beta 0.0
  ./fbcenv/bin/python3 eval.py --config_path configs/adults/adults_pareto_mmd.yml --seed $seed --gamma 1600.0 --beta 0.0
  ./fbcenv/bin/python3 eval.py --config_path configs/adults/adults_pareto_mmd.yml --seed $seed --gamma 2000.0 --beta 0.0
  ./fbcenv/bin/python3 eval.py --config_path configs/adults/adults_pareto_mmd.yml --seed $seed --gamma 3000.0 --beta 0.0
  ./fbcenv/bin/python3 eval.py --config_path configs/adults/adults_pareto_mmd.yml --seed $seed --gamma 4000.0 --beta 0.0
  ./fbcenv/bin/python3 eval.py --config_path configs/adults/adults_pareto_mmd.yml --seed $seed --gamma 5000.0 --beta 0.0
  ./fbcenv/bin/python3 eval.py --config_path configs/adults/adults_pareto_mmd.yml --seed $seed --gamma 6000.0 --beta 0.0
  ./fbcenv/bin/python3 eval.py --config_path configs/adults/adults_pareto_mmd.yml --seed $seed --gamma 10000.0 --beta 0.0
done