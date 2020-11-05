#!/bin/bash

_files="$@"

for seed in {0..50..1}
do
	echo $seed
	./fbcenv/bin/python3 eval.py --config_path configs/heritage/heritage_pareto_adv.yml --seed $seed --gamma 0.0 --beta 0
  ./fbcenv/bin/python3 eval.py --config_path configs/heritage/heritage_pareto_adv.yml --seed $seed --gamma 0.0 --beta 0.05
  ./fbcenv/bin/python3 eval.py --config_path configs/heritage/heritage_pareto_adv.yml --seed $seed --gamma 0.0 --beta 0.1
  ./fbcenv/bin/python3 eval.py --config_path configs/heritage/heritage_pareto_adv.yml --seed $seed --gamma 0.0 --beta 0.15
  ./fbcenv/bin/python3 eval.py --config_path configs/heritage/heritage_pareto_adv.yml --seed $seed --gamma 0.0 --beta 0.2
  ./fbcenv/bin/python3 eval.py --config_path configs/heritage/heritage_pareto_adv.yml --seed $seed --gamma 0.0 --beta 0.25
  ./fbcenv/bin/python3 eval.py --config_path configs/heritage/heritage_pareto_adv.yml --seed $seed --gamma 0.0 --beta 0.3
  ./fbcenv/bin/python3 eval.py --config_path configs/heritage/heritage_pareto_adv.yml --seed $seed --gamma 0.0 --beta 0.35
  ./fbcenv/bin/python3 eval.py --config_path configs/heritage/heritage_pareto_adv.yml --seed $seed --gamma 0.0 --beta 0.4
  ./fbcenv/bin/python3 eval.py --config_path configs/heritage/heritage_pareto_adv.yml --seed $seed --gamma 0.0 --beta 0.45
  ./fbcenv/bin/python3 eval.py --config_path configs/heritage/heritage_pareto_adv.yml --seed $seed --gamma 0.0 --beta 0.5
done