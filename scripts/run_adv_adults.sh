#!/bin/bash

_files="$@"

for seed in {0..50..1}
do
	echo $seed
	./fbcenv/bin/python3 eval.py --config_path configs/adults/adults_pareto_adv.yml --seed $seed --gamma 0.0 --beta 0.0
  ./fbcenv/bin/python3 eval.py --config_path configs/adults/adults_pareto_adv.yml --seed $seed --gamma 0.0 --beta 0.2
  ./fbcenv/bin/python3 eval.py --config_path configs/adults/adults_pareto_adv.yml --seed $seed --gamma 0.0 --beta 0.4
  ./fbcenv/bin/python3 eval.py --config_path configs/adults/adults_pareto_adv.yml --seed $seed --gamma 0.0 --beta 0.6
  ./fbcenv/bin/python3 eval.py --config_path configs/adults/adults_pareto_adv.yml --seed $seed --gamma 0.0 --beta 0.8
  ./fbcenv/bin/python3 eval.py --config_path configs/adults/adults_pareto_adv.yml --seed $seed --gamma 0.0 --beta 1.0
  ./fbcenv/bin/python3 eval.py --config_path configs/adults/adults_pareto_adv.yml --seed $seed --gamma 0.0 --beta 1.2
  ./fbcenv/bin/python3 eval.py --config_path configs/adults/adults_pareto_adv.yml --seed $seed --gamma 0.0 --beta 1.5
  ./fbcenv/bin/python3 eval.py --config_path configs/adults/adults_pareto_adv.yml --seed $seed --gamma 0.0 --beta 2.0
  ./fbcenv/bin/python3 eval.py --config_path configs/adults/adults_pareto_adv.yml --seed $seed --gamma 0.0 --beta 2.5
  ./fbcenv/bin/python3 eval.py --config_path configs/adults/adults_pareto_adv.yml --seed $seed --gamma 0.0 --beta 3.0
  ./fbcenv/bin/python3 eval.py --config_path configs/adults/adults_pareto_adv.yml --seed $seed --gamma 0.0 --beta 3.5
done