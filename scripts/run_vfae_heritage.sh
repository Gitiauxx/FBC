#!/bin/bash

_files="$@"

for seed in {0..50..1}
do
	echo $seed
	./fbcenv/bin/python3 eval.py --config_path configs/heritage/heritage_pareto_vfae.yml --seed $seed --gamma 500.0 --beta 0
  ./fbcenv/bin/python3 eval.py --config_path configs/heritage/heritage_pareto_vfae.yml --seed $seed --gamma 500.0 --beta 0.2
  ./fbcenv/bin/python3 eval.py --config_path configs/heritage/heritage_pareto_vfae.yml --seed $seed --gamma 500.0 --beta 0.4
  ./fbcenv/bin/python3 eval.py --config_path configs/heritage/heritage_pareto_vfae.yml --seed $seed --gamma 500.0 --beta 0.6
  ./fbcenv/bin/python3 eval.py --config_path configs/heritage/heritage_pareto_vfae.yml --seed $seed --gamma 500.0 --beta 0.8
  ./fbcenv/bin/python3 eval.py --config_path configs/heritage/heritage_pareto_vfae.yml --seed $seed --gamma 500.0 --beta 1.0
  ./fbcenv/bin/python3 eval.py --config_path configs/heritage/heritage_pareto_vfae.yml --seed $seed --gamma 500.0 --beta 1.2
  ./fbcenv/bin/python3 eval.py --config_path configs/heritage/heritage_pareto_vfae.yml --seed $seed --gamma 500.0 --beta 1.4
  ./fbcenv/bin/python3 eval.py --config_path configs/heritage/heritage_pareto_vfae.yml --seed $seed --gamma 500.0 --beta 1.6
  ./fbcenv/bin/python3 eval.py --config_path configs/heritage/heritage_pareto_vfae.yml --seed $seed --gamma 500.0 --beta 1.8
  ./fbcenv/bin/python3 eval.py --config_path configs/heritage/heritage_pareto_vfae.yml --seed $seed --gamma 500.0 --beta 2.0
  ./fbcenv/bin/python3 eval.py --config_path configs/heritage/heritage_pareto_vfae.yml --seed $seed --gamma 500.0 --beta 2.2
  ./fbcenv/bin/python3 eval.py --config_path configs/heritage/heritage_pareto_vfae.yml --seed $seed --gamma 500.0 --beta 2.4
  ./fbcenv/bin/python3 eval.py --config_path configs/heritage/heritage_pareto_vfae.yml --seed $seed --gamma 500.0 --beta 2.6
  ./fbcenv/bin/python3 eval.py --config_path configs/heritage/heritage_pareto_vfae.yml --seed $seed --gamma 500.0 --beta 2.8
  ./fbcenv/bin/python3 eval.py --config_path configs/heritage/heritage_pareto_vfae.yml --seed $seed --gamma 500.0 --beta 3.0
  ./fbcenv/bin/python3 eval.py --config_path configs/heritage/heritage_pareto_vfae.yml --seed $seed --gamma 500.0 --beta 3.2
done