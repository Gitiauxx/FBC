#!/bin/bash

#SBATCH --job-name=synth_fbc
#SBATCH --output=/scratch/xgitiaux/synth_fbc_%j_%a.out
#SBATCH --error=/scratch/xgitiaux/synth_fbc_%j_%a.error
#SBATCH --mail-user=xgitiaux@gmu.edu
#SBATCH --mail-type=END
#SBATCH --export=ALL
#SBATCH --partition=all-HiPri
#SBATCH --nodes 1
#SBATCH --mem=4G
#SBATCH --tasks 1
#SBATCH --qos=csqos
#SBATCH --array=0-50

echo $SLURM_ARRAY_TASK_ID
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.0
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.02
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.04
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.06
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.08
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.10
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.12
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.14
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.16
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.18
#../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.2
#../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 1.1
#../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 1.2
#../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 1.3
#../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 1.4
#../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 1.5
#../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 1.6
#../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 1.7
#../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 1.8

