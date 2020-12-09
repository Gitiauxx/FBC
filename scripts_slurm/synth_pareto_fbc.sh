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
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.1
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.2
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.3
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.4
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.5
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.6
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.7
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.8
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.9
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 1.0
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 1.1
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 1.2
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 1.3
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 1.4
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 1.5
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 1.6
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 1.7
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 1.8

