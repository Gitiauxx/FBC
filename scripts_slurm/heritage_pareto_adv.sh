#!/bin/bash

#SBATCH --job-name=heritage_eo
#SBATCH --output=/scratch/xgitiaux/heritage_eoadv_%j_%a.out
#SBATCH --error=/scratch/xgitiaux/heritage_eoadv_%j_%a.error
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
../fvae-env/bin/python3 eval.py --config_path configs/heritage/heritage_pareto_adv.yml --seed $SLURM_ARRAY_TASK_ID --beta 0
../fvae-env/bin/python3 eval.py --config_path configs/heritage/heritage_pareto_adv.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.05
../fvae-env/bin/python3 eval.py --config_path configs/heritage/heritage_pareto_adv.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.1
../fvae-env/bin/python3 eval.py --config_path configs/heritage/heritage_pareto_adv.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.15
../fvae-env/bin/python3 eval.py --config_path configs/heritage/heritage_pareto_adv.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.2
../fvae-env/bin/python3 eval.py --config_path configs/heritage/heritage_pareto_adv.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.25
../fvae-env/bin/python3 eval.py --config_path configs/heritage/heritage_pareto_adv.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.3
../fvae-env/bin/python3 eval.py --config_path configs/heritage/heritage_pareto_adv.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.35
../fvae-env/bin/python3 eval.py --config_path configs/heritage/heritage_pareto_adv.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.4
../fvae-env/bin/python3 eval.py --config_path configs/heritage/heritage_pareto_adv.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.45
../fvae-env/bin/python3 eval.py --config_path configs/heritage/heritage_pareto_adv.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.5

