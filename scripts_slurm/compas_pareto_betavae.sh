#!/bin/bash

#SBATCH --job-name=compas_eo_bv
#SBATCH --output=/scratch/xgitiaux/compas_eobv_%j_%a.out
#SBATCH --error=/scratch/xgitiaux/compas_eobv_%j_%a.error
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
../fvae-env/bin/python3 eval.py --config_path configs/compas/compas_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.0
../fvae-env/bin/python3 eval.py --config_path configs/compas/compas_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.2
../fvae-env/bin/python3 eval.py --config_path configs/compas/compas_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.4
../fvae-env/bin/python3 eval.py --config_path configs/compas/compas_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.6
../fvae-env/bin/python3 eval.py --config_path configs/compas/compas_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.8
../fvae-env/bin/python3 eval.py --config_path configs/compas/compas_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 1.0
../fvae-env/bin/python3 eval.py --config_path configs/compas/compas_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 1.2
../fvae-env/bin/python3 eval.py --config_path configs/compas/compas_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 1.4
../fvae-env/bin/python3 eval.py --config_path configs/compas/compas_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 1.6
../fvae-env/bin/python3 eval.py --config_path configs/compas/compas_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 1.8
../fvae-env/bin/python3 eval.py --config_path configs/compas/compas_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 2.0
