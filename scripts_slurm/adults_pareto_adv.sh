#!/bin/bash

#SBATCH --job-name=adults_eo
#SBATCH --output=/scratch/xgitiaux/adults_eobv_%j_%a.out
#SBATCH --error=/scratch/xgitiaux/adults_eobv_%j_%a.error
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
../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 4.0
../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 4.5
../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 5.0
../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 5.5
../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 6.0
../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 6.5
../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 7.0
../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 7.5
../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 8.0
../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 8.5
../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 9.0
#../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.2
#../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.4
#../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.6
#../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.8
#../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 1.0
#../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 1.2
#../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 1.4
#../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 1.6
#../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 1.8
#../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 2.0
#../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 2.2
#../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 2.4
#../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 2.6
#../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 2.8
#../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 3.0
#../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 3.2
#../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 3.4
#../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 3.6

