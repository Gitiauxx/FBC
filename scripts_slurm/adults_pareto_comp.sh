#!/bin/bash

#SBATCH --job-name=adults_eo
#SBATCH --output=/scratch/xgitiaux/adults_eocomp_%j_%a.out
#SBATCH --error=/scratch/xgitiaux/adults_eocomp_%j_%a.error
#SBATCH --mail-user=xgitiaux@gmu.edu
#SBATCH --mail-type=../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.025
../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.05
../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.075
../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_fbc_reg.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.1
../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_fbc._regyml --seed $SLURM_ARRAY_TASK_ID --beta 0.125
../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.15
../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.175
../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.2
../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.225
../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.275
../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.3
../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.325
../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.35
../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.375
../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_fbc.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.4
END
#SBATCH --export=ALL
#SBATCH --partition=all-LoPri
#SBATCH --nodes 1
#SBATCH --mem=4G
#SBATCH --tasks 1
#SBATCH --qos=csqos
#SBATCH --array=0-50

echo $SLURM_ARRAY_TASK_ID
../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_fbc_reg.yml --seed $SLURM_ARRAY_TASK_ID --beta 0
../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_fbc_reg.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.025
../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_fbc_reg.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.05
../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_fbc_reg.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.075
../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_fbc_reg.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.1
../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_fbc_reg._regyml --seed $SLURM_ARRAY_TASK_ID --beta 0.125
../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_fbc_reg.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.15
../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_fbc_reg.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.175
../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_fbc_reg.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.2
../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_fbc_reg.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.225
../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_fbc_reg.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.275
../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_fbc_reg.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.3
../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_fbc_reg.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.325
../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_fbc_reg.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.35
../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_fbc_reg.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.375
../fvae-env/bin/python3 eval.py --config_path configs/adults/adults_pareto_fbc_reg.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.4

