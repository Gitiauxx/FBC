#!/bin/bash

#SBATCH --job-name=synth_bv
#SBATCH --output=/scratch/xgitiaux/synth_bv_%j_%a.out
#SBATCH --error=/scratch/xgitiaux/synth_bv_%j_%a.error
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
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.0
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.2
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.4
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.6
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 0.8
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 1.0
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 1.2
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 1.4
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 1.6
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 1.8
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 2.0
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 2.2
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 2.4
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 2.6
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 2.8
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 3.0
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 3.2
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 3.4
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 3.6
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 3.8
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 4.0
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 4.2
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 4.4
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 4.6
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 4.8
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 5.0
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 5.2
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 5.4
../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 5.6
#../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 10.0
#../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 12.5
#../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 15.0
#../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 17.5
#../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 20.0
#../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 25.0
#../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 30.0
#../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 35.0
#../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 40.0
#../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 50.0
#../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 60.0
#../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 70.0
#../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 80.0
#../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 90.0
#../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 100.0
#../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 150.0
#../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 200.0
#../fvae-env/bin/python3 eval.py --config_path configs/synthetic/synth_pareto_betavae.yml --seed $SLURM_ARRAY_TASK_ID --beta 250.0

