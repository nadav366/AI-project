#!/bin/bash

#SBATCH --mem=64g
#SBATCH -c8
#SBATCH --time=1-0
#SBATCH --gres=gpu:2
#SBATCH --killable
#SBATCH --requeue

. /cs/ep/106/src/new_env/bin/activate
module load tensorflow cuda/10.0 
PATH_TO_SCRIPT="/cs/ep/106/AI"
python $PATH_TO_SCRIPT/run_train.py 