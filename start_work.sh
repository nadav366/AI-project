
# train on local machine
source /cs/ep/106/src/new_env/bin/activate.csh
module load tensorflow opencv cuda cudnn
cd /cs/ep/106/AI
python run_train.py params.json


# train on slurm
ssh nadav366%phoenix-gw@gw.cs.huji.ac.il
cd /cs/ep/106/AI

# send one-
sbatch /cs/ep/106/AI/train_one_in_gpu.sh

# send dir-
bash run_dir.sh /cs/ep/106/AI/train_jsons/basic_model_params/

squeue | grep nadav366
squeue | grep script.s
