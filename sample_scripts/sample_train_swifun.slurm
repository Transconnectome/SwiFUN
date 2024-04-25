#!/bin/bash
#SBATCH -t 6:00:00
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --chdir=/global/cfs/cdirs/m4244/junbeom/SwiFUN
#SBATCH --nodelist=node1
#SBATCH --output=R-%j-%x.out
set +x

export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4ZWNlMzk4Ny1hOTVlLTRjMzgtOWI4ZS1hY2FkYTY4MzNhYzMifQ=="

cd /global/cfs/cdirs/m4244/junbeom/SwiFUN 

#source /usr/anaconda3/etc/profile.d/conda.sh
#conda activate py39

module load pytorch/1.13.1
module load cpe/23.03

TRAINER_ARGS="--accelerator gpu --max_epochs 20 --precision 16 --num_nodes 1 --devices 1" # --strategy ddp_find_unused_parameters_false" #--strategy ddp_find_unused_parameters_false" # devices should be 4 for sbatch
MAIN_ARGS='--loggername neptune --dataset_name UKB_v3 --image_path  /pscratch/sd/j/junbeom/UKB_20227_3_MNI_to_TRs --task_path /pscratch/sd/j/junbeom/3.task_fMRI/20249_3_MNI --mask_filename Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii.gz' #--mask_filename harvard_oxford-wholebrain-maxprob-thr25-2mm.nii.gz' # /global/cfs/cdirs/m4244/junbeom/20249_pickled_map
DATA_ARGS='--batch_size 4 --eval_batch_size 16 --num_workers 8'
DEFAULT_ARGS='--project_name kjb961013/rs-to-task'
OPTIONAL_ARGS='--cope 1 --c_multiplier 2 --clf_head_version v1 --downstream_task tfMRI_3D --use_scheduler --gamma 0.5 --cycle 0.7 --loss_type mse --last_layer_full_MSA True --input_scaling_method minmax'
RESUME_ARGS="" 

# node1의 gpu 6,7번 기준 batch size 2로 3시간 20분 정도 소요.
# 랩서버에서 돌릴 때는 0,1 // 2,3 // 4,5 // 6,7 이런식으로 두개씩 엮어서 사용할 것. 그 이상부터는 속도 하락이 큼.
# 해당 노드에 직접 이동해서 gpu 공간이 있는지 확인해서 수동으로 아래 gpu 번호를 지정해줄 것.

#srun bash -c "
#source export_DDP_vars.sh
python project/main.py $TRAINER_ARGS \
  $MAIN_ARGS \
  $DEFAULT_ARGS \
  $DATA_ARGS \
  $OPTIONAL_ARGS \
  $RESUME_ARGS \
  --dataset_split_num 1 \
  --seed 1 \
  --learning_rate 5e-5 \
  --model swinunetr \
  --time_as_channel \
  --attn_drop_rate 0.3 \
  --depth 2 2 2 2 \
  --embed_dim 24 \
  --sequence_length 20 \
  --first_window_size 4 4 4 6 \
  --window_size 4 4 4 6 \
  --img_size 96 96 96
#  "
