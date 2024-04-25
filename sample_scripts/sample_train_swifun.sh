#!/bin/bash

export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1ZmQ4ZjYxZS02Zjk2LTQ5Y2EtYmZlNC0zOWZjNzM3MjNjYzYifQ=="

TRAINER_ARGS="--accelerator gpu --max_epochs 20 --precision 16 --num_nodes 1 --devices 2 --strategy ddp_find_unused_parameters_false" #--strategy ddp_find_unused_parameters_false" # devices should be 4 for sbatch
MAIN_ARGS='--loggername neptune --classifier_module v6 --dataset_name UKB --image_path /scratch/connectome/junb/UKB_20227_3_MNI_to_TRs --task_path /storage/bigdata/UKB/brain_prep/3.task_fMRI/20249_3_MNI --mask_filename Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii.gz' #--mask_filename harvard_oxford-wholebrain-maxprob-thr25-2mm.nii.gz' # /global/cfs/cdirs/m4244/junbeom/20249_pickled_map
DATA_ARGS='--batch_size 2 --eval_batch_size 16 --num_workers 8'
DEFAULT_ARGS='--project_name symoon-ConnectomeLab/SwiFUN-v2'
OPTIONAL_ARGS='--cope 1 --c_multiplier 2 --clf_head_version v1 --downstream_task tfMRI_3D --use_scheduler --gamma 0.5 --cycle 0.7 --loss_type mse --last_layer_full_MSA True '  
RESUME_ARGS="" 

# node1의 gpu 6,7번 기준 batch size 2로 3시간 20분 정도 소요.
# 랩서버에서 돌릴 때는 0,1 // 2,3 // 4,5 // 6,7 이런식으로 두개씩 엮어서 사용할 것. 그 이상부터는 속도 하락이 큼.
# 해당 노드에 직접 이동해서 gpu 공간이 있는지 확인해서 수동으로 아래 gpu 번호를 지정해줄 것.
CUDA_VISIBLE_DEVICES=6,7 python project/main.py $TRAINER_ARGS \
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