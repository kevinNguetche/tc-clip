# fully-supervised ucf101 training with 2 V100 gpus
export CUDA_VISIBLE_DEVICES=0,1
export GPUS_PER_NODE=2

protocol=fully_supervised
dataset_name=ucf101
data=${protocol}_${dataset_name}

expr_name=tc_clip_reproduce
trainer=tc_clip
use_wandb=true

torchrun --nproc_per_node=${GPUS_PER_NODE} main.py -cn ${protocol} \
data=${data} \
output=workspace/expr/${data}/${expr_name}/${data}_${expr_name}_${trainer} \
trainer=${trainer} \
use_wandb=${use_wandb}
