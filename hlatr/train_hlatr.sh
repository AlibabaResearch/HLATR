#!/bin/bash
DATA_DIR=./data
model_path=./fintune_models/
train_dir=${DATA_DIR}/
dev_dir=${DATA_DIR}/
dev_id_file=${DATA_DIR}/

layer=4
bz=256
hidden_size=128
lr=1e-3
head=2

output_dir=fintune_models/hltar_bz${bz}_lr${lr}_head${head}_hs${hidden_size}_layer${layer}

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 29501 run.py  \
  --output_dir ${output_dir} \
  --model_name_or_path ${model_path} \
  --fp16 \
  --do_train \
  --train_dir ${train_dir} \
  --dev_path ${dev_dir} \
  --dev_id_file ${dev_id_file} \
  --max_len 128 \
  --per_device_train_batch_size ${bz} \
  --gradient_accumulation_steps 1 \
  --per_device_eval_batch_size 512 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --learning_rate ${lr} \
  --num_train_epochs 80 \
  --overwrite_output_dir \
  --dataloader_num_workers 88 \
  --evaluation_strategy steps \
  --eval_steps 125 \
  --save_steps 125 \
  --load_best_model_at_end true \
  --metric_for_best_model mrr10 \
  --hidden_size ${hidden_size} \
  --num_attention_heads ${head} \
  --num_hidden_layers ${layer} 
