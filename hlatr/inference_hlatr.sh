DATA_DIR=./msmarco_passage
train_dir=${DATA_DIR}/
dev_dir=${DATA_DIR}/
dev_id_file=${DATA_DIR}/

MODEL_PATH=fintune_models/ 
RESULT_PATH=${DATA_DIR}/

export NCCL_SOCKET_IFNAME=bond0,docker0,eth2,eth3,lo
export NCCL_IB_DISABLE=1
echo "model_path  $MODEL_PATH"
echo "result path $RESULT_PATH"

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 run_list_marco.py \
  --output_dir ${MODEL_PATH} \
  --model_name_or_path ${MODEL_PATH} \
  --do_predict \
  --max_len 256 \
  --fp16 \
  --per_device_eval_batch_size 64 \
  --eval_accumulation_steps 1000 \
  --dataloader_num_workers 128 \
  --pred_path ${dev_dir} \
  --pred_id_file ${dev_id_file}\
  --rank_score_path ${RESULT_PATH} 

