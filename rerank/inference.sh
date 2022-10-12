DEV_DATA_PATH=dev/
MODEL_PATH=fintune_models/
RESULT_PATH=result/

CUDA_VISIBLE_DEVICES=4 python run_marco.py \
  --output_dir ${MODEL_PATH} \
  --model_name_or_path ${MODEL_PATH} \
  --tokenizer_name ${MODEL_APTH}  \
  --do_predict \
  --max_len 256 \
  --fp16 \
  --per_device_eval_batch_size 64 \
  --dataloader_num_workers 8 \
  --pred_path ${DEV_DATA_PATH}/dev.top1000.json  \
  --pred_id_file  ${DEV_DATA_PATH}/dev.top1000.label.txt \
  --rank_score_path ${RESULT_PATH} \
  --output_emb \
  --eval_accumulation_steps 100 

