## Hybrid List Aware Transformer Reranking
Code for training hlatr model. 

## Training

- Data preprocessing

To build train and dev data for HLATR, you first need to get the retrieval and rerank result. Then you can convert them to the HLATR data by the following script:
```
usage: convert_res_to_listrank.py [-h] --score_file SCORE_FILE --output_file OUTPUT_FILE [--qrel_path QREL_PATH] [--eval_id_file EVAL_ID_FILE] [--recall_path RECALL_PATH] [--tag TAG]

optional arguments:
  -h, --help            show this help message and exit
  --score_file SCORE_FILE
                        rerank result file
  --output_file OUTPUT_FILE
                        output file path
  --qrel_path QREL_PATH
                        output file path
  --eval_id_file EVAL_ID_FILE
                        output evallist path
  --recall_path RECALL_PATH
                        retrieval result file path
  --tag TAG             trian/dev/test for different set
```
You can see a demo data in "data/".

- Training
This starts training on 4 GPUs with DDP.
```
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
  --num_train_epochs 30 \
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
```

## Inference
```
DATA_DIR=./msmarco_passage
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
```

## Evaluation
```
usage: evaluate.py [-h] [--topk_path TOPK_PATH] [--qrel_path QREL_PATH] [--topk TOPK]

optional arguments:
  -h, --help            show this help message and exit
  --topk_path TOPK_PATH
  --qrel_path QREL_PATH
  --topk TOPK
```
