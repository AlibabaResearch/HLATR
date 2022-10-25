# BERT-base Passage Reranking

Code for training reranking model. Our code is developed based on the [reranker](https://github.com/luyug/Reranker) framework. 

## Training

- Data preprocessing

building train and dev dataset
```
usage: build_train.py [-h] [--tokenizer_name TOKENIZER_NAME] [--truncate TRUNCATE] [--qrel_file QREL_FILE] [--query_file QUERY_FILE] [--corpus_file CORPUS_FILE] [--retrieval_file RETRIEVAL_FILE] [--ranking_file RANKING_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --tokenizer_name TOKENIZER_NAME
                        bert tokenizer name
  --truncate TRUNCATE   bert tokenizer max sequence length
  --qrel_file QREL_FILE
                        qrels train file
  --query_file QUERY_FILE
                        query train file
  --corpus_file CORPUS_FILE
                        corpus file
  --retrieval_file RETRIEVAL_FILE
                        train query retrieval result
  --ranking_file RANKING_FILE
                        ranking train save file
```

```
usage: build_dev.py [-h] [--tokenizer_name TOKENIZER_NAME] [--truncate TRUNCATE] [--qrel_file QREL_FILE] [--query_file QUERY_FILE] [--corpus_file CORPUS_FILE] [--topk TOPK] [--retrieval_file RETRIEVAL_FILE] [--ranking_file RANKING_FILE] [--label_file LABEL_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --tokenizer_name TOKENIZER_NAME
                        bert tokenizer name
  --truncate TRUNCATE   bert tokenizer max sequence length
  --qrel_file QREL_FILE
                        qrels train file
  --query_file QUERY_FILE
                        query train file
  --corpus_file CORPUS_FILE
                        corpus file
  --topk TOPK           select topk result
  --retrieval_file RETRIEVAL_FILE
                        train query retrieval result
  --ranking_file RANKING_FILE
                        ranking train save file
  --label_file LABEL_FILE
                        ranking train save file
```

- Training
```
TRAIN_DATA_DIR=$train_data
DEV_DATA_DIR=$train_data
gpu="0,1,2,3"
num_gpu=4

CUDA_VISBLE_DEVICES=$gpu python -m torch.distributed.launch --master_port 2056 --nproc_per_node $num_gpu run_marco.py \
  --output_dir fintune_models/bert \ 
  --model_name_or_path bert-base-chinese \
  --do_train \
  --train_dir ${TRAIN_DATA_DIR} \
  --dev_path ${DEV_DATA_DIR}/dev.top100.json \
  --dev_id_file ${DEV_DATA_DIR}/dev.top100.label.txt \
  --max_len 256 \
  --fp16 \
  --per_device_train_batch_size 8 \
  --train_group_size 8 \
  --gradient_accumulation_steps 2 \
  --per_device_eval_batch_size 16 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --learning_rate 1e-5 \
  --num_train_epochs 3 \
  --overwrite_output_dir \
  --dataloader_num_workers 8 \
  --evaluation_strategy steps \
  --eval_steps 1000 \
  --save_steps 1000 \
```

## Inference

Inference the ranking score of each query-passage pair (here, we only rerank the top1000 retrieval passages for each query)
```
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
