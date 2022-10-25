DATA_PATH=./data
SCORE_FILE=$DATA_PATH/ # rerrank result with logits
RECALL_FILE=${DATA_PATH}/ # recall result 
OUTPUT_FILE=${DATA_PATH}/ # hltar data file 
QREL_PATH=$DATA_PATH/ # label file
EVAL_ID_FILE=${DATA_PATH}/ # hltar data id file
TAG=train # train/dev/test 

mkdir ${DATA_PATH}/${PREFIX}

SCORE_FILE=./data/demo_rerank.res
RECALL_FILE=./data/demo_retrieval.txt
OUTPUT_FILE=./data/demo_hlatr_train.json
QREL_PATH=/nas-alinlp/yanzhao.zyz/research/data/msmarco_passage/qrels.dev.tsv

python3 src/convert_res_to_listrank.py \
  --score_file $SCORE_FILE \
  --output_file $OUTPUT_FILE \
  --qrel_path $QREL_PATH \
  --eval_id_file $EVAL_ID_FILE \
  --tag $TAG \
  --recall_path $RECALL_FILE
