
## Hybrid List Aware Transformer Reranking
Code for hltar stage.

## Data Generation:
```
bash construct_data.sh
```

train data format:
```
{"qry": {"qid": "query id"},
"psg": [{
"pid": "passage	id",
"emb": "reranker logits",
"label": "0/1 indicates irrelevant/relevant",
"rank": "retrieve order",
"score": "reranker score",
"recall_score": "retrieve score"
}]}
```

The format of eval data is same as the train data.

dev_id_file::
```
{"qid":[{"did": "passage id","label":"label"}]}
```
## Training

```
bash train_hltar.sh
```

## Inference
```
bash inference_hltar.sh
```

## Evalutaion
usage: evaluate.py [-h] [--topk_path TOPK_PATH] [--qrel_path QREL_PATH] [--topk TOPK]

```
optional arguments:
  -h, --help            show this help message and exit
  --topk_path TOPK_PATH
  --qrel_path QREL_PATH
  --topk TOPK
```
