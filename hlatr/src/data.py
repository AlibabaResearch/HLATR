# Copyright 2021 Reranker Author. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
from dataclasses import dataclass

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

from .arguments import DataArguments, RerankerTrainingArguments
from transformers import PreTrainedTokenizer, BatchEncoding
from transformers import DataCollatorWithPadding
import datasets
from collections import defaultdict 
from tqdm import tqdm
import os
import json

@dataclass
class GroupedTrainDataset(Dataset):
    query_columns = ['qid', 'query']
    document_columns = ['pid', 'passage']

    def __init__(
            self,
            args: DataArguments,
            path_to_tsv: Union[List[str], str],
            train_args: RerankerTrainingArguments = None,
    ):
        self.nlp_dataset = datasets.load_dataset(
            # '/home/kunka.xgw/.cache/huggingface/datasets/json.py',
            # path='/home/admin/workspace/dataset_cache/json.py',
            'json',
            data_files=path_to_tsv,
            ignore_verifications=False,
            features=datasets.Features({
                'qry': {
                    'qid': datasets.Value('string'),
                },
                'psg': [{
                    'pid': datasets.Value('string'),
                    'emb': [datasets.Value('float32')],
                    'label': datasets.Value('int32'),
                    'rank': datasets.Value('int32'),
                    'score': datasets.Value('float32'),
                    'recall_score': datasets.Value('float32'),
                }]
            }
            )
        )['train']
        self.total_len = len(self.nlp_dataset)


    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> List[BatchEncoding]:
        group = self.nlp_dataset[item]['psg']
        # sorted_index = [(i,ele['rank']) for i,ele in enumerate(group)]
        sorted_index = [(i,ele['recall_score']) for i,ele in enumerate(group)]
        sorted_index = sorted(sorted_index,key=lambda x:x[1],reverse=True)
        # psgs = sorted(group,key=lambda x:x['rank'])
        psgs = [group[i[0]] for i in sorted_index]
        examples = [ele['emb'] for ele in psgs]
        labels = [ele['label'] for ele in psgs]
        scores = [ele['score'] for ele in psgs]
        mask = [1] * len(examples)
        features = {'inputs':examples,'attention_mask':mask,'scores':scores,'labels':labels}
        return features

@dataclass
class PredictionDataset(Dataset):
    def __init__(self, path_to_json: List[str],eval_id_file=None):
        self.nlp_dataset = datasets.load_dataset(
            'json',
            # path ='/home/admin/workspace/dataset_cache/json.py',
            # path = '/home/kunka.xgw/.cache/huggingface/datasets/json.py',
            data_files=path_to_json,
            )['train']
        if eval_id_file is not None and os.path.exists(eval_id_file):
            with open(eval_id_file) as f:
                self.eval_lis = json.load(f)
        else:
            self.eval_lis=defaultdict(list)
            for ele in tqdm(self.nlp_dataset):
                qid = ele['qry']['qid']
                psgs = ele['psg']
                sorted_index = [(i,ele['recall_score']) for i,ele in enumerate(psgs)]
                sorted_index = sorted(sorted_index,key =lambda x :x[1],reverse=True)
                # psgs = sorted(psgs,key=lambda x:x['rank'])
                # print(score_list)
                # pids = [ele['pid'] for ele in psgs]
                # labels = [ele[]]
                for index in sorted_index:
                    psg = psgs[index[0]]
                    self.eval_lis[qid].append((psg['pid'],psg['label']))
            with open(eval_id_file,'w') as fout:
                json.dump(self.eval_lis,fout)
                    
        
    def __len__(self):
        return len(self.nlp_dataset)

    def __getitem__(self, item):
        # qid, pid, emb = (self.nlp_dataset[item][f] for f in self.columns)
        # embs = [ele['emb'] for ele in self.nlp_dataset[item]]
        group = self.nlp_dataset[item]['psg']
        psgs = sorted(group,key=lambda x:x['recall_score'],reverse=True)
        # psgs = sorted(group,key=lambda x:x['rank'],reverse=False)
        examples = [ele['emb'] for ele in psgs]
        scores = [ele['score'] for ele in psgs]
        mask = [1] * len(examples)
        features = {'inputs':examples,'attention_mask':mask,'scores':scores}
        return features


@dataclass
class GroupCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """

    def __call__(
            self, features
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if isinstance(features[0], list):
            features = sum(features, [])
        return super().__call__(features)


def padding(lis,max_len,pad_tok=0):
    if isinstance(lis[0],float) or isinstance(lis[0],int):
        lis = lis + [pad_tok] * (max_len - len(lis))
    elif isinstance(lis[0],list):
        pad = [pad_tok] * len(lis[0])
        lis = lis + [pad] * (max_len-len(lis))
    # assert len(lis) == max_len
    return lis


@dataclass
class GroupCollator:
    """
    Very simple data collator that simply collates batches of dict-like objects and performs special handling for
    potential keys named:
        - `label`: handles a single value (int or float) per object
        - `label_ids`: handles a list of values per object
    Does not do any additional preprocessing: property names of the input object will be used as corresponding inputs
    to the model. See glue and ner for example of how it's useful.
    This is an object (like other data collators) rather than a pure function like default_data_collator. This can be
    helpful if you need to set a return_tensors value at initialization.
    Args:
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:
        if not isinstance(features[0], (dict, BatchEncoding)):
            features = [vars(f) for f in features]
        first = features[0]
        batch = {}
        # Special handling for labels.
        # Ensure that tensor is created with the correct type
        # (it should be automatically the case, but let's make sure of it.)
        # max_len = max([len(ele['inputs']) for ele in features])
        max_len=99
        if "labels" in first and first["labels"] is not None:
            label = first["labels"].item() if isinstance(first["labels"], torch.Tensor) else first["labels"]
            dtype = torch.long if isinstance(label, int) else torch.float
            batch["labels"] = torch.tensor([padding(f["labels"],max_len) for f in features], dtype=dtype)
        elif "label_ids" in first and first["label_ids"] is not None:
            if isinstance(first["label_ids"], torch.Tensor):
                batch["labels"] = torch.stack([f["label_ids"] for f in features])
            else:
                dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
                batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

        # Handling of all other possible keys.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        for k, v in first.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                else:
                    try:
                        batch[k] = torch.tensor([padding(f[k],max_len) for f in features])
                    except Exception as e:
                        print(k,[len(padding(f[k],max_len)) for f in features])
                        raise ValueError
        return batch
