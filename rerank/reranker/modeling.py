# Copyright 2021 Reranker Author. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
import math
import torch
import torch.nn.functional as F
import copy
from transformers import AutoModelForSequenceClassification, AutoTokenizer,\
    PreTrainedModel, PreTrainedTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPooling
from torch import nn
import torch.distributed as dist

from .arguments import ModelArguments, DataArguments, \
    RerankerTrainingArguments as TrainingArguments
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from .FGM import FGM

logger = logging.getLogger(__name__)


class Reranker(nn.Module):
    def __init__(self, hf_model: PreTrainedModel, model_args: ModelArguments, data_args: DataArguments,
                 train_args: TrainingArguments):
        super().__init__()
        self.hf_model = hf_model
        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.register_buffer(
            'target_label',
            torch.zeros(self.train_args.per_device_train_batch_size, dtype=torch.long)
        )
        self._keys_to_ignore_on_load_missing=None
        self._keys_to_ignore_on_save=None
        if train_args.local_rank >= 0:
            self.world_size = dist.get_world_size()
        
   
    def forward(self, batch):
        # print('model device:',self.hf_model.device,'input_device',batch['input_ids'].device)
        batch = {k:v.to(self.hf_model.device) for k,v in batch.items()}
        
        # bert_batch = {k:v for k,v in batch.items() if k != 'score'}
        ranker_out: SequenceClassifierOutput = self.hf_model(**batch, return_dict=True,output_hidden_states=True)
        logits = ranker_out.logits

        if self.model_args.temperature is not None:
            assert self.model_args.temperature > 0
            logits = logits / self.model_args.temperature

        if self.train_args.collaborative:
            logits = self.dist_gather_tensor(logits)
            logits = logits.view(
                self.world_size,
                self.train_args.per_device_train_batch_size,
                -1  # chunk
            )
            logits = logits.transpose(0, 1).contiguous()

        if self.training:
            scores = logits.view(
                self.train_args.per_device_train_batch_size,
                self.data_args.train_group_size
            )
            # prior_scores = batch['score'].view(self.train_args.per_device_train_batch_size,self.data_args.train_group_size)
            # scores = scores + prior_scores
            ''' 
            scores = torch.exp(scores)
            pos_score = scores[:,0]
            neg_score = scores[:,1:]
            # loss = -torch.log(pos_score / (pos_score + torch.sum(neg_score,1)))
            beta = 2
            tau_plus = 0.001 
            N = neg_score.size(1)
            # reweight = 2*neg_score / torch.max(neg_score.max(1).values.abs(), neg_score.min(1).values.abs()).unsqueeze(1)
            reweight = (beta*neg_score) / neg_score.mean(1).unsqueeze(1)
             Neg = torch.max((-N*tau_plus*pos_score.unsqueeze(1) + reweight*neg_score).sum(1) / (1-tau_plus), torch.ones(neg_score.size(0)).to(neg_score.device) *math.exp(-1))
            # Neg = (-N*tau_plus*pos_score.unsqueeze(1) + reweight*neg_score).sum(1) / (1-tau_plus)
            loss = -torch.log( pos_score / (pos_score + Neg))
            loss = loss.mean()
            ''' 
            # loss = -pos_score + torch.log(torch.sum(torch.exp(neg_score),1))
            # loss = torch.mean(loss)
            loss = self.cross_entropy(scores, self.target_label)
            # if self.train_args.collaborative or self.train_args.distance_cahce:
                # account for avg in all reduce
                # loss = loss.float() * self.world_size

            ## R-dropout
            
            return SequenceClassifierOutput(
                loss=loss,
                **ranker_out,
            )
        else:
            return ranker_out

    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments, data_args: DataArguments, train_args: TrainingArguments,
            *args, **kwargs
    ):
        hf_model = AutoModelForSequenceClassification.from_pretrained(*args, **kwargs)
        reranker = cls(hf_model, model_args, data_args, train_args)
        return reranker

    def save_pretrained(self, output_dir: str):
        self.hf_model.save_pretrained(output_dir)

    def dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.train_args.local_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors


class RerankerDC(Reranker):
    def compute_grad(self, scores: torch.Tensor):
        scores = scores.view(
            self.train_args.per_device_train_batch_size,
            self.data_args.train_group_size
        ).detach().requires_grad_()
        loss = self.cross_entropy(scores, self.target_label)
        loss.backward()

        return loss.detach(), scores.grad

    def forward(self, batch, grad_tensor: torch.Tensor = None):
        ranker_out: SequenceClassifierOutput = self.hf_model(**batch, return_dict=True)
        logits = ranker_out.logits

        if self.training:
            if grad_tensor is not None:
                return torch.dot(logits.float().flatten(), grad_tensor.flatten())
            else:
                return logits

        else:
            return ranker_out


class RerankerForInference(nn.Module):
    def __init__(
            self,
            hf_model: Optional[PreTrainedModel] = None,
            tokenizer: Optional[PreTrainedTokenizer] = None
    ):
        super().__init__()
        self.hf_model = hf_model
        self.tokenizer = tokenizer

    def tokenize(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    def forward(self, batch):
        return self.hf_model(**batch)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        hf_model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path)
        hf_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path)

        hf_model.eval()
        return cls(hf_model, hf_tokenizer)

    def load_pretrained_model(self, pretrained_model_name_or_path, *model_args, **kwargs):
        self.hf_model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )

    def load_pretrained_tokenizer(self, pretrained_model_name_or_path, *inputs, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, *inputs, **kwargs
        )
