# Copyright 2021 Reranker Author. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Dict, List, Tuple, Optional, Any, Union, Callable

from .dist.sampler import SyncedSampler
from .modeling import Reranker, RerankerDC
from .arguments import TrainingArguments
import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.utils.checkpoint import get_device_states, set_device_states
from torch.utils.data.distributed import DistributedSampler
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.file_utils import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
)
from transformers.trainer import Trainer, nested_detach
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollator 
from transformers.trainer_utils import PredictionOutput, EvalPrediction
import logging
import collections
import time
import math
import numpy as np
from sklearn.metrics import ndcg_score
from .FGM import FGM
logger = logging.getLogger(__name__)


class RerankerTrainer(Trainer):
    def __init__(
        self,
        model: Union[nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
    ):
        super(RerankerTrainer, self).__init__(model,args,data_collator,train_dataset,eval_dataset,tokenizer,model_init,compute_metrics,callbacks,optimizers)
        self.use_fgm = args.fgm
        if self.use_fgm:
            self.fgm = FGM(model)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            scaler = self.scaler if self.use_amp else None
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps, scaler=scaler)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        if self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()
        return loss.detach()
    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not hasattr(self.model, 'save_pretrained'):
            raise NotImplementedError(
                f'MODEL {self.model.__class__.__name__} '
                f'does not support save_pretrained interface')
        else:
            self.model.save_pretrained(output_dir)
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one :obj:`data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, dict):
            return type(data)(**{k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = dict(device=self.args.device)
            if self.deepspeed and data.dtype != torch.int64:
                # NLP models inputs are int64 and those get adjusted to the right dtype of the
                # embedding. Other models such as wav2vec2's inputs are already float and thus
                # may need special handling to match the dtypes of the model
                kwargs.update(dict(dtype=self.args.hf_deepspeed_config.dtype()))
            return data.to(**kwargs)
        return data
    def _get_train_sampler(self):
        if self.args.local_rank == -1:
            return RandomSampler(self.train_dataset)
        elif self.args.collaborative:
            logger.info(f'Collaborative Mode.')
            return SyncedSampler(self.train_dataset, seed=self.args.seed)
        else:
            return DistributedSampler(self.train_dataset)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        if self.args.warmup_ratio > 0:
            self.args.warmup_steps = num_training_steps * self.args.warmup_ratio

        return super(RerankerTrainer, self).create_optimizer_and_scheduler(num_training_steps)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` is a :obj:`torch.utils.data.IterableDataset`, a random sampler
        (adapted to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self._get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=True,
            num_workers=self.args.dataloader_num_workers,
        )
    def get_eval_dataloader(self,eval_dataset) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` is a :obj:`torch.utils.data.IterableDataset`, a random sampler
        (adapted to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            sampler=eval_sampler,
            collate_fn=self.data_collator,
            drop_last=False,
            num_workers=self.args.dataloader_num_workers,
        )
    def compute_loss(self, model: Reranker, inputs):
        return model(inputs)['loss']

    def compute_mrr(self,result):
        MRR10 = 0
        MRR100 = 0
        for res in result.values():
            sorted_res = sorted(res,key=lambda x:x[1],reverse=True)
            ar10 = 0
            ar100 = 0
            for index,ele in enumerate(sorted_res):
                if str(ele[2]) == '1':
                    if index < 100:
                        ar100 = 1.0 / (index + 1)
                    if index < 10:
                        ar10 = 1.0 / (index + 1)
                    break
            MRR10 += ar10
            MRR100 += ar100
        return {'eval_mrr10':MRR10 / len(result),'eval_mrr100':MRR100 / len(result)}
    
    def compute_ndcg(self,result):
        Ndcg = 0
        for res in result.values():
            sorted_res = sorted(res,key=lambda x:[1],reverse=True)
            labels = np.array([[ele[2] for ele in sorted_res]])
            scores = np.array([[ele[1] for ele in sorted_res]])
            ndcg = ndcg_score(labels, scores)
            ndcg = float(ndcg)
            Ndcg += ndcg
        Ndcg = Ndcg / len(result)
        # print('ndcg',Ndcg)
        return {'ndcg': float(Ndcg)} 
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        # memory metrics - must set up as early as possible

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=False, 
            ignore_keys=ignore_keys
            # metric_key_prefix=metric_key_prefix,
        )

        
        eval_scores = output.predictions[0]
        result = collections.defaultdict(list)
        
        for (qid,pid,score,prior,label) in zip(self.eval_qids,self.eval_pids,eval_scores,self.prior_scores,self.eval_labels):
            result[qid].append((pid,score,label))
        
        output.metrics.update(self.compute_mrr(result))
        # output.metrics.update(self.compute_ndcg(result))

        output.metrics.update({'steps':self.state.global_step})
        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        return output.metrics


    def emb_step():
        pass

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Tuple[Dict[str, Union[torch.Tensor, Any]]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor],Optional[torch.Tensor]]:

        # labels = inputs['labels'] if 'labels' in inputs else None
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            if self.args.fp16:
                with autocast():
                    outputs = model(inputs)
            else:
                outputs = model(inputs)

            loss = None
            if isinstance(outputs, dict):
                # logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                logits = outputs['logits']
                outputs['hidden_states'] = nested_detach(outputs['hidden_states'])
                hidden = outputs['hidden_states'][-1][:,0].contiguous().detach()
                logits = (logits,hidden)

            else:
                logits = outputs

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        # hidden = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        labels = None

        return (loss, logits, labels)
    
    def prediction_loop(
            self,
            *args,
            **kwargs
    ) -> PredictionOutput:
        pred_outs = super().evaluation_loop(*args, **kwargs)
        preds, label_ids, metrics = pred_outs.predictions, pred_outs.label_ids, pred_outs.metrics
        # preds = preds.squeeze()
        if self.compute_metrics is not None:
            metrics_no_label = self.compute_metrics(EvalPrediction(predictions=preds[0].squeeze(), label_ids=label_ids))
        else:
            metrics_no_label = {}

        for key in list(metrics_no_label.keys()):
            if not key.startswith("eval_"):
                metrics_no_label[f"eval_{key}"] = metrics_no_label.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics={**metrics, **metrics_no_label})

class RandContext:
    def __init__(self, *tensors):
        self.fwd_cpu_state = torch.get_rng_state()
        self.fwd_gpu_devices, self.fwd_gpu_states = get_device_states(*tensors)

    def __enter__(self):
        self._fork = torch.random.fork_rng(
            devices=self.fwd_gpu_devices,
            enabled=True
        )
        self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._fork.__exit__(exc_type, exc_val, exc_tb)
        self._fork = None

class RerankerDCTrainer(RerankerTrainer):
    def _chunk_input(self, inputs: Dict[str, torch.Tensor], chunk_size: int = None):
        if chunk_size is None:
            chunk_size = self.args.distance_cache_stride
        keys = list(inputs.keys())
        for k, v in inputs.items():
            inputs[k] = v.split(chunk_size)

        chunks = []
        n_chunks = len(inputs[keys[0]])

        for i in range(n_chunks):
            chunks.append({k: inputs[k][i] for k in keys})

        return chunks

    def training_step(self, model: RerankerDC, inputs):
        model.train()
        _model = getattr(model, 'module', model)
        inputs = self._prepare_inputs(inputs)

        rnd_states = []
        all_logits = []
        chunks = self._chunk_input(inputs)

        for chunk in chunks:
            rnd_states.append(RandContext())
            if self.args.fp16:
                with torch.no_grad():
                    with autocast():
                        chunk_logits = model(chunk)
            else:
                with torch.no_grad():
                    chunk_logits = model(chunk)
            all_logits.append(chunk_logits)

        all_logits = torch.cat(all_logits).float()
        loss, grads = _model.compute_grad(all_logits)
        grads = grads.view(-1, self.args.distance_cache_stride)

        for chunk_id, chunk in enumerate(chunks):
            with rnd_states[chunk_id]:
                if self.args.fp16:
                    with autocast():
                        surrogate = model(chunk, grads[chunk_id])
                else:
                    surrogate = model(chunk, grads[chunk_id])

            if self.args.gradient_accumulation_steps > 1:
                surrogate = surrogate / self.args.gradient_accumulation_steps

            if self.args.fp16:
                self.scaler.scale(surrogate).backward()
            else:
                surrogate.backward()

        return loss.detach()
