# Copyright 2021 Reranker Author. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
import math
import torch
import torch.nn.functional as F
import copy
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPooling
from torch import nn
import torch.distributed as dist
from transformers.models.bert.modeling_bert import BertEncoder
from .arguments import ModelArguments, DataArguments, \
    RerankerTrainingArguments as TrainingArguments
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch import Tensor, device, nn
import os
from transformers.utils import logging
from transformers.file_utils import  WEIGHTS_NAME
logger = logging.get_logger(__name__)

def get_extended_attention_mask(attention_mask: Tensor, input_shape: Tuple[int], device: device) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.
        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.
            device: (`torch.device`):
                The device of the input to the model.
        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        extended_attention_mask = extended_attention_mask.to(attention_mask.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).
    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model

def get_parameter_dtype(parameter: Union[nn.Module]):
    try:
        return next(parameter.parameters()).dtype
    except StopIteration:

        def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].dtype

class Reranker(nn.Module):
    def __init__(self,config,model_args):
        super().__init__()
        self.config = config
        self.model_args = model_args
        self.model = BertEncoder(config)
        self.emb_layer = nn.Linear(config.emb_size,config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.linear = nn.Linear(config.hidden_size, 1)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.alpha = torch.nn.Parameter(torch.Tensor([0.1]))

    def emb(self,inputs):
        input_shape = inputs.size()
        seq_length = input_shape[1]
        inputs = self.emb_layer(inputs)
        position_ids = self.position_ids[:,: seq_length]
        position_emb = self.position_embeddings(position_ids)
        embs = inputs + position_emb
        embs = self.LayerNorm(embs)
        embs = self.dropout(embs)
        return embs

    def forward(self,inputs,scores=None,attention_mask=None,labels=None):
        # print(inputs.keys())

        embs = self.emb(inputs)
        input_shape = embs.size()
        seq_length = input_shape[1]
        batch_size = input_shape[0]
        device = inputs.device
        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)
        attention_mask = get_extended_attention_mask(attention_mask,input_shape,device)
        output = self.model(embs,attention_mask=attention_mask)
        logits = output['last_hidden_state'] # B*Seq*Emb

        logits = self.linear(logits).squeeze(-1) # B*Seq*1
        dic = {}
        dic['logits'] = logits
        if labels is not None:
            labels = labels.double()
            loss = self.cross_entropy(logits, labels)
            dic['loss'] = loss
        return dic


    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        save_config: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        **kwargs,
    ):
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        `[`~PreTrainedModel.from_pretrained`]` class method.
        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
            save_config (`bool`, *optional*, defaults to `True`):
                Whether or not to save the config of the model. Useful when in distributed training like TPUs and need
                to call this function on all processes. In this case, set `save_config=True` only on the main process
                to avoid race conditions.
            state_dict (nested dictionary of `torch.Tensor`):
                The state dictionary of the model to save. Will default to `self.state_dict()`, but can be used to only
                save parts of the model or if special precautions need to be taken when recovering the state dictionary
                of a model (like when using model parallelism).
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful on distributed training like TPUs when one
                need to replace `torch.save` by another method.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it.
                <Tip warning={true}>
                Using `push_to_hub=True` will synchronize the repository you are pushing to with `save_directory`,
                which requires `save_directory` to be a local clone of the repo you are pushing to if it's an existing
                folder. Pass along `temp_dir=True` to use a temporary directory instead.
                </Tip>
            kwargs:
                Additional key word arguments passed along to the [`~file_utils.PushToHubMixin.push_to_hub`] method.
        """
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        # Only save the model itself if we are using distributed training
        model_to_save = unwrap_model(self)

        # save the string version of dtype to the config, e.g. convert torch.float32 => "float32"
        # we currently don't use this setting automatically, but may start to use with v5
        dtype = get_parameter_dtype(model_to_save)
        model_to_save.config.torch_dtype = str(dtype).split(".")[1]

        # Attach architecture to the config
        model_to_save.config.architectures = [model_to_save.__class__.__name__]

        # Save the config
        if save_config:
            model_to_save.config.save_pretrained(save_directory)

        # Save the model
        if state_dict is None:
            state_dict = model_to_save.state_dict()

        # Handle the case where some state_dict keys shouldn't be saved
        '''
        if self._keys_to_ignore_on_save is not None:
            for ignore_key in self._keys_to_ignore_on_save:
                if ignore_key in state_dict.keys():
                    del state_dict[ignore_key]
        '''
        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        save_function(state_dict, output_model_file)

        logger.info(f"Model weights saved in {output_model_file}")

