# Copyright 2021 Reranker Author. All rights reserved.
# Code structure inspired by HuggingFace run_glue.py in the transformers library.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

from src import Reranker
from src import RerankerTrainer
from src.data import GroupedTrainDataset, PredictionDataset, GroupCollator
from src.arguments import ModelArguments, DataArguments, \
    RerankerTrainingArguments as TrainingArguments

from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)
import torch

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set seed
    set_seed(training_args.seed)

    num_labels = 1

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    if training_args.do_train:
        config.emb_size = model_args.emb_size
        config.hidden_size = model_args.hidden_size
        config.num_hidden_layers = model_args.num_hidden_layers
        config.num_attention_heads = model_args.num_attention_heads
        config.concat_startegy = model_args.concat_startegy
    model = Reranker(config,model_args)

    # Get datasets
    if training_args.do_train:
        train_dataset = GroupedTrainDataset(
            data_args, data_args.train_path, train_args=training_args
        )
        dev_dataset = PredictionDataset(
            data_args.dev_path,
            data_args.dev_id_file
        )
        
    else:
        train_dataset = None
        dev_dataset = None


    # Initialize our Trainer
    # _trainer_class = RerankerTrainer
    trainer = RerankerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=GroupCollator(),
    )       
    # Training

    if training_args.do_train:
        print('train_batch_size',training_args.train_batch_size)
        if training_args.continue_train_checkpoint is not None:
            trainer.train(training_args.continue_train_checkpoint)
        else:
            trainer.train()
        trainer.save_model()
        
    if training_args.do_eval:
        trainer.evaluate()

    if training_args.do_predict:
        logging.info("*** Prediction ***")
        state_dict = torch.load(model_args.model_name_or_path+'/pytorch_model.bin')
        model.load_state_dict(state_dict)
        if os.path.exists(data_args.rank_score_path):
            if os.path.isfile(data_args.rank_score_path):
                raise FileExistsError(f'score file {data_args.rank_score_path} already exists')
            else:
                raise ValueError(f'Should specify a file name')
        else:
            score_dir = os.path.split(data_args.rank_score_path)[0]
            if not os.path.exists(score_dir):
                logger.info(f'Creating score directory {score_dir}')
                os.makedirs(score_dir)

        test_dataset = PredictionDataset(
            data_args.pred_path,
            data_args.pred_id_file
        )
        # assert data_args.pred_id_file is not None


        eval_dataloader = trainer.get_eval_dataloader(test_dataset)

        output = trainer.prediction_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=False, 
        )
        eval_scores = output.predictions
        eval_lis = test_dataset.eval_lis
        if trainer.is_world_process_zero():
            with open(data_args.rank_score_path, "w") as writer:
                for (qid,score_list) in zip(eval_lis,eval_scores):
                    for i,(psg,score) in enumerate(zip(eval_lis[qid],score_list)):
                        writer.write(f'{qid} Q0 {psg[0]} {i} {score} {data_args.run_id}\n')
                

def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()
