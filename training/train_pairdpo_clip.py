# coding=utf-8
# Copyright 2024 HuggingFace, NUS Show Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Union
from copy import deepcopy

import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import wandb
import torch
from torch.optim import AdamW
from lightning.pytorch.utilities import CombinedLoader

from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed

from data import UniDPODatasetGeneration, create_dataloader_generation

from models import Showo, MAGVITv2, get_mask_chedule
from prompting_utils import UniversalPrompting, create_attention_mask_predict_next, create_attention_mask_for_mmu
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from llava.llava_data_vq_unified import get_unidpo_understanding_data_loader



SYSTEM_PROMPT_LEN = 28

from utils import get_config, flatten_omega_conf, mask_or_random_replace_tokens_dpo, AverageMeter

try:
    import apex

    is_apex_available = True
except ImportError:
    is_apex_available = False

logger = get_logger(__name__, log_level="INFO")


def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    else:
        raise ValueError(f"model_type {model_type} not supported.")


def main():
    #########################
    # SETUP Accelerator     #
    #########################
    config = get_config()

    # Enable TF32 on Ampere GPUs
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(config.experiment.output_dir) / "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=config.experiment.logging_dir,
        split_batches=True,
    )
    total_batch_size_per_gpu = config.training.batch_size_t2i

    total_batch_size = (
            config.training.batch_size_t2i * accelerator.num_processes * config.training.gradient_accumulation_steps
    )

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (
            total_batch_size_per_gpu
        )

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        resume_wandb_run = config.wandb.resume
        run_id = config.wandb.get("run_id", None)
        if run_id is None:
            resume_wandb_run = False
            run_id = wandb.util.generate_id()
            config.wandb.run_id = run_id

        wandb_init_kwargs = dict(
            name=config.experiment.name,
            id=run_id,
            resume=resume_wandb_run,
            entity=config.wandb.get("entity", None),
            config_exclude_keys=[],
        )
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb_config.pop("experiment.resume_from_checkpoint")

        accelerator.init_trackers(
            config.experiment.project,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    if accelerator.is_main_process:
        os.makedirs(config.experiment.output_dir, exist_ok=True)
        config_path = Path(config.experiment.output_dir) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed)

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    logger.info("Loading models and optimizer")

    tokenizer = AutoTokenizer.from_pretrained(config.model.showo.llm_model_path, padding_side="left")

    # unified prompting for show-o
    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                       special_tokens=(
                                           "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>",
                                           "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"
                                       ),
                                       ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob)

    print('special tokens : \n', uni_prompting.sptids_dict)

    # VQ model for processing image into discrete tokens
    vq_model = get_vq_model_class(config.model.vq_model.type)
    if config.model.vq_model.get("pretrained_model_path", None):
        vq_model = vq_model().to(accelerator.device)
        state_dict = torch.load(config.model.vq_model.pretrained_model_path)['model']
        vq_model.load_state_dict(state_dict)
    else:
        vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name).to(accelerator.device)
    vq_model.eval()
    vq_model.requires_grad_(False)

    # Initialize Show-o model
    if config.model.showo.load_from_showo:
        model = Showo.from_pretrained(config.model.showo.pretrained_model_path).to(accelerator.device)
        if config.model.showo.vocab_size != model.vocab_size:
            model.showo.resize_token_embeddings(config.model.showo.vocab_size)
            model.config.codebook_size = config.model.showo.codebook_size
            model.config.vocab_size = config.model.showo.vocab_size
            model.vocab_size = config.model.showo.vocab_size
            model.output_size = config.model.showo.vocab_size
            model.config.mask_token_id = config.model.showo.vocab_size - 1
            model.mask_token_id = config.model.showo.vocab_size - 1
    else:
        model = Showo(**config.model.showo).to(accelerator.device)
    reference_model = deepcopy(model)
    reference_model.eval()
    reference_model.requires_grad_(False)
    mask_id = model.mask_token_id # 58497

    ##################################
    #   Optimizer and LR scheduler   #
    #################################
    optimizer_config = config.optimizer.params

    # no decay on bias and layernorm and embedding
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    # Create mask scheduler
    if config.get("mask_schedule", None) is not None:
        schedule = config.mask_schedule.schedule
        args = config.mask_schedule.get("params", {})
        mask_schedule = get_mask_chedule(schedule, **args)
    else:
        mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
    )

    ##################################
    #         DATALOADER             #
    #################################
    logger.info("Creating dataloaders and lr_scheduler")

    total_batch_size_t2i_without_accum = config.training.batch_size_t2i * accelerator.num_processes

    # DataLoaders creation:
    # We use webdataset for data loading. The dataloaders are created with sampling with replacement.
    # We don't do dataset resuming here, instead we resample the shards and buffer each time. The sampling is stochastic.
    # This means that the dataloading is not deterministic, but it's fast and efficient.
    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params

    # Data for generation
    dataset_generation = UniDPODatasetGeneration(
        data_path=dataset_config.dpo_data_path,
        tokenizer=None,  
        max_seq_length=preproc_config.max_seq_length,
        num_train_examples=config.experiment.max_train_examples_t2i,  # 你实际的训练样本数
        per_gpu_batch_size=config.training.batch_size_t2i,
        global_batch_size=total_batch_size_t2i_without_accum,
        num_workers=dataset_config.num_workers,
        resolution=preproc_config.resolution,
        shuffle_buffer_size=dataset_config.shuffle_buffer_size,
    )
    train_dataloader_t2i = create_dataloader_generation(
        dataset=dataset_generation,
        batch_size=config.training.batch_size_t2i,  # 每个GPU的batch size
        num_workers=dataset_config.num_workers,  # 工作线程数量
    )
    num_update_steps_per_epoch = math.ceil(
        dataset_generation.num_batches / config.training.gradient_accumulation_steps)
    num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)
    # Data for understanding
    train_dataloader_mmu = get_unidpo_understanding_data_loader(
        tokenizer,
        batch_size=config.training.batch_size_mmu,
        num_workers=dataset_config.num_workers,
        world_size=accelerator.num_processes,
        local_rank=accelerator.process_index,
        max_length=preproc_config.max_seq_length if config.dataset.add_system_prompt else preproc_config.max_seq_length + SYSTEM_PROMPT_LEN,
        phase="tuning"
    )
   
    # Combine these dataloaders into a single iterable model
    iterables = {
        "t2i_flow": train_dataloader_t2i,
        "mmu_flow": train_dataloader_mmu,
    }

    combined_dataloader = CombinedLoader(iterables, mode=config.dataset.combined_loader_mode)

    ##################################
    #         MODEL RESUME          #
    #################################
    global_step = 0
    first_epoch = 0

    if config.experiment.resume_from_checkpoint:
        dirs = os.listdir(config.experiment.output_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None
        if path is not None:
            path = os.path.join(config.experiment.output_dir, path)

            global_step = int(os.path.basename(path).split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch

            accelerator.print(f"Resuming from checkpoint {path}/unwrapped_model/pytorch_model.bin")
            state_dict = torch.load(f'{path}/unwrapped_model/pytorch_model.bin', map_location="cpu")
            model.load_state_dict(state_dict, strict=True)
            del state_dict

    ##################################
    #       Prepare accelerator     #
    #################################
    logger.info("Preparing model, optimizer and dataloaders")
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    vq_model.to(device=accelerator.device)
    reference_model.to(device=accelerator.device)

    if hasattr(model, 'module'):
        mask_dtype = model.module.showo.model.embed_tokens.weight.dtype
    else:
        mask_dtype = model.showo.model.embed_tokens.weight.dtype

    ##################################
    #             Training          #
    #################################
    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {total_batch_size_per_gpu}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")

    @torch.no_grad()
    def prepare_inputs_and_labels(
            pixel_values_or_image_ids_win: Union[torch.FloatTensor, torch.LongTensor],
            pixel_values_or_image_ids_lose: Union[torch.FloatTensor, torch.LongTensor],
            texts: Union[str, str],
            min_masking_rate: float = 0.0,
            is_train: bool = True,
    ):

        image_tokens_win = vq_model.get_code(pixel_values_or_image_ids_win)
        image_tokens_win = image_tokens_win + len(uni_prompting.text_tokenizer)
        image_tokens_lose = vq_model.get_code(pixel_values_or_image_ids_lose)
        image_tokens_lose = image_tokens_lose + len(uni_prompting.text_tokenizer)

        # create MLM mask and labels
        input_ids_win, input_ids_lose, labels_win, labels_lose, loss_weight, mask_prob = mask_or_random_replace_tokens_dpo(
            image_tokens_win,
            image_tokens_lose,
            mask_id,
            config,
            mask_schedule=mask_schedule,
            is_train=is_train,
        )
        input_ids_win, _, labels_win = uni_prompting((texts, input_ids_win, labels_win), 't2i')
        input_ids_lose, _, labels_lose = uni_prompting((texts, input_ids_lose, labels_lose), 't2i')

        return input_ids_win, input_ids_lose, labels_win, labels_lose, mask_prob, image_tokens_win, image_tokens_lose

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        for batch, batch_idx, dataloader_idx in combined_dataloader:
            # for loss calculation
            batch_size_t2i = batch["t2i_flow"]["win_image"].shape[0]

            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            # Build formatted sequences for class-conditional/text-to-image generation
            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            pixel_values_win, pixel_values_lose, texts = batch["t2i_flow"]["win_image"], batch["t2i_flow"]["lose_image"], batch["t2i_flow"]["input_ids"]
            pixel_values_win = pixel_values_win.to(accelerator.device, non_blocking=True)
            pixel_values_lose = pixel_values_lose.to(accelerator.device, non_blocking=True)
            data_time_m.update(time.time() - end)

            # Encode images to image tokens, mask them and create input and labels
            (
                input_ids_win,
                input_ids_lose,
                labels_win,
                labels_lose,
                mask_prob,
                image_tokens_ori_win,
                image_tokens_ori_lose,
            ) = prepare_inputs_and_labels(pixel_values_win, pixel_values_lose, texts, config.training.min_masking_rate)
            attention_mask_win = create_attention_mask_predict_next(input_ids_win,
                                                                pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                                soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                                eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                                rm_pad_in_image=True,
                                                                return_inverse_mask=True)
            attention_mask_win = attention_mask_win.to(mask_dtype)
            attention_mask_lose = create_attention_mask_predict_next(input_ids_lose,
                                                                pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                                soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                                eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                                rm_pad_in_image=True,
                                                                return_inverse_mask=True)
            attention_mask_lose = attention_mask_lose.to(mask_dtype)


            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            # Build formatted sequences for multimodal understanding
            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*

            pixel_values_mmu, input_ids_mmu_lose, labels_mmu_lose, input_ids_mmu_win, labels_mmu_win = (
                batch["mmu_flow"]["images"],
                batch["mmu_flow"]["input_ids_lose"],
                batch["mmu_flow"]["labels_lose"],
                batch["mmu_flow"]["input_ids_win"],
                batch["mmu_flow"]["labels_win"],
            )

            pixel_values_mmu = pixel_values_mmu.to(accelerator.device, non_blocking=True)
            input_ids_mmu_lose = input_ids_mmu_lose.to(accelerator.device, non_blocking=True)

            input_ids_mmu_win = input_ids_mmu_win.to(accelerator.device, non_blocking=True)

            image_tokens_mmu = vq_model.get_code(pixel_values_mmu)
            image_tokens_mmu = image_tokens_mmu + len(uni_prompting.text_tokenizer)

            input_ids_mmu_lose = torch.cat([
                (torch.ones(input_ids_mmu_lose.shape[0], 1) * uni_prompting.sptids_dict['<|mmu|>']).to(accelerator.device),
                (torch.ones(input_ids_mmu_lose.shape[0], 1) * uni_prompting.sptids_dict['<|soi|>']).to(accelerator.device),
                image_tokens_mmu,
                (torch.ones(input_ids_mmu_lose.shape[0], 1) * uni_prompting.sptids_dict['<|eoi|>']).to(accelerator.device),
                input_ids_mmu_lose,
            ], dim=1).long()

            labels_mmu_lose = torch.cat([
                (torch.ones(input_ids_mmu_lose.shape[0], 1) * uni_prompting.ignore_id).to(accelerator.device),
                (torch.ones(input_ids_mmu_lose.shape[0], 1) * uni_prompting.ignore_id).to(accelerator.device),
                torch.ones_like(image_tokens_mmu) * uni_prompting.ignore_id,
                (torch.ones(input_ids_mmu_lose.shape[0], 1) * uni_prompting.ignore_id).to(accelerator.device),
                labels_mmu_lose.to(accelerator.device),
            ], dim=1).long()

            input_ids_mmu_win = torch.cat([
                (torch.ones(input_ids_mmu_win.shape[0], 1) * uni_prompting.sptids_dict['<|mmu|>']).to(accelerator.device),
                (torch.ones(input_ids_mmu_win.shape[0], 1) * uni_prompting.sptids_dict['<|soi|>']).to(accelerator.device),
                image_tokens_mmu,
                (torch.ones(input_ids_mmu_win.shape[0], 1) * uni_prompting.sptids_dict['<|eoi|>']).to(accelerator.device),
                input_ids_mmu_win,
            ], dim=1).long()

            labels_mmu_win = torch.cat([
                (torch.ones(input_ids_mmu_win.shape[0], 1) * uni_prompting.ignore_id).to(accelerator.device),
                (torch.ones(input_ids_mmu_win.shape[0], 1) * uni_prompting.ignore_id).to(accelerator.device),
                torch.ones_like(image_tokens_mmu) * uni_prompting.ignore_id,
                (torch.ones(input_ids_mmu_win.shape[0], 1) * uni_prompting.ignore_id).to(accelerator.device),
                labels_mmu_win.to(accelerator.device),
            ], dim=1).long()

            attention_mask_mmu_lose = create_attention_mask_for_mmu(
                input_ids_mmu_lose.to(input_ids_mmu_lose.device),
                eoi_id=int(uni_prompting.sptids_dict['<|eoi|>'])
            )
            attention_mask_mmu_lose = attention_mask_mmu_lose.to(mask_dtype)

            attention_mask_mmu_win = create_attention_mask_for_mmu(
                input_ids_mmu_win.to(input_ids_mmu_win.device),
                eoi_id=int(uni_prompting.sptids_dict['<|eoi|>'])
            )
            attention_mask_mmu_win = attention_mask_mmu_win.to(mask_dtype)

            attention_mask_lose = torch.cat([attention_mask_lose, attention_mask_mmu_lose], dim=0)
            attention_mask_win = torch.cat([attention_mask_win, attention_mask_mmu_win], dim=0)
            input_ids_lose = torch.cat([input_ids_lose, input_ids_mmu_lose.to(input_ids_lose.device)], dim=0)
            input_ids_win = torch.cat([input_ids_win, input_ids_mmu_win.to(input_ids_win.device)], dim=0)
            labels_lose = torch.cat([labels_lose, labels_mmu_lose.to(input_ids_lose.device)], dim=0)
            labels_win = torch.cat([labels_win, labels_mmu_win.to(input_ids_lose.device)], dim=0)

            if global_step == 0 and epoch == 0:
                logger.info("Input ids win: {}".format(input_ids_win))
                logger.info("Input ids lose: {}".format(input_ids_lose))
                logger.info("Labels win: {}".format(labels_win))
                logger.info("Labels lose: {}".format(labels_lose))

            with accelerator.accumulate(model):
                log_probs_sum_win_t2i, log_probs_sum_win_mmu = model.loss_for_unidpo(
                    input_ids=input_ids_win,
                    input_embeddings=None,
                    attention_mask=attention_mask_win,
                    labels=labels_win,
                    label_smoothing=config.training.label_smoothing,
                    batch_size_t2i=batch_size_t2i,
                    max_seq_length=config.dataset.preprocessing.max_seq_length,
                )
                log_probs_sum_lose_t2i, log_probs_sum_lose_mmu = model.loss_for_unidpo(
                    input_ids=input_ids_lose,
                    input_embeddings=None,
                    attention_mask=attention_mask_lose,
                    labels=labels_lose,
                    label_smoothing=config.training.label_smoothing,
                    batch_size_t2i=batch_size_t2i,
                    max_seq_length=config.dataset.preprocessing.max_seq_length,
                )
                with torch.no_grad():
                    log_probs_sum_win_reference_t2i, log_probs_sum_win_reference_mmu = reference_model.loss_for_unidpo(
                        input_ids=input_ids_win,
                        input_embeddings=None,
                        attention_mask=attention_mask_win,
                        labels=labels_win,
                        label_smoothing=config.training.label_smoothing,
                        batch_size_t2i=batch_size_t2i,
                        max_seq_length=config.dataset.preprocessing.max_seq_length,
                    )
                    log_probs_sum_lose_reference_t2i, log_probs_sum_lose_reference_mmu = reference_model.loss_for_unidpo(
                        input_ids=input_ids_lose,
                        input_embeddings=None,
                        attention_mask=attention_mask_lose,
                        labels=labels_lose,
                        label_smoothing=config.training.label_smoothing,
                        batch_size_t2i=batch_size_t2i,
                        max_seq_length=config.dataset.preprocessing.max_seq_length,
                    )
                delta_win_t2i = log_probs_sum_win_t2i - log_probs_sum_win_reference_t2i
                delta_lose_t2i= log_probs_sum_lose_t2i - log_probs_sum_lose_reference_t2i

                delta_win_mmu = log_probs_sum_win_mmu - log_probs_sum_win_reference_mmu
                delta_lose_mmu = log_probs_sum_lose_mmu - log_probs_sum_lose_reference_mmu

                dpo_loss_t2i = -torch.log(torch.sigmoid(config.training.beta * (delta_win_t2i - delta_lose_t2i))).mean()
                dpo_loss_mmu = -torch.log(torch.sigmoid(config.training.beta * (delta_win_mmu - delta_lose_mmu))).mean()

                avg_loss_t2i = accelerator.gather(dpo_loss_t2i.repeat(config.training.batch_size_t2i)).mean()
                avg_loss_mmu = accelerator.gather(dpo_loss_mmu.repeat(config.training.batch_size_mmu)).mean()
                loss = config.training.t2i_coeff * dpo_loss_t2i + config.training.mmu_coeff * dpo_loss_mmu

                avg_masking_rate = accelerator.gather(mask_prob.repeat(config.training.batch_size_t2i)).mean()

                accelerator.backward(loss)

                if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()

                # log gradient norm before zeroing it
                if (
                        accelerator.sync_gradients
                        and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                        and accelerator.is_main_process
                ):
                    log_grad_norm(model, accelerator, global_step + 1)

                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:

                batch_time_m.update(time.time() - end)
                end = time.time()

                # Log metrics
                if (global_step + 1) % config.experiment.log_every == 0:
                    samples_per_second_per_gpu = (
                            config.training.gradient_accumulation_steps * total_batch_size_per_gpu / batch_time_m.val
                    )
                    logs = {
                        "step_loss_t2i": avg_loss_t2i.item(),
                        "step_loss_mmu": avg_loss_mmu.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "avg_masking_rate": avg_masking_rate.item(),
                        "samples/sec/gpu": samples_per_second_per_gpu,
                        "data_time": data_time_m.val,
                        "batch_time": batch_time_m.val,
                    }
                    accelerator.log(logs, step=global_step + 1)

                    logger.info(
                        f"Step: {global_step + 1} "
                        f"Loss: {loss.item():0.4f} "
                        f"Data (t): {data_time_m.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                        f"Batch (t): {batch_time_m.val:0.4f} "
                        f"LR: {lr_scheduler.get_last_lr()[0]:0.6f}"
                    )

                    # resetting batch / data time meters per log window
                    batch_time_m.reset()
                    data_time_m.reset()

                # Save model checkpoint
                if (global_step + 1) % config.experiment.save_every == 0:
                    save_checkpoint(model, config, accelerator, global_step + 1)

                if (global_step + 1) % config.experiment.generate_every == 0 and accelerator.is_main_process:
                    generate_images(
                        model,
                        vq_model,
                        uni_prompting,
                        accelerator,
                        config,
                        global_step + 1,
                        mask_schedule=mask_schedule,
                    )

                    # visualize_predictions(
                    #     model,
                    #     vq_model,
                    #     uni_prompting,
                    #     config,
                    #     global_step + 1,
                    #     input_ids_win,
                    #     image_tokens_ori_win,
                    #     batch["t2i_flow"]["images"],
                    #     texts,
                    #     logits_win,
                    # )

                global_step += 1

            # Stop training if max steps is reached
            if global_step >= config.training.max_train_steps:
                break
            # End for

    accelerator.wait_for_everyone()

    # Evaluate and save checkpoint at the end of training
    save_checkpoint(model, config, accelerator, global_step)

    # Save the final trained checkpoint
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        model.save_pretrained(config.experiment.output_dir, safe_serialization=False)

    accelerator.end_training()


@torch.no_grad()
def visualize_predictions(
        model,
        vq_model,
        uni_prompting,
        config,
        global_step,
        input_ids,
        image_tokens_ori,
        ori_images,
        texts,
        logits,
):
    logger.info("Visualizing predictions...")
    model.eval()

    recons_images = vq_model.decode_code(image_tokens_ori - len(uni_prompting.text_tokenizer))
    recons_images = torch.clamp((recons_images + 1.0) / 2.0, min=0.0, max=1.0)
    recons_images *= 255.0
    recons_images = recons_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

    images = torch.clamp((ori_images + 1.0) / 2.0, min=0.0, max=1.0)
    images *= 255.0
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

    predictions = logits[:config.training.batch_size_t2i, -(config.model.showo.num_vq_tokens + 1):-1:,
                  config.model.showo.llm_vocab_size + config.model.showo.num_new_special_tokens:-1]
    predictions = predictions.argmax(axis=-1)

    mask_token_id = config.model.showo.vocab_size - 1 - len(uni_prompting.text_tokenizer)
    input_ids = input_ids[:config.training.batch_size_t2i, -(config.model.showo.num_vq_tokens + 1):-1:] - len(
        uni_prompting.text_tokenizer)
    mask_ratio = list((torch.where(input_ids == mask_token_id, 1, 0).sum(
        dim=-1) / config.model.showo.num_vq_tokens).cpu().numpy())
    predicted_images = torch.where(input_ids == mask_token_id, predictions, input_ids)

    predicted_images = vq_model.decode_code(predicted_images)
    predicted_images = torch.clamp((predicted_images + 1.0) / 2.0, min=0.0, max=1.0)
    predicted_images *= 255.0
    predicted_images = predicted_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    predicted_images = np.concatenate((images, recons_images, predicted_images), 2)
    pil_images = [Image.fromarray(image) for image in predicted_images]

    # Log images
    wandb_images = [wandb.Image(image, caption=f'mask ratio: {r:0.2f} \n caption: {texts[i]}') for i, (image, r) in
                    enumerate(zip(pil_images, mask_ratio))]
    wandb.log({"Original images v.s. Reconstructed images v.s. Predicted images": wandb_images}, step=global_step)

    model.train()


@torch.no_grad()
def generate_images(
        model,
        vq_model,
        uni_prompting,
        accelerator,
        config,
        global_step,
        mask_schedule,
):
    logger.info("Generating images...")
    model.eval()

    # read validation prompts from file
    with open(config.dataset.params.validation_prompts_file, "r") as f:
        validation_prompts = f.read().splitlines()

    if hasattr(model, 'module'):
        mask_dtype = model.module.showo.model.embed_tokens.weight.dtype
    else:
        mask_dtype = model.showo.model.embed_tokens.weight.dtype

    mask_token_id = config.model.showo.vocab_size - 1
    image_tokens = torch.ones((len(validation_prompts), config.model.showo.num_vq_tokens), dtype=torch.long,
                              device=accelerator.device) * mask_token_id
    input_ids, _ = uni_prompting((validation_prompts, image_tokens), 't2i_gen')
    if config.training.guidance_scale > 0:
        uncond_input_ids, _ = uni_prompting(([''] * len(validation_prompts), image_tokens), 't2i_gen')
        attention_mask = create_attention_mask_predict_next(torch.cat([input_ids, uncond_input_ids], dim=0),
                                                            pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                            soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                            eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                            rm_pad_in_image=True).to(mask_dtype)
    else:
        attention_mask = create_attention_mask_predict_next(input_ids,
                                                            pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                            soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                            eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                            rm_pad_in_image=True).to(mask_dtype)
        uncond_input_ids = None

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    with torch.autocast("cuda", dtype=weight_dtype, enabled=accelerator.mixed_precision != "no"):
        # Generate images
        gen_token_ids = accelerator.unwrap_model(model).t2i_generate(
            input_ids=input_ids,
            uncond_input_ids=uncond_input_ids,
            attention_mask=attention_mask,
            guidance_scale=config.training.guidance_scale,
            temperature=config.training.get("generation_temperature", 1.0),
            timesteps=config.training.generation_timesteps,
            noise_schedule=mask_schedule,
            noise_type=config.training.get("noise_type", "mask"),
            predict_all_tokens=config.training.get("predict_all_tokens", False),
            seq_len=config.model.showo.num_vq_tokens,
            uni_prompting=uni_prompting,
            config=config,
        )
    # In the beginning of training, the model is not fully trained and the generated token ids can be out of range
    # so we clamp them to the correct range.
    gen_token_ids = torch.clamp(gen_token_ids, max=accelerator.unwrap_model(model).config.codebook_size - 1, min=0)
    images = vq_model.decode_code(gen_token_ids)

    model.train()

    if config.training.get("pre_encode", False):
        del vq_model

    # Convert to PIL images
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
    images *= 255.0
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    pil_images = [Image.fromarray(image) for image in images]

    # Log images
    wandb_images = [wandb.Image(image, caption=validation_prompts[i]) for i, image in enumerate(pil_images)]
    wandb.log({"Generated images": wandb_images}, step=global_step)


def save_checkpoint(model, config, accelerator, global_step):
    output_dir = config.experiment.output_dir
    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)

    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    if accelerator.is_main_process and checkpoints_total_limit is not None:
        checkpoints = os.listdir(output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
        if len(checkpoints) >= checkpoints_total_limit:
            num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)

    save_path = Path(output_dir) / f"checkpoint-{global_step}"

    # retrieve the model on all processes for deepspeed stage 3 to work then save on one process (we are not using stage 3 yet)
    # XXX: could also make this conditional on deepspeed
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            save_path / "unwrapped_model",
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=False
        )
        json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
        logger.info(f"Saved state to {save_path}")


def log_grad_norm(model, accelerator, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)


if __name__ == "__main__":
    main()
