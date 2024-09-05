import os
import logging
import tempfile
from copy import deepcopy

import torch
import torch.nn as nn
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from transformers import AutoConfig, AutoModelForCausalLM
from composer import Event, State, Trainer, Callback
from composer.utils import dist
from composer.loggers import Logger
from llmfoundry.utils.builders import (
    build_algorithm,
    build_callback,
    build_logger,
    build_optimizer,
    build_scheduler,
)
from llmfoundry.utils.config_utils import pop_config, log_config
from llmfoundry.models.utils import init_empty_weights
from llmfoundry.models.hf.hf_fsdp import hf_get_init_device
from omegaconf import OmegaConf as om
from transformers import AutoTokenizer
import fsspec


log = logging.getLogger(__name__)

#####################################################
## MISC
#####################################################

def build_tokenizer(tokenizer_name):
    
    if tokenizer_name == "meta-llama/Meta-Llama-3-8B":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return tokenizer


#####################################################
## Callbacks
#####################################################

# Code taken from: https://github.com/mosaicml/llm-foundry/blob/main/llmfoundry/callbacks/hf_checkpointer.py
class HFCheckpointer(Callback):

    def __init__(self, save_base, module_name):
        self.module_name = module_name
        self.dtype = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
        }['bfloat16']
        self.save_name = os.path.join(save_base, "hf")

    def run_event(self, event: Event, state: State, logger: Logger) -> None:
        if event == Event.FIT_END:
            self.save_checkpoint(state)
    
    def save_checkpoint(self, state: State):

        log.info('Saving HuggingFace formatted checkpoint')

        log.debug('Gathering state dict')
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        if state.is_model_ddp:
            original_model = getattr(state.model.module, self.module_name)
            state_dict_model = getattr(state.model.module, self.module_name)
            original_tokenizer = state.model.module.tokenizer
        elif isinstance(getattr(state.model, self.module_name), FSDP): 
            original_model = getattr(state.model, self.module_name).module
            state_dict_model = getattr(state.model, self.module_name)
            original_tokenizer = state.model.tokenizer
        else:
            original_model = getattr(state.model, self.module_name)
            state_dict_model = getattr(state.model, self.module_name)
            original_tokenizer = state.model.tokenizer


        cpu_offload = True

        # Add a dtensor->cpu tensor hook to avoid CUDA OOM
        def dtensor_to_tensor_hook(
            module: nn.Module,
            state_dict,
            prefix: str,
            *args,
        ):
            dtensor_fqns = []
            for fqn in state_dict.keys():
                tensor = state_dict[fqn]
                if isinstance(tensor, DTensor):
                    dtensor_fqns.append(fqn)
                    tensor = tensor.full_tensor()  # type: ignore
                    if dist.get_global_rank() == 0:
                        if cpu_offload:
                            tensor = tensor.cpu()
                        state_dict[fqn] = tensor
            if dist.get_global_rank() != 0:
                for fqn in dtensor_fqns:
                    del state_dict[fqn]
            return state_dict

        hooks = []
        for _, module in state_dict_model.named_modules():
            if isinstance(module, FSDP):
                hooks.append(
                    module.
                    _register_state_dict_hook(dtensor_to_tensor_hook),
                )

        state_dict = get_model_state_dict(
            state_dict_model,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=cpu_offload,
            ),
        )
        for hook in hooks:
            hook.remove()

        # Convert the state dict to the requested precis
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                state_dict[k] = v.to(dtype=self.dtype)

        new_model_instance = None  # Need this for pyright because variable could be unbound

        if dist.get_global_rank() == 0:
            log.debug('Saving Hugging Face checkpoint in global rank 0')

            log.debug(f'Creating new model instance')

            # First create the model instance on meta device to avoid the
            # initialization cost.
            with init_empty_weights():
                new_model_instance = type(original_model)(original_model.config)

            # Then load the state dict in with "assign" so that the state dict
            # is loaded properly even though the model is initially on meta device.
            new_model_instance.load_state_dict(state_dict, assign=True) # Still throwing warnings
            del state_dict

            log.debug('Saving Hugging Face checkpoint to remote')

            new_model_instance.save_pretrained(self.save_name)
            original_tokenizer.save_pretrained(self.save_name)

        dist.barrier()

#####################################################
## Model initialization
#####################################################

# Code taken from https://github.com/mosaicml/llm-foundry/blob/502eb12ff40c69b8a7d693ace8120057afd34338/llmfoundry/models/hf/hf_causal_lm.py#L170
def build_causal_lm(
    pretrained_model_name_or_path,
):
    # Resolve "mixed" init device to either "cpu" or "meta"
    resolved_init_device = hf_get_init_device("mixed")

    # Construct the Hugging Face config to use
    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path,
        use_cache=
        False,  # Necessary due to https://github.com/huggingface/transformers/issues/28056
    )

    # We need to have all non-zero local ranks be not-pretrained
    # Rank 0 will still be pretrained, and distribute the weights appropriately
    pretrained = dist.get_local_rank() == 0
    

    # Hugging Face copies the modules into the
    # transformers modules cache. On particular systems, this operation seems to cause contention between
    # the different processes. To avoid this contention, we first create the model (on meta device) on local rank
    # zero. This will set up the transformers model cache and avoid the future contention.
    if dist.get_local_rank() == 0:
        if os.path.isdir(pretrained_model_name_or_path):
            with init_empty_weights(include_buffers=False):
                AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path,
                    config=config,
                )
        else:
            with init_empty_weights(include_buffers=False):
                AutoModelForCausalLM.from_config(
                    config,
                )

    dist.barrier()

    # initialize the model on the correct device
    if resolved_init_device == 'cpu':
        if pretrained:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path,
                config=config,
            )
        else:
            model = AutoModelForCausalLM.from_config(
                config,
            )
    elif resolved_init_device == 'meta':
        if pretrained:
            raise ValueError(
                'Setting cfg.pretrained=True is not supported when init_device="meta".',
            )
        with init_empty_weights(include_buffers=False):
            model = AutoModelForCausalLM.from_config(
                config,
            )
    else:
        raise ValueError(
            f'init_device="{resolved_init_device}" must be either "cpu" or "meta".',
        )

    signal_file_path = f'.node_{dist.get_node_rank()}_local_rank0_completed'
    if dist.get_local_rank() == 0:
        with open(signal_file_path, 'wb') as f:
            f.write(b'local_rank0_completed_download')

    # Avoid the collective call until the local rank zero has finished trying to download the checkpoint
    # so that we don't timeout for large downloads. This syncs all processes on the node
    with dist.local_rank_zero_download_and_wait(signal_file_path):
        # Then, wait to ensure every node has finished downloading the checkpoint
        dist.barrier()

    if dist.get_local_rank() == 0:
        os.remove(signal_file_path)

    return model


#####################################################
## Trainer
#####################################################
    

def train(cfg, model, train_loader, evaluators):

    # Before popping anything
    to_log_cfg = deepcopy(cfg)

    # Read FSDP Config as a dict
    fsdp_config = cfg.get('fsdp_config', None)
    fsdp_config = (
        om.to_container(fsdp_config, resolve=True) if fsdp_config else None
    )

    # Build optimization and scheduler parameters
    optimizer_config = pop_config(
        cfg,
        'optimizer',
        must_exist=True,
        convert=True,
    )
    optimizer_name = optimizer_config.pop('name')
    optimizer = build_optimizer(model, optimizer_name, optimizer_config)

    scheduler_config = pop_config(cfg, 'scheduler')
    scheduler_name = scheduler_config.pop('name')
    schedulers = build_scheduler(scheduler_name, scheduler_config)

    # Loggers
    loggers = [
        build_logger(name, logger_cfg)
        for name, logger_cfg in (cfg.get('loggers') or {}).items()
    ]

    # Callbacks
    callbacks = [
        build_callback(name, callback_cfg)
        for name, callback_cfg in (cfg.get('callbacks') or {}).items()
    ]
    if cfg.model.feedback_method == "csft":
        save_module_name = "model"
    else:
        save_module_name = "reward_model"
    callbacks.append(HFCheckpointer(cfg.save_folder, save_module_name))

    # Algorithms
    algorithms = [
        build_algorithm(name, algorithm_cfg)
        for name, algorithm_cfg in (cfg.get('algorithms') or {}).items()
    ]

    print('Building trainer...')
    print('save ignore keys are: ', cfg.get('save_ignore_keys'))
    print('saving checkpoints to: ', cfg.get('save_folder'))
    trainer = Trainer(
        run_name=pop_config(
            cfg, 'run_name', must_exist=False, default_value="dynamic_reward",
        ),
        seed=cfg.seed,
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=evaluators,
        optimizers=[optimizer],
        schedulers=schedulers,
        max_duration=cfg.max_duration,
        eval_interval=cfg.eval_interval,
        eval_subset_num_batches=cfg.get('eval_subset_num_batches', -1),
        progress_bar=cfg.get('progress_bar', False),
        log_to_console=cfg.get('log_to_console', True),
        console_log_interval=cfg.get('console_log_interval', '1ba'),
        loggers=loggers,
        callbacks=callbacks,
        precision=cfg.precision,
        algorithms=algorithms,
        device_train_microbatch_size=cfg.get(
            'device_train_microbatch_size', 'auto',
        ),
        fsdp_config=fsdp_config,
        save_folder=cfg.get('save_folder', None),
        save_interval=cfg.get('save_interval', '1000ba'),
        save_num_checkpoints_to_keep=cfg.get(
            'save_num_checkpoints_to_keep', -1,
        ),
        save_ignore_keys=cfg.get('save_ignore_keys', None),
        save_overwrite=cfg.get('save_overwrite', False),
        load_path=cfg.get('load_path', None),
        load_weights_only=cfg.get('load_weights_only', False),
        load_strict_model_weights=cfg.get('load_strict_model_weights', True),
        load_ignore_keys=cfg.get('load_ignore_keys', None),
        autoresume=cfg.get('autoresume', True),
    )

    # Log all cfg params
    if to_log_cfg is not None:
        log_config(to_log_cfg)

    # Eval first if requested
    if cfg.get('eval_first', False) and evaluators is not None and trainer.state.timestamp.batch.value == 0:
        trainer.eval()

    trainer.fit()