import warnings
import logging

from transformers import AutoConfig, AutoTokenizer


filtered_out = [
    'The `srun` command is available on your system but is not used',
    'does not have many workers',
    'YPU available', 'TPU available', 'IPU available', 'HPU available',
    'exists and is not empty', 'Setting ds_accelerator', 'that has Tensor Cores',
    '- CUDA_VISIBLE_DEVICES', 'Found keys that are in the model state',
    'Positional args are being deprecated', 'could not find the monitored key',
    'torch.cuda.*DtypeTensor constructors',
    'Special tokens have been added in the vocabulary',
    'We detected that you are passing `past_key_values`',
    'Setting `pad_token_id` to `eos_token_id`',
]

for fo_ in filtered_out:
    warnings.filterwarnings('ignore', f'.*{fo_}.*')


class SupressFilter(logging.Filter):
    def filter(self, record):
        for fo in filtered_out:
            if fo.lower() in record.msg.lower():
                return False
        return True


for logger_name in [
    'DeepSpeed', 'lightning_utilities.core.rank_zero', 'lightning.pytorch.utilities.rank_zero',
    'lightning.pytorch.accelerators.cuda'
]:
    logging.getLogger(logger_name).addFilter(SupressFilter())


def suppress():
    try:
        import datasets
        datasets.logging.set_verbosity_error()
    except ImportError:
        pass
    try:
        import deepspeed
        logging.getLogger('DeepSpeed').setLevel('WARNING')
    except ImportError:
        pass
    try:
        import transformers
        transformers.logging.set_verbosity_error()
    except ImportError:
        pass


def load_tokenizer(pretrained: str, **kwargs):
    # warnings are avoided
    tokenizer_base_logger = logging.getLogger('transformers.tokenization_utils_base')
    tokenizer_base_logger.setLevel('ERROR')
    config = AutoConfig.from_pretrained(pretrained)
    if config.model_type == 't5':
        kwargs['model_max_length'] = 999999
        kwargs['legacy'] = False
    elif config.model_type == 'llama':
        kwargs['legacy'] = False
    return AutoTokenizer.from_pretrained(pretrained, **kwargs)
