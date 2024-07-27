import os
from argparse import ArgumentParser, BooleanOptionalAction

import torch
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.trainer import Trainer
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.plugins import DeepSpeedPrecision
from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from lmft.modeling.ln_model import ReportGenerationModel
from lmft.data_io.clerc import load_clerc_data
from lmft.modeling.save_callback import GenerationSaver
from lmft.utils.suppress_warnings import suppress


def get_save_path(args, logger):
    if args.o is not None:
        return args.o
    if args.ckpt is not None:
        name = os.path.basename(args.ckpt)
        if name.endswith('.ckpt'):
            name = name[:-len('.ckpt')]
        return os.path.join(os.path.dirname(args.ckpt), name + '.predict')
    else:
        return os.path.join(logger.log_dir, 'training.end.predict')


def main():
    parser = ArgumentParser()
    parser.add_argument('action', choices=['train', 'predict'])
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--pretrained', default='meta-llama/Meta-Llama-3.1-8B-Instruct')

    parser.add_argument('--cache', default='./cache')
    parser.add_argument('--exp', default='debug', type=str)

    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--warmup', default=2000, type=int)
    parser.add_argument('--strategy', default="deepspeed", type=str, choices=['deepspeed', 'ddp'])
    parser.add_argument('--precision', default='bf16-mixed', type=str)
    parser.add_argument('--lora', default=32, type=int)

    parser.add_argument('--max-length', default=6000, type=int)
    parser.add_argument('--use-ref', action=BooleanOptionalAction, default=True)
    parser.add_argument('--n-val', default=999999999999, type=int)

    parser.add_argument('--patience', type=int, default=16, help='early stop')
    parser.add_argument('--check-interval', default=200, type=int)
    parser.add_argument('--save-top-k', default=10, type=int)
    parser.add_argument('--eff-bsz', default=32, type=int, help='effective batch size')
    parser.add_argument('--n-gpu', default=8, type=int)

    parser.add_argument('--ckpt', type=str)
    parser.add_argument('-o', type=str)
    parser.add_argument('--max-new', type=int, default=512)

    args = parser.parse_args()
    suppress()

    if args.action == 'train':
        logger = pl_loggers.TensorBoardLogger(args.cache, args.exp)
        callbacks = [
            ModelCheckpoint(
                os.path.join(logger.log_dir, 'ckpt'), monitor='dev_loss', mode='min', save_top_k=args.save_top_k,
                save_last=True, filename='{step:06d}', auto_insert_metric_name=False,
            ),
            LearningRateMonitor('step'),
            EarlyStopping(monitor='dev_loss', min_delta=1e-4, patience=args.patience, verbose=False),
        ]
    elif args.action == 'test':
        logger = pl_loggers.TensorBoardLogger(os.path.join('/tmp/reportgen', args.exp))
        callbacks = []
        args.strategy = 'ddp'
    else:
        raise NotImplementedError
    callbacks.append(GenerationSaver(get_save_path(args, logger), args.pretrained))

    args.n_gpu = min(args.n_gpu, torch.cuda.device_count())
    if args.n_gpu > 0 and torch.cuda.is_available():
        if args.strategy == "deepspeed":
            gpu_kwargs = {
                'plugins': [DeepSpeedPrecision(args.precision)],
                'strategy': "deepspeed_stage_3",
                'devices': args.n_gpu,
            }
        elif args.strategy == 'ddp':
            gpu_kwargs = {
                'precision': args.precision,
                'strategy': DDPStrategy('gpu'),
                'devices': args.n_gpu,
            }
        else:
            raise NotImplementedError
    else:
        gpu_kwargs = {'accelerator': 'cpu'}

    accumulate = args.eff_bsz // (args.n_gpu+0.1)
    trainer = Trainer(
        log_every_n_steps=20, use_distributed_sampler=args.n_gpu > 1, gradient_clip_val=.8,
        gradient_clip_algorithm='norm', max_epochs=128, logger=logger, enable_progress_bar=True,
        callbacks=callbacks, accumulate_grad_batches=accumulate, check_val_every_n_epoch=1,
        val_check_interval=args.check_interval, **gpu_kwargs,
    )

    if args.ckpt is None:
        model = ReportGenerationModel(
            pretrained=args.pretrained, lr=args.lr, warmup=args.warmup, lora_rank=args.lora,
            max_new=args.max_new
        )
    else:
        if os.path.isdir(args.ckpt):
            new_ckpt_path = args.ckpt.replace('.ckpt', '.ln.ckpt')
            convert_zero_checkpoint_to_fp32_state_dict(args.ckpt, new_ckpt_path)
            args.ckpt = new_ckpt_path
        model = ReportGenerationModel.load_from_checkpoint(args.ckpt, strict=False)

    if args.data == 'clerc':
        if args.action == 'predict':
            args.n_val = 9999999
        train_dl, test_dl = load_clerc_data(
            bsz=1, pretrained=args.pretrained, max_length=args.max_length, shuffle=True, use_ref=args.use_ref,
            n_val=args.n_val,
        )
    else:
        raise NotImplementedError

    if args.action == 'train':
        model.train()
        trainer.fit(model, train_dl, test_dl)
    elif args.action == 'predict':
        trainer.predict(model, test_dl)


if __name__ == '__main__':
    main()
