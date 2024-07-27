from argparse import ArgumentParser
import os

from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
import torch

from lmft.utils.params import param_names


def main():
    parser = ArgumentParser()
    parser.add_argument('-i')
    parser.add_argument('-o')
    parser.add_argument('-c', action='store_true')
    args = parser.parse_args()
    if args.o is None:
        args.o = args.i.replace('.ckpt', '.lightning.ckpt')
    if not os.path.exists(args.o):
        convert_zero_checkpoint_to_fp32_state_dict(args.i, args.o)
    if not args.c:
        return
    ckpt = torch.load(args.o)
    for line in param_names(list(ckpt['state_dict'].keys())):
        print(line[0])


if __name__ == '__main__':
    main()

