# from data import Custom_Data
from utils import argument
import random
import os

import numpy as np
import torch

from datasets_pretrain import ChromophoreAll, ChromophoreSubset, SolvationAll, SolvationSubset

torch.set_num_threads(4)
os.environ['OMP_NUM_THREADS'] = "4"


def seed_everything(seed=0):
    # To fix the random seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataset(args):

    if args.dataset == "Chromophore":
        if args.subset:
            dataset = ChromophoreAll(args.data_path, args.rotation, args.radius, args.fixed_direction, args.sample, args.no_solvent)
            size = int(len(dataset) * args.ratio)
            dataset = ChromophoreSubset(args.data_path, size, args.rotation, args.radius, args.fixed_direction, args.sample, args.no_solvent)
            print("Prepared Subset Data: {}".format(size))
        else:
            dataset = ChromophoreAll(args.data_path, args.rotation, args.radius, args.fixed_direction, args.sample, args.no_solvent)
            print("Prepared Full Data: {}".format(len(dataset)))
    
    elif args.dataset == "Solvation":
        if args.subset:
            dataset = SolvationAll(args.data_path, args.rotation, args.radius, args.fixed_direction, args.sample, args.no_solvent)
            size = int(len(dataset) * args.ratio)
            dataset = SolvationSubset(args.data_path, size, args.rotation, args.radius, args.fixed_direction, args.sample, args.no_solvent)
            print("Prepared Subset Data: {}".format(size))
        else:
            dataset = SolvationAll(args.data_path, args.rotation, args.radius, args.fixed_direction, args.sample, args.no_solvent)
            print("Prepared Full Data: {}".format(len(dataset)))
    
    else:
        raise Exception
    
    return dataset


def main():

    args, unknown = argument.pretrain_parse_args()
    seed_everything(0)

    dataset = get_dataset(args)

    if args.pretrainer == '3DMRL':
        from pretrainers import MRL3D_PreTrainer
        pretrainer = MRL3D_PreTrainer(args, dataset)
    
    else:
        raise Exception
    
    pretrainer.train()
    pretrainer.save_checkpoints()


if __name__ == "__main__":
    main()