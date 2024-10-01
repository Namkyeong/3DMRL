import torch
import random
import numpy as np
import os
from utils import argument
from utils.utils import write_summary, get_stats, write_summary_total

from datasets_eval import DDI
from torch_geometric.loader import DataLoader as pyg_DataLoader


torch.set_num_threads(2)


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


def get_dataset(args, repeat):

    dataset = DDI(args.data_path, args.dataset)

    if args.split == "random":
        train_index, valid_index, test_index = dataset.get_random_split(repeat)

    elif args.split == "molecule":
        train_index, valid_index, test_index = dataset.get_molecule_split(repeat)
    
    elif args.split == "scaffold":
        train_index, valid_index, test_index = dataset.get_scaffold_split(repeat)

    train_set = dataset.copy(train_index)
    valid_set = dataset.copy(valid_index)
    test_set = dataset.copy(test_index)

    return train_set, valid_set, test_set


def experiment():

    args, unknown = argument.parse_args()

    best_rocs, best_aps, best_f1s, best_accs = [], [], [], []
    
    for repeat in range(1, args.repeat + 1):

        # Set Random Seed
        seed_everything(repeat)

        train_set, valid_set, test_set = get_dataset(args, repeat)

        stats, config_str, _, _ = main(args, train_set, valid_set, test_set, repeat = repeat)
        
        # get Stats
        best_rocs.append(stats[0])
        best_aps.append(stats[1])
        best_f1s.append(stats[2])
        best_accs.append(stats[3])

        write_summary(args, config_str, stats)
    
    roc_mean, roc_std = get_stats(best_rocs)
    ap_mean, ap_std = get_stats(best_aps)
    f1_mean, f1_std = get_stats(best_f1s)
    accs_mean, accs_std = get_stats(best_accs)

    write_summary_total(args, config_str, [roc_mean, roc_std, ap_mean, ap_std, f1_mean, f1_std, accs_mean, accs_std])


def main(args, train, valid, test, repeat):

    if args.embedder == 'CIGIN':
        from models_2d import CIGIN_ModelTrainer
        embedder = CIGIN_ModelTrainer(args, train, valid, test, repeat)
    
    best_roc, best_ap, best_f1, best_acc = embedder.train()

    return [best_roc, best_ap, best_f1, best_acc], embedder.config_str, embedder.best_config_roc, embedder.best_config_f1


if __name__ == "__main__":
    
    experiment()