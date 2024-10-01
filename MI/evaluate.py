import torch
import random
import numpy as np
import os
from utils import argument
from utils.utils import write_summary, write_summary_cv

from datasets_eval import Chromophore, Solvation

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


def get_dataset(args):

    if args.dataset in ["Absorption max (nm)", "Emission max (nm)", "Lifetime (ns)"]:
        dataset = Chromophore(args.data_path, args.dataset, args.log_target)
    if args.dataset in ["Abraham", "CombiSolv", "CompSol", "FreeSol", "MNSol"]:
        dataset = Solvation(args.data_path, args.dataset)
    
    return dataset


def experiment():

    args, unknown = argument.parse_args()
    
    cv_rmses = []
    cv_maes = []
    cv_r2s = []

    for repeat in range(args.repeat):

        # Set Random Seed
        seed_everything(repeat)

        dataset = get_dataset(args)
        len_data = len(dataset)
        len_test = int(len_data / args.fold)
        len_val = int(len_test * 0.5)

        total_index = [i for i in range(len_data)]
        random.shuffle(total_index) # Shuffle index
        total_index = np.asarray(total_index)

        best_mses = []
        best_maes = []
        best_r2s = []

        for fold in range(args.fold):

            test_index = np.asarray([i for i in range(fold * len_test, (fold + 1) * len_test)])
            train_index = np.asarray([i for i in range(len_data) if np.isin(i, test_index, invert = True)])
            
            test_index = total_index[test_index]
            train_index = total_index[train_index]
            
            # Get validation set
            val_index = test_index[:len_val]
            test_index = test_index[len_val:]

            train_set = dataset.copy(train_index)
            val_set = dataset.copy(val_index)
            test_set = dataset.copy(test_index)
            
            best_mse, best_mae, best_r2, config_str, best_config = main(args, train_set, val_set, test_set, repeat, fold)

            best_mses.append(best_mse)
            best_maes.append(best_mae)
            best_r2s.append(best_r2)
            
        rmse = np.mean(np.sqrt(np.asarray(best_mses)))
        rmse_std = np.std(np.sqrt(np.asarray(best_mses)))

        mae = np.mean(np.asarray(best_maes))
        mae_std = np.std(np.asarray(best_maes))

        r2 = np.mean(np.asarray(best_r2s))
        r2_std = np.std(np.asarray(best_r2s))

        cv_rmses.append(rmse)
        cv_maes.append(mae)
        cv_r2s.append(r2)
        write_summary_cv(args, config_str, rmse, rmse_std, np.sqrt(np.asarray(best_mses)), mae, mae_std, np.asarray(best_maes), r2, r2_std, np.asarray(best_r2s))

    rmse = np.mean(np.asarray(cv_rmses))
    rmse_std = np.std(np.asarray(cv_rmses))

    mae = np.mean(np.asarray(cv_maes))
    mae_std = np.std(np.asarray(cv_maes))

    r2 = np.mean(np.asarray(cv_r2s))
    r2_std = np.std(np.asarray(cv_r2s))

    write_summary(args, config_str, rmse, rmse_std, mae, mae_std, r2, r2_std)


def main(args, train, valid, test, repeat, fold = 0):

    if args.embedder == 'CIGIN':
        from models_2d import CIGIN_ModelTrainer
        embedder = CIGIN_ModelTrainer(args, train, valid, test, repeat, fold)
    
    elif args.embedder == 'AttentiveFP':
        from models_2d import AttentiveFP_ModelTrainer
        embedder = AttentiveFP_ModelTrainer(args, train, valid, test, repeat, fold)

    elif args.embedder == 'MPNN':
        from models_2d import MPNN_ModelTrainer
        embedder = MPNN_ModelTrainer(args, train, valid, test, repeat, fold)
    
    elif args.embedder == 'CGIB':
        from models_2d import CGIB_ModelTrainer
        embedder = CGIB_ModelTrainer(args, train, valid, test, repeat, fold)
    
    elif args.embedder == 'CGIB_cont':
        from models_2d import CGIB_cont_ModelTrainer
        embedder = CGIB_cont_ModelTrainer(args, train, valid, test, repeat, fold)
    
    else:
        raise Exception
    
    best_mse, best_mae, best_r2 = embedder.train()

    return best_mse, best_mae, best_r2, embedder.config_str, embedder.best_config


if __name__ == "__main__":
    experiment()