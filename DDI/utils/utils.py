import os
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, f1_score, accuracy_score

def get_stats(array):
    
    mean = np.mean(np.asarray(array))
    std = np.std(np.asarray(array))

    return mean, std


def write_summary(args, config_str, stats):

    if args.pretrained and "Single" in args.pretrained_path:
        WRITE_PATH = "results_single/{}/".format(args.dataset)
        os.makedirs(WRITE_PATH, exist_ok=True) # Create directory if it does not exist
        f = open("results_single/{}/{}.txt".format(args.dataset, args.embedder), "a")

    else:
        WRITE_PATH = "results/{}/".format(args.dataset)
        os.makedirs(WRITE_PATH, exist_ok=True) # Create directory if it does not exist
        f = open("results/{}/{}.txt".format(args.dataset, args.embedder), "a")

    f.write("--------------------------------------------------------------------------------- \n")
    f.write(config_str)
    f.write("\n")
    if args.pretrained:
        f.write(args.pretrained_path)
        f.write("\n")
    f.write("ROC : {:.4f} || AP : {:.4f} || F1 : {:.4f} || Acc : {:.4f} ".format(stats[0], stats[1], stats[2], stats[3]))
    f.write("\n")
    f.write("--------------------------------------------------------------------------------- \n")
    f.close()


def write_summary_total(args, config_str, stats):
    
    if args.pretrained and "Single" in args.pretrained_path:
        WRITE_PATH = "results_single/{}/".format(args.dataset)
        os.makedirs(WRITE_PATH, exist_ok=True) # Create directory if it does not exist
        f = open("results_single/{}/summary_{}.txt".format(args.dataset, args.embedder), "a")

    else:
        WRITE_PATH = "results/{}/".format(args.dataset)
        os.makedirs(WRITE_PATH, exist_ok=True) # Create directory if it does not exist
        f = open("results/{}/summary_{}.txt".format(args.dataset, args.embedder), "a")

    f.write("--------------------------------------------------------------------------------- \n")
    f.write(config_str)
    f.write("\n")
    if args.pretrained:
        f.write(args.pretrained_path)
        f.write("\n")
    f.write("ROC : {:.4f}({:.4f}) || AP : {:.4f}({:.4f}) || F1 : {:.4f}({:.4f}) || Acc : {:.4f}({:.4f}) ".format(stats[0], stats[1], stats[2], stats[3],
                                                                                                                                    stats[4], stats[5], stats[6], stats[7]))
    f.write("\n")
    f.write("--------------------------------------------------------------------------------- \n")
    f.close()


def create_batch_mask(samples):
    batch0 = samples[0].batch.reshape(1, -1)
    index0 = torch.cat([batch0, torch.tensor(range(batch0.shape[1])).reshape(1, -1)])
    mask0 = torch.sparse_coo_tensor(index0, torch.ones(index0.shape[1]), size = (batch0.max() + 1, batch0.shape[1]))

    batch1 = samples[1].batch.reshape(1, -1)
    index1 = torch.cat([batch1, torch.tensor(range(batch1.shape[1])).reshape(1, -1)])
    mask1 = torch.sparse_coo_tensor(index1, torch.ones(index1.shape[1]), size = (batch1.max() + 1, batch1.shape[1]))

    return mask0, mask1


def save(checkpoint_path, model_path, model, config):

    saved_file_path = os.path.join(checkpoint_path, model_path)
    
    ##### Create directory if it does not exist #####
    os.makedirs(saved_file_path, exist_ok=True) 
    
    saved_file_path = os.path.join(saved_file_path, "{}.pth".format(config))
    torch.save(model.state_dict(), saved_file_path)

    print("{} has been saved!".format(model_path))


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]

    return list(map(lambda s: x == s, allowable_set))


EPSILON = 1e-6


def coord2basis_SDE(pos, row, col):
    coord_diff = pos[row] - pos[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    coord_cross = torch.cross(pos[row], pos[col])

    norm = torch.sqrt(radial) + EPSILON
    coord_diff = coord_diff / norm
    cross_norm = torch.sqrt(torch.sum((coord_cross) ** 2, 1).unsqueeze(1)) + EPSILON
    coord_cross = coord_cross / cross_norm

    coord_vertical = torch.cross(coord_diff, coord_cross)

    return coord_diff, coord_cross, coord_vertical


def coord2basis(solute_pos, solvent_pos):
    coord_diff = solute_pos - solvent_pos
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    coord_cross = torch.cross(solute_pos, solvent_pos)

    norm = torch.sqrt(radial) + EPSILON
    coord_diff = coord_diff / norm
    cross_norm = torch.sqrt(torch.sum((coord_cross) ** 2, 1).unsqueeze(1)) + EPSILON
    coord_cross = coord_cross / cross_norm

    coord_vertical = torch.cross(coord_diff, coord_cross)

    return coord_diff, coord_cross, coord_vertical


def get_perturb_distance(p_pos, edge_index):
    pos = p_pos
    row, col = edge_index
    d = (pos[row] - pos[col]).norm(dim=-1).unsqueeze(-1)  # (num_edge, 1)
    return d


def get_roc_score(preds, labels):

    preds_all, preds_all_ = eval_threshold(labels, preds)

    roc_score = roc_auc_score(labels, preds_all)
    ap_score = average_precision_score(labels, preds_all)
    f1_score_ = f1_score(labels, preds_all_)
    acc_score = accuracy_score(labels, preds_all_)
    return roc_score, ap_score, f1_score_, acc_score


def eval_threshold(labels_all, preds_all):

    # fpr, tpr, thresholds = roc_curve(labels_all, preds_all)
    # optimal_idx = np.argmax(tpr - fpr)
    # optimal_threshold = thresholds[optimal_idx]
    optimal_threshold = 0.5
    preds_all_ = []
    
    for p in preds_all:
        if p >=optimal_threshold:
            preds_all_.append(1)
        else:
            preds_all_.append(0)

    return preds_all, preds_all_