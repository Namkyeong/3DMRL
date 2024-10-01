import os
import torch

def write_experiment(args, config_str, best_config):
    
    WRITE_PATH = "results/{}/".format(args.dataset)
    os.makedirs(WRITE_PATH, exist_ok=True) # Create directory if it does not exist
    f = open("results/{}/{}.txt".format(args.dataset, args.embedder), "a")
    f.write("--------------------------------------------------------------------------------- \n")
    f.write(config_str)
    f.write("\n")
    f.write(best_config)
    f.write("\n")
    f.write("--------------------------------------------------------------------------------- \n")
    f.close()

def write_summary_cv(args, config_str, rmse, rmse_std, best_mses, mae, mae_std, best_maes, r2, r2_std, best_r2s):

    if args.pretrained and "Single" in args.pretrained_path:
        WRITE_PATH = "results_single/{}/".format(args.dataset)
        os.makedirs(WRITE_PATH, exist_ok=True) # Create directory if it does not exist
        f = open("results_single/{}/cv_{}.txt".format(args.dataset, args.embedder), "a")
    
    else:
        WRITE_PATH = "results/{}/".format(args.dataset)
        os.makedirs(WRITE_PATH, exist_ok=True) # Create directory if it does not exist
        f = open("results/{}/cv_{}.txt".format(args.dataset, args.embedder), "a")
    
    f.write("--------------------------------------------------------------------------------- \n")
    f.write(config_str)
    f.write("\n")
    if args.pretrained:
        f.write(args.pretrained_path)
        f.write("\n")
    # f.write("5 fold results --> RMSE : {:.4f}({:.4f}) || R2 : {:.4f}({:.4f}) || MAE : {:.4f}({:.4f}) ".format(rmse, rmse_std, r2, r2_std, mae, mae_std))
    f.write("5 fold results --> RMSE : {:.4f}({:.4f}) || R2 : {:.4f}({:.4f})".format(rmse, rmse_std, r2, r2_std))
    f.write("\n")
    f.write("Individual folds result --> RMSE : {:.4f} || {:.4f} || {:.4f} || {:.4f} || {:.4f}".format(best_mses[0], best_mses[1], best_mses[2], best_mses[3], best_mses[4]))
    f.write("\n")
    # f.write("Individual folds result --> MAE : {:.4f} || {:.4f} || {:.4f} || {:.4f} || {:.4f}".format(best_maes[0], best_maes[1], best_maes[2], best_maes[3], best_maes[4]))
    f.write("Individual folds result --> R2 : {:.4f} || {:.4f} || {:.4f} || {:.4f} || {:.4f}".format(best_r2s[0], best_r2s[1], best_r2s[2], best_r2s[3], best_r2s[4]))
    f.write("\n")
    f.write("--------------------------------------------------------------------------------- \n")
    f.close()

def write_summary(args, config_str, rmse, rmse_std, mae, mae_std, r2, r2_std):

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
    # f.write("{} run results --> RMSE : {:.4f}({:.4f}) || MAE : {:.4f}({:.4f})".format(args.repeat, rmse, rmse_std, mae, mae_std))
    f.write("{} run results --> RMSE : {:.4f}({:.4f}) || R2 : {:.4f}({:.4f})".format(args.repeat, rmse, rmse_std, r2, r2_std))
    f.write("\n")
    f.write("--------------------------------------------------------------------------------- \n")
    f.close()

def write_summary_ood(args, config_str, rmse, rmse_std, mae, mae_std, r2, r2_std):

    if args.pretrained and "Single" in args.pretrained_path:
        WRITE_PATH = "results_single/{}/".format(args.dataset)
        os.makedirs(WRITE_PATH, exist_ok=True) # Create directory if it does not exist
        f = open("results_single/{}/ood_summary_{}.txt".format(args.dataset, args.embedder), "a")

    else:
        WRITE_PATH = "results/{}/".format(args.dataset)
        os.makedirs(WRITE_PATH, exist_ok=True) # Create directory if it does not exist
        f = open("results/{}/ood_summary_{}.txt".format(args.dataset, args.embedder), "a")

    f.write("--------------------------------------------------------------------------------- \n")
    f.write(config_str)
    f.write("\n")
    if args.pretrained:
        f.write(args.pretrained_path)
        f.write("\n")
    # f.write("{} run results --> RMSE : {:.4f}({:.4f}) || MAE : {:.4f}({:.4f})".format(args.repeat, rmse, rmse_std, mae, mae_std))
    f.write("{} run results --> RMSE : {:.4f}({:.4f}) || R2 : {:.4f}({:.4f})".format(args.repeat, rmse, rmse_std, r2, r2_std))
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