import argparse


def pretrain_parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--pretrainer", type=str, default="3DMRL")
    
    # Argument for Datasets
    parser.add_argument("--data_path", type=str, default="./data_pretrain")
    parser.add_argument("--dataset", type=str, default="Chromophore")

    parser.add_argument("--use_subset", dest = "subset", action = "store_true")
    parser.add_argument("--use_full", dest = "subset", action = "store_false")    
    parser.set_defaults(subset=False)

    if parser.parse_known_args()[0].subset:
        parser.add_argument("--ratio", type=float, default=0.1)

    # Argument for Training
    parser.add_argument("--device", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default = 0.0)
    parser.add_argument("--dropout", type=float, default = 0.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--save_freq", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    
    # Argument for 2D Models
    parser.add_argument("--message_passing", type=int, default=3)
    
    # Argument for 3D Models
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--cut_off", type=float, default=10.0)
    parser.add_argument("--t_message_passing", type=int, default=6)
    
    parser.add_argument("--alpha", type=float, default=1.0)

    parser.add_argument("--save_checkpoints", dest = "save_checkpoints", action="store_true")
    parser.add_argument("--not_save_checkpoints", dest = "save_checkpoints", action="store_false")
    parser.set_defaults(save_checkpoints=False)

    parser.add_argument("--writer", dest = "writer", action = "store_true")
    parser.add_argument("--no_writer", dest = "writer", action = "store_false")
    parser.set_defaults(writer=True)

    # Argument for Virtual Environment
    parser.add_argument("--no_solvent", dest = "no_solvent", action = "store_true")
    parser.add_argument("--solvent", dest = "no_solvent", action = "store_false")
    parser.set_defaults(no_solvent=False)

    parser.add_argument("--sample", type=int, default=None)

    parser.add_argument("--rotation", dest = "rotation", action = "store_true")
    parser.add_argument("--no_rotation", dest = "rotation", action = "store_false")
    parser.set_defaults(rotation=False)

    parser.add_argument("--radius", dest = "radius", action = "store_true")
    parser.add_argument("--no_radius", dest = "radius", action = "store_false")
    parser.set_defaults(radius=False)

    parser.add_argument("--fixed_direction", dest = "fixed_direction", action = "store_true")
    parser.add_argument("--no_fixed_direction", dest = "fixed_direction", action = "store_false")
    parser.set_defaults(fixed_direction=False)

    parser.add_argument("--only_solute", dest = "only_solute", action = "store_true")
    parser.add_argument("--not_only_solute", dest = "only_solute", action = "store_false")
    parser.set_defaults(only_solute=False)

    parser.add_argument("--no_node_feature", dest = "no_node_feature", action = "store_true")
    parser.add_argument("--node_feature", dest = "no_node_feature", action = "store_false")    
    parser.set_defaults(no_node_feature=False)

    return parser.parse_known_args()



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="./data_eval")

    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--fold", type=int, default=5)
    parser.add_argument("--embedder", type=str, default="CIGIN", 
                        choices=["CIGIN", "CGIB", "CGIB_cont", "MPNN", "AttentiveFP"])
    parser.add_argument("--dataset", type = str, default = 'Absorption max (nm)',
                        choices=["Absorption max (nm)", "Emission max (nm)", "Lifetime (ns)", "Abraham", "CombiSolv", "CompSol", "FreeSol", "MNSol"])

    parser.add_argument("--device", type=int, default=0)

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default = 0.0)
    parser.add_argument("--dropout", type=float, default = 0.0)
    parser.add_argument("--scheduler", type=str, default="plateau", help = "plateau")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--es", type=int, default=50)
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--message_passing", type=int, default=3)

    parser.add_argument("--save_checkpoints", dest = "save_checkpoints", action="store_true")
    parser.add_argument("--not_save_checkpoints", dest = "save_checkpoints", action="store_false")
    parser.set_defaults(save_checkpoints=False)

    parser.add_argument("--writer", dest = "writer", action = "store_true")
    parser.add_argument("--no_writer", dest = "writer", action = "store_false")
    parser.set_defaults(writer=True)

    parser.add_argument("--normalize", dest = "normalize", action = "store_true")
    parser.add_argument("--no_normalize", dest = "normalize", action = "store_false")
    parser.set_defaults(normalize=False)

    parser.add_argument("--log_target", dest = "log_target", action = "store_true")
    parser.add_argument("--target", dest = "log_target", action = "store_false")
    parser.set_defaults(log_target=False)

    parser.add_argument("--pretrained", dest = "pretrained", action = "store_true")
    parser.add_argument("--no_pretrained", dest = "pretrained", action = "store_false")
    parser.set_defaults(pretrained=False)

    if "CGIB" in parser.parse_known_args()[0].embedder:
        parser.add_argument("--beta", type=float, default=1.0)
    
    if "cont" in parser.parse_known_args()[0].embedder:
        parser.add_argument("--tau", type=float, default=1.0)

    if parser.parse_known_args()[0].pretrained:
        parser.add_argument("--pretrained_path", 
                            type=str, default="epoch_100pretrainer_Infomax_dataset_Absorption max (nm)_lr_0.0001_lr_scale_2d_1.0_lr_scale_3d_1.0_message_passing_5_tau_0.2_cut_off_5.0_t_message_passing_6_only_solute_True_noise_True.pth")

    return parser.parse_known_args()


def parse_args_ood():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="./data_eval")

    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--embedder", type=str, default="CIGIN")
    parser.add_argument("--dataset", type = str, default = 'Absorption max (nm)',
                        choices=["Absorption max (nm)", "Emission max (nm)", "Lifetime (ns)", "Abraham", "CombiSolv", "CompSol", "FreeSol", "MNSol"])
    parser.add_argument("--split", type = str, default = 'molecule', choices=["molecule", "scaffold"])

    parser.add_argument("--device", type=int, default=0)

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default = 0.0)
    parser.add_argument("--dropout", type=float, default = 0.0)
    parser.add_argument("--scheduler", type=str, default="plateau", help = "plateau")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--es", type=int, default=50)
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--message_passing", type=int, default=3)

    parser.add_argument("--save_checkpoints", dest = "save_checkpoints", action="store_true")
    parser.add_argument("--not_save_checkpoints", dest = "save_checkpoints", action="store_false")
    parser.set_defaults(save_checkpoints=False)

    parser.add_argument("--writer", dest = "writer", action = "store_true")
    parser.add_argument("--no_writer", dest = "writer", action = "store_false")
    parser.set_defaults(writer=False)

    parser.add_argument("--normalize", dest = "normalize", action = "store_true")
    parser.add_argument("--no_normalize", dest = "normalize", action = "store_false")
    parser.set_defaults(normalize=False)

    parser.add_argument("--log_target", dest = "log_target", action = "store_true")
    parser.add_argument("--target", dest = "log_target", action = "store_false")
    parser.set_defaults(log_target=False)

    parser.add_argument("--pretrained", dest = "pretrained", action = "store_true")
    parser.add_argument("--no_pretrained", dest = "pretrained", action = "store_false")
    parser.set_defaults(pretrained=False)

    if parser.parse_known_args()[0].pretrained:
        parser.add_argument("--pretrained_path", 
                            type=str, default="epoch_100pretrainer_Infomax_dataset_Absorption max (nm)_lr_0.0001_lr_scale_2d_1.0_lr_scale_3d_1.0_message_passing_5_tau_0.2_cut_off_5.0_t_message_passing_6_only_solute_True_noise_True.pth")

    return parser.parse_known_args()


def enumerateConfig(args):
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))

    return args_names, args_vals


def printConfig(args):
    args_names, args_vals = enumerateConfig(args)
    print(args_names)
    print(args_vals)


def config2string(args):
    args_names, args_vals = enumerateConfig(args)
    st = ''
    for name, val in zip(args_names, args_vals):
        if val == False:
            continue
        if name not in ['fold', 'repeat', 'data_path', 'root', 'task', 'eval_freq', 'patience', 'device', 'writer', "batch_size", 'num_workers',
                        'scheduler', 'fg_pooler', 'prob_temp', 'es', 'epochs', 'cv', 'interaction', "save_checkpoints",
                        'norm_loss', 'layers', 'pred_hid', 'mad', "anneal_rate", "temp_min", "sparsity_regularizer", "entropy_regularizer", 
                        "train_ratio", "pretrained_path", "lr_scale_2d", "lr_scale_3d", "t_message_passing", "save_freq"]:
            st_ = "{}_{}_".format(name, val)
            st += st_

    return st[:-1]