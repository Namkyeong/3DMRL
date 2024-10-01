import os
import torch

from tensorboardX import SummaryWriter

from utils.argument import config2string
from utils.utils import save

from torch_geometric.loader import DataLoader as pyg_DataLoader

import datetime

class pretrainer:

    def __init__(self, args, train_data):

        d = datetime.datetime.now()
        date = d.strftime("%x")[-2:] + d.strftime("%x")[0:2] + d.strftime("%x")[3:5]
        
        self.args = args
        self.config_str = "{}_".format(date) + config2string(args)
        print("Configuration: ", self.config_str)

        # Model Checkpoint Path
        self.check_dir = "pretrained_weights/"
        os.makedirs(self.check_dir, exist_ok=True) # Create directory if it does not exist

        # Select GPU device
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        self.device = f'cuda:{args.device}' if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(self.device)

        self.writer = SummaryWriter(log_dir="runs_pretrain/{}".format(self.config_str))

        dataloader_class = pyg_DataLoader
        self.loader = dataloader_class(train_data, batch_size=args.batch_size, shuffle=True, num_workers = 8)

    
    def save_checkpoints(self, epochs = None):
        
        save(self.check_dir, "2d", self.model_2d, "epoch_{}".format(epochs) + self.config_str)
        save(self.check_dir, "3d", self.model_3d, "epoch_{}".format(epochs) + self.config_str)