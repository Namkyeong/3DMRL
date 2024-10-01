import numpy as np
import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
import random
random.seed(0)

from tensorboardX import SummaryWriter
import os

from utils.argument import config2string
from utils.utils import create_batch_mask
from sklearn.metrics import r2_score

from torch_geometric.loader import DataLoader as pyg_DataLoader


class embedder:

    def __init__(self, args, train, valid, test, repeat, fold):
        self.args = args
        self.repeat = repeat
        self.config_str = "experiment{}_fold{}_".format(repeat + 1, fold + 1) + config2string(args)
        print("\n[Config] {}\n".format(self.config_str))
        
        if args.writer:
            if repeat == 0 and fold == 0:
                self.writer = SummaryWriter(log_dir="runs/{}".format(self.config_str))
            else:
                self.writer = SummaryWriter(log_dir="runs_/{}".format(self.config_str))
        else:
            self.writer = SummaryWriter(log_dir="runs_/{}".format(self.config_str))

        self.save_checkpoints = args.save_checkpoints

        # Model Checkpoint Path
        CHECKPOINT_PATH = "model_checkpoints/{}/".format(args.embedder)
        os.makedirs(CHECKPOINT_PATH, exist_ok=True) # Create directory if it does not exist
        self.check_dir = CHECKPOINT_PATH + self.config_str + ".pth"

        # Select GPU device
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        self.device = f'cuda:{args.device}' if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(self.device)
        # self.device = "cpu"

        dataloader_class = pyg_DataLoader
        self.train_loader = dataloader_class(train, batch_size=args.batch_size, shuffle=True)
        self.val_loader = dataloader_class(valid, batch_size=args.batch_size)
        self.test_loader = dataloader_class(test, batch_size=args.batch_size)

        self.best_val_loss = 100000000000000.0
        self.best_val_losses = []

        self.fold = fold
    

    def evaluate(self, epoch, final = False):
        
        valid_losses = []
        valid_mae_losses = []
        valid_preds = []
        valid_ys = []
    
        test_losses = []
        test_mae_losses = []
        test_preds = []
        test_ys = []
        
        for bc, samples in enumerate(self.val_loader):

            with torch.no_grad():

                masks = create_batch_mask(samples)
                output, _ = self.model([samples[0].to(self.device), samples[1].to(self.device), masks[0].to(self.device), masks[1].to(self.device)], test = True)
                
                val_loss = ((output - samples[2].reshape(-1, 1).to(self.device))**2).reshape(-1)
                val_mae_loss = torch.abs((output - samples[2].reshape(-1, 1).to(self.device))).reshape(-1)
                valid_losses.append(val_loss.cpu().detach().numpy())
                valid_mae_losses.append(val_mae_loss.cpu().detach().numpy())
                valid_preds.append(output.reshape(-1).cpu().detach().numpy())
                valid_ys.append(samples[2].reshape(-1).cpu().detach().numpy())
        
        valid_preds = np.hstack(valid_preds)
        valid_ys = np.hstack(valid_ys)

        for bc, samples in enumerate(self.test_loader):

            with torch.no_grad():
            
                masks = create_batch_mask(samples)
                output, _ = self.model([samples[0].to(self.device), samples[1].to(self.device), masks[0].to(self.device), masks[1].to(self.device)], test = True)
                
                test_loss = ((output - samples[2].reshape(-1, 1).to(self.device))**2).reshape(-1)
                test_mae_loss = torch.abs((output - samples[2].reshape(-1, 1).to(self.device))).reshape(-1)
                test_losses.append(test_loss.cpu().detach().numpy())
                test_mae_losses.append(test_mae_loss.cpu().detach().numpy())
                test_preds.append(output.reshape(-1).cpu().detach().numpy())
                test_ys.append(samples[2].reshape(-1).cpu().detach().numpy())
        
        test_preds = np.hstack(test_preds)
        test_ys = np.hstack(test_ys)

        self.val_loss = np.mean(np.hstack(valid_losses))
        self.val_rmse_loss = np.sqrt(self.val_loss)
        self.val_mae_loss = np.mean(np.hstack(valid_mae_losses))
        self.val_r2 = r2_score(valid_ys, valid_preds)
        
        self.test_loss = np.mean(np.hstack(test_losses))
        self.test_rmse_loss = np.sqrt(self.test_loss)
        self.test_mae_loss = np.mean(np.hstack(test_mae_losses))
        self.test_r2 = r2_score(test_ys, test_preds)

        self.writer.add_scalar("accs/train_loss", self.train_loss, epoch)
        self.writer.add_scalar("accs/valid_RMSE", self.val_rmse_loss, epoch)
        self.writer.add_scalar("accs/test_RMSE", self.test_rmse_loss, epoch)
        self.writer.add_scalar("accs/valid_R2", self.val_r2, epoch)
        self.writer.add_scalar("accs/test_R2", self.test_r2, epoch)

        if self.val_loss < self.best_val_loss :
            self.best_val_loss = self.val_loss
            self.best_test_loss = self.test_loss
            self.best_val_mae_loss = self.val_mae_loss
            self.best_test_mae_loss = self.test_mae_loss
            self.best_val_rmse_loss = self.val_rmse_loss
            self.best_test_rmse_loss = self.test_rmse_loss
            self.best_val_r2 = self.val_r2
            self.best_test_r2 = self.test_r2
            self.best_epoch = epoch

            if self.save_checkpoints:
                if self.repeat == 0:
                    self.save_model()
                else:
                    pass

        self.best_val_losses.append(self.best_val_loss)

        self.eval_config = "[Epoch: {}] valid RMSE / R2 --> {:.4f} / {:.4f} || test RMSE / R2 --> {:.4f} / {:.4f} ".format(epoch, self.val_rmse_loss, self.val_r2, self.test_rmse_loss, self.test_r2)
        self.best_config = "[Best Epoch: {}] Best valid RMSE / R2 --> {:.4f} / {:.4f} || Best test RMSE / R2 --> {:.4f} / {:.4f} ".format(self.best_epoch, self.best_val_rmse_loss, self.best_val_r2, self.best_test_rmse_loss, self.best_test_r2)

        print(self.eval_config)
        print(self.best_config)


    def evaluate_VR(self, epoch, final = False):
        
        valid_losses = []
        valid_mae_losses = []
        valid_preds = []
        valid_ys = []
    
        test_losses = []
        test_mae_losses = []
        test_preds = []
        test_ys = []

        for bc, samples in enumerate(self.val_loader):

            with torch.no_grad():
                output, _ = self.model(samples.to(self.device), test = True)
                
                val_loss = ((output - samples[2].reshape(-1, 1).to(self.device))**2).reshape(-1)
                val_mae_loss = torch.abs((output - samples[2].reshape(-1, 1).to(self.device))).reshape(-1)
                valid_losses.append(val_loss.cpu().detach().numpy())
                valid_mae_losses.append(val_mae_loss.cpu().detach().numpy())
                valid_preds.append(output.reshape(-1).cpu().detach().numpy())
                valid_ys.append(samples[2].reshape(-1).cpu().detach().numpy())
        
        valid_preds = np.hstack(valid_preds)
        valid_ys = np.hstack(valid_ys)


        for bc, samples in enumerate(self.test_loader):

            with torch.no_grad():
                output, _ = self.model(samples.to(self.device), test = True)
                
                test_loss = ((output - samples[2].reshape(-1, 1).to(self.device))**2).reshape(-1)
                test_mae_loss = torch.abs((output - samples[2].reshape(-1, 1).to(self.device))).reshape(-1)
                test_losses.append(test_loss.cpu().detach().numpy())
                test_mae_losses.append(test_mae_loss.cpu().detach().numpy())
                test_preds.append(output.reshape(-1).cpu().detach().numpy())
                test_ys.append(samples[2].reshape(-1).cpu().detach().numpy())
        
        test_preds = np.hstack(test_preds)
        test_ys = np.hstack(test_ys)

        self.val_loss = np.mean(np.hstack(valid_losses))
        self.val_rmse_loss = np.sqrt(self.val_loss)
        self.val_mae_loss = np.mean(np.hstack(valid_mae_losses))
        self.val_r2 = r2_score(valid_ys, valid_preds)
        
        self.test_loss = np.mean(np.hstack(test_losses))
        self.test_rmse_loss = np.sqrt(self.test_loss)
        self.test_mae_loss = np.mean(np.hstack(test_mae_losses))
        self.test_r2 = r2_score(test_ys, test_preds)

        self.writer.add_scalar("accs/train_loss", self.train_loss, epoch)
        self.writer.add_scalar("accs/valid_RMSE", self.val_rmse_loss, epoch)
        self.writer.add_scalar("accs/test_RMSE", self.test_rmse_loss, epoch)
        self.writer.add_scalar("accs/valid_R2", self.val_r2, epoch)
        self.writer.add_scalar("accs/test_R2", self.test_r2, epoch)

        if self.val_loss < self.best_val_loss :
            self.best_val_loss = self.val_loss
            self.best_test_loss = self.test_loss
            self.best_val_mae_loss = self.val_mae_loss
            self.best_test_mae_loss = self.test_mae_loss
            self.best_val_rmse_loss = self.val_rmse_loss
            self.best_test_rmse_loss = self.test_rmse_loss
            self.best_val_r2 = self.val_r2
            self.best_test_r2 = self.test_r2
            self.best_epoch = epoch

        self.best_val_losses.append(self.best_val_loss)

        self.eval_config = "[Epoch: {}] valid RMSE / R2 --> {:.4f} / {:.4f} || test RMSE / R2 --> {:.4f} / {:.4f} ".format(epoch, self.val_rmse_loss, self.val_r2, self.test_rmse_loss, self.test_r2)
        self.best_config = "[Best Epoch: {}] Best valid RMSE / R2 --> {:.4f} / {:.4f} || Best test RMSE / R2 --> {:.4f} / {:.4f} ".format(self.best_epoch, self.best_val_rmse_loss, self.best_val_r2, self.best_test_rmse_loss, self.best_test_r2)

        print(self.eval_config)
        print(self.best_config)


    def save_model(self, epoch = None):

        torch.save(self.model.state_dict(), self.check_dir)
        print("{} has been saved!".format(self.args.embedder))

