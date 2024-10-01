import sys
sys.path.append('.')

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

from torch_geometric.nn import Set2Set

from embedder import embedder
from utils.utils import create_batch_mask

from torch_geometric.loader import DataLoader as pyg_DataLoader
from layers import AttentiveFP


class AttentiveFP_ModelTrainer(embedder):
    def __init__(self, args, train_df, valid_df, test_df, repeat, fold):
        embedder.__init__(self, args, train_df, valid_df, test_df, repeat, fold)

        self.model = DualAttentiveFP(num_step_message_passing = self.args.message_passing).to(self.device)

        if args.pretrained:

            # For single pretrained models
            if "Single" in args.pretrained_path:
                print("Loading Single Pretrained Models...")
                PATH = "./pretrained_weights_single/2d/{}".format(args.pretrained_path)
                state_dict = torch.load(PATH, map_location=self.device)
                self.model.solute_gather.load_state_dict(state_dict)
                self.model.solvent_gather.load_state_dict(state_dict)

            # For dual pretrained models
            else:
                print("Loading Dual Pretrained Models...")
                PATH = "./pretrained_weights/2d/{}".format(args.pretrained_path)
                state_dict = torch.load(PATH, map_location=self.device)
                self.model.load_state_dict(state_dict)
            
        else:
            state_dict = None        

        self.optimizer = optim.Adam(params = self.model.parameters(), lr = self.args.lr, weight_decay = self.args.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, patience=self.args.patience, mode='min', verbose=True)
        
    def train(self):
        
        loss_fn = torch.nn.MSELoss()
        
        for epoch in range(1, self.args.epochs + 1):
            self.model.train()
            self.train_loss = 0

            for bc, samples in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                masks = create_batch_mask(samples)
                
                outputs, _ = self.model([samples[0].to(self.device), samples[1].to(self.device), masks[0].to(self.device), masks[1].to(self.device)])
                loss = loss_fn(outputs, samples[2].reshape(-1, 1).to(self.device).float())
                
                loss.backward()
                self.optimizer.step()
                self.train_loss += loss
                
            self.model.eval()
            self.evaluate(epoch)

            self.scheduler.step(self.val_loss)
            
            # Early stopping
            if len(self.best_val_losses) > int(self.args.es / self.args.eval_freq):
                if self.best_val_losses[-1] == self.best_val_losses[-int(self.args.es / self.args.eval_freq)]:
                    self.is_early_stop = True
                    break

        self.evaluate(epoch, final = True)
        self.writer.close()
        
        return self.best_test_loss, self.best_test_mae_loss, self.best_test_r2


class DualAttentiveFP(nn.Module):
    """
    This the main class for CIGIN model
    """

    def __init__(self,
                node_input_dim=56,
                edge_input_dim=10,
                node_hidden_dim=56,
                edge_hidden_dim=56,
                projector_dim = 64,
                num_step_message_passing=3,
                num_step_set2_set=2,
                num_layer_set2set=1,
                ):
        super(DualAttentiveFP, self).__init__()

        self.node_input_dim = node_input_dim
        self.node_hidden_dim = node_hidden_dim
        self.edge_input_dim = edge_input_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.num_step_message_passing = num_step_message_passing

        self.solute_gather = AttentiveFP(self.node_input_dim, self.node_hidden_dim,
                                    self.node_hidden_dim, self.edge_input_dim,
                                    self.num_step_message_passing, num_timesteps = 2
                                    )
        self.solvent_gather = AttentiveFP(self.node_input_dim, self.node_hidden_dim,
                                    self.node_hidden_dim, self.edge_input_dim,
                                    self.num_step_message_passing, num_timesteps = 2
                                    )

        self.projector = nn.Linear(self.node_hidden_dim * 2, projector_dim)
 
        self.fc1 = nn.Linear(2 * self.node_hidden_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.imap = nn.Linear(80, 1)

        self.num_step_set2set = num_step_set2_set
        self.num_layer_set2set = num_layer_set2set
        self.set2set_solute = Set2Set(self.node_hidden_dim, self.num_step_set2set, self.num_layer_set2set)
        self.set2set_solvent = Set2Set(self.node_hidden_dim, self.num_step_set2set, self.num_layer_set2set)

        self.init_model()
    
    def init_model(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def get_representations(self, data, test = False):
        
        solute = data[0]
        solvent = data[1]

        # node embeddings after interaction phase
        solute_feat, solute_features = self.solute_gather.get_repr(solute)
        solvent_feat, solvent_features = self.solvent_gather.get_repr(solvent)

        final_features = torch.cat((solute_feat, solvent_feat), 1)
        
        return self.projector(final_features), solute_features, solvent_features
    
    def forward(self, data, test = False):

        solute = data[0]
        solvent = data[1]
        
        # node embeddings after interaction phase
        solute_features = self.solute_gather(solute)
        solvent_features = self.solvent_gather(solvent)

        final_features = torch.cat((solute_features, solvent_features), 1)
        predictions = torch.relu(self.fc1(final_features))
        predictions = torch.relu(self.fc2(predictions))
        predictions = self.fc3(predictions)
        
        return predictions, None
    

if __name__ == "__main__":

    from utils import argument
    from datasets_eval import Chromophore

    args, unknown = argument.parse_args()

    dataset = Chromophore(args.data_path, args.dataset, args.log_target)
    dataloader_class = pyg_DataLoader
    dataloader = dataloader_class(dataset, batch_size=args.batch_size, shuffle=True)

    from models_2d import AttentiveFP_ModelTrainer
    embedder = AttentiveFP_ModelTrainer(args, dataset, dataset, dataset, 0, 0)

    best_mse, best_mae = embedder.train()