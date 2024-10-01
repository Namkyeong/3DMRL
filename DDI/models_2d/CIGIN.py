import sys
sys.path.append('.')

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

from torch_geometric.nn import Set2Set

from layers import GINE
from embedder import embedder
from utils.utils import create_batch_mask

from torch_geometric.loader import DataLoader as pyg_DataLoader


class CIGIN_ModelTrainer(embedder):
    def __init__(self, args, train_df, valid_df, test_df, repeat):
        embedder.__init__(self, args, train_df, valid_df, test_df, repeat)

        self.model = CIGIN(num_step_message_passing = self.args.message_passing).to(self.device)

        if args.pretrained:

            # For single pretrained models
            if "Single" in args.pretrained_path:
                print("Loading Single Pretrained Models...")
                PATH = "./pretrained_weights_single/2d/{}".format(args.pretrained_path)
                state_dict = torch.load(PATH, map_location=self.device)
                self.model.gather.load_state_dict(state_dict)

            # For dual pretrained models
            else:
                print("Loading Dual Pretrained Models...")
                PATH = "./pretrained_weights/2d/{}".format(args.pretrained_path)
                state_dict = torch.load(PATH, map_location=self.device)
                self.model.load_state_dict(state_dict)
            
        else:
            state_dict = None        

        self.optimizer = optim.Adam(params = self.model.parameters(), lr = self.args.lr, weight_decay = self.args.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, patience=self.args.patience, mode='max', verbose=True)
        
    def train(self):
        
        loss_function_BCE = nn.BCEWithLogitsLoss(reduction='none')
        
        for epoch in range(1, self.args.epochs + 1):
            self.model.train()
            self.train_loss = 0

            for bc, samples in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                masks = create_batch_mask(samples)
                
                outputs, _ = self.model([samples[0].to(self.device), samples[1].to(self.device), masks[0].to(self.device), masks[1].to(self.device)])
                loss = loss_function_BCE(outputs, samples[2].reshape(-1, 1).to(self.device).float()).mean()
                
                loss.backward()
                self.optimizer.step()
                self.train_loss += loss
                
            self.model.eval()
            self.evaluate(epoch)

            self.scheduler.step(self.val_roc_score)
            
            # Early stopping
            if len(self.best_val_rocs) > int(self.args.es / self.args.eval_freq):
                if self.best_val_rocs[-1] == self.best_val_rocs[-int(self.args.es / self.args.eval_freq)]:
                    if self.best_val_accs[-1] == self.best_val_accs[-int(self.args.es / self.args.eval_freq)]:
                        self.is_early_stop = True
                        break

        self.evaluate(epoch, final = True)
        self.writer.close()
        
        return self.best_test_roc, self.best_test_ap, self.best_test_f1, self.best_test_acc


class CIGIN(nn.Module):
    """
    This the main class for CIGIN model
    """

    def __init__(self,
                node_input_dim=134,
                edge_input_dim=14,
                node_hidden_dim=300,
                edge_hidden_dim=300,
                projector_dim = 64,
                num_step_message_passing=3,
                num_step_set2_set=2,
                num_layer_set2set=1,
                ):
        super(CIGIN, self).__init__()

        self.node_input_dim = node_input_dim
        self.node_hidden_dim = node_hidden_dim
        self.edge_input_dim = edge_input_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.num_step_message_passing = num_step_message_passing

        self.gather = GINE(self.node_input_dim, self.edge_input_dim, 
                            self.node_hidden_dim, self.num_step_message_passing,
                            )

        self.projector = nn.Linear(self.node_hidden_dim * 8, projector_dim)

        self.predictor = nn.Linear(8 * self.node_hidden_dim, 1)

        self.num_step_set2set = num_step_set2_set
        self.num_layer_set2set = num_layer_set2set
        self.set2set = Set2Set(2 * self.node_hidden_dim, self.num_step_set2set, self.num_layer_set2set)

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
        solute_len = data[2]
        solvent_len = data[3]
        # node embeddings after interaction phase
        solute_features = self.gather(solute)
        solvent_features = self.gather(solvent)

        # Add normalization
        self.solute_features = F.normalize(solute_features, dim = 1)
        self.solvent_features = F.normalize(solvent_features, dim = 1)

        # Interaction phase
        len_map = torch.sparse.mm(solute_len.t(), solvent_len)

        interaction_map = torch.mm(self.solute_features, self.solvent_features.t())
        interaction_map = interaction_map * len_map.to_dense()

        self.solvent_prime = torch.mm(interaction_map.t(), self.solute_features)
        self.solute_prime = torch.mm(interaction_map, self.solvent_features)

        # Prediction phase
        solute_features = torch.cat((self.solute_features, self.solute_prime), dim=1)
        solvent_features = torch.cat((self.solvent_features, self.solvent_prime), dim=1)

        self.solute_features_s2s = self.set2set(solute_features, solute.batch)
        self.solvent_features_s2s = self.set2set(solvent_features, solvent.batch)

        final_features = torch.cat((self.solute_features_s2s, self.solvent_features_s2s), 1)

        return self.projector(final_features), solute_features, solvent_features
    
    def forward(self, data, test = False):
        solute = data[0]
        solvent = data[1]
        solute_len = data[2]
        solvent_len = data[3]
        # node embeddings after interaction phase
        solute_features = self.gather(solute)
        solvent_features = self.gather(solvent)

        # Add normalization
        self.solute_features = F.normalize(solute_features, dim = 1)
        self.solvent_features = F.normalize(solvent_features, dim = 1)

        # Interaction phase
        len_map = torch.sparse.mm(solute_len.t(), solvent_len)

        interaction_map = torch.mm(self.solute_features, self.solvent_features.t())
        interaction_map = interaction_map * len_map.to_dense()

        self.solvent_prime = torch.mm(interaction_map.t(), self.solute_features)
        self.solute_prime = torch.mm(interaction_map, self.solvent_features)

        # Prediction phase
        solute_features = torch.cat((self.solute_features, self.solute_prime), dim=1)
        solvent_features = torch.cat((self.solvent_features, self.solvent_prime), dim=1)

        self.solute_features_s2s = self.set2set(solute_features, solute.batch)
        self.solvent_features_s2s = self.set2set(solvent_features, solvent.batch)

        final_features = torch.cat((self.solute_features_s2s, self.solvent_features_s2s), 1)
        predictions = self.predictor(final_features)
        
        return predictions, None
    

if __name__ == "__main__":

    from utils import argument
    from datasets_eval import DrugBank

    args, unknown = argument.parse_args()

    dataset = DrugBank(args.data_path)
    dataloader_class = pyg_DataLoader
    dataloader = dataloader_class(dataset, batch_size=args.batch_size, shuffle=True)

    from models_2d import CIGIN_ModelTrainer
    embedder = CIGIN_ModelTrainer(args, dataset, dataset, dataset, 0, 0)

    best_mse, best_mae = embedder.train()