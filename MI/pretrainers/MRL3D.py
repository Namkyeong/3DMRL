import sys
sys.path.append('.')

from tqdm import tqdm
from utils.utils import create_batch_mask

import torch
from torch import nn
from torch import optim

import numpy as np

from models_2d import CIGIN
from models_3d import GeomGNN
from pretrainer import pretrainer
from layers import EquivariantScoreNetwork, EquiLayer
from utils.utils import coord2basis_SDE

from typing import Optional, List, Tuple

EPSILON = 1e-5



class MRL3D_PreTrainer(pretrainer):

    def __init__(self, args, train_data):
        pretrainer.__init__(self, args, train_data)

        self.model_2d = CIGIN(num_step_message_passing = self.args.message_passing).to(self.device)
        self.model_3d = GeomGNN(hidden_dim = 64, num_interactions = args.t_message_passing, cutoff = args.cut_off, 
                                only_solute = args.only_solute, no_node_feature = args.no_node_feature).to(self.device)
        self.denoiser = Denoiser(emb_dim = 56, hidden_dim=56, device = self.device)
        self.pos_normalizer = AccumulatedNormalization(accumulator_shape=(3,))
        
        model_param_group = [
                {"params": self.model_2d.parameters(), "lr": args.lr},
                {"params": self.model_3d.parameters(), "lr": args.lr},
            ]
        
        self.optimizer = optim.Adam(model_param_group)

    
    def measure_gpu_memory(self):
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize()
        start_memory = torch.cuda.memory_allocated(self.device)
        start_max_memory = torch.cuda.max_memory_allocated(self.device)
        return start_memory, start_max_memory


    def contrastive_loss(self, rep_a, rep_b):

        batch_size, _ = rep_a.size()
        rep_a_abs = rep_a.norm(dim = 1)
        rep_b_abs = rep_b.norm(dim = 1)

        sim_matrix = torch.einsum('ik,jk->ij', rep_a, rep_b) / torch.einsum('i,j->ij', rep_a_abs, rep_b_abs)
        sim_matrix = torch.exp(sim_matrix / self.args.tau)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]

        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        return loss

    def train(self):

        loss_fn = torch.nn.MSELoss()

        for epoch in tqdm(range(1, self.args.epochs + 1), desc='Epochs'):

            self.model_2d.train()
            self.model_3d.train()
            
            accm_loss, accm_cont_loss, accm_denoising_loss = 0, 0, 0

            for bc, samples in enumerate(tqdm(self.loader, desc='Batches', leave = False)):

                self.optimizer.zero_grad()
                masks = create_batch_mask(samples)                
                
                rep_2d, solute_2d, solvent_2d = self.model_2d.get_representations([samples[0].to(self.device), samples[1].to(self.device), masks[0].to(self.device), masks[1].to(self.device)])
                rep_3d = self.model_3d(samples[2].to(self.device))

                cont_loss = self.contrastive_loss(rep_2d, rep_3d)
                _cont_loss = self.contrastive_loss(rep_3d, rep_2d)
                accm_cont_loss += cont_loss + _cont_loss

                denoising_loss = self.denoiser(solvent_2d, solute_2d, samples[2])
                accm_denoising_loss += denoising_loss

                loss = (cont_loss + _cont_loss) + self.args.alpha * denoising_loss
                accm_loss += loss

                loss.backward()
                self.optimizer.step()

            
            self.writer.add_scalar("loss/train", accm_loss/bc, epoch)
            self.writer.add_scalar("loss/contrastive train", accm_cont_loss/bc, epoch)
            self.writer.add_scalar("loss/denoising train", accm_denoising_loss/bc, epoch)
            
            if epoch % self.args.save_freq == 0:
                if self.args.save_checkpoints:
                    self.save_checkpoints(epoch)
                else:
                    pass


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, embedding_size, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Denoiser(nn.Module):
    def __init__(self, emb_dim, hidden_dim, device):

        super(Denoiser, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        self.coff_gaussian_fourier = GaussianFourierProjection(embedding_size=self.hidden_dim, scale=1).to(device)
        self.coff_mlp = nn.Linear(4 * self.hidden_dim, self.hidden_dim).to(device)

        self.node_emb = nn.Linear(self.emb_dim * 4, self.hidden_dim).to(device)
        self.edge_2D_emb = nn.Sequential(nn.Linear(self.emb_dim * 4, self.emb_dim), nn.BatchNorm1d(self.emb_dim), nn.SiLU(), nn.Linear(self.emb_dim, self.hidden_dim)).to(device)
        self.project = nn.Sequential(nn.Linear(self.hidden_dim * 2 + 2, self.hidden_dim), nn.SiLU(), nn.Linear(self.emb_dim, self.hidden_dim)).to(device)

        self.basis_mlp = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.SiLU(), nn.Linear(self.hidden_dim, 3)).to(device)

        self.score_network = EquivariantScoreNetwork(hidden_dim=self.hidden_dim, hidden_coff_dim=128, activation="silu").to(device)

    def get_embedding(self, coff_index):
        coff_embeds = []
        for i in [0, 2]:  # if i=1, then x=0
            coff_embeds.append(self.coff_gaussian_fourier(coff_index[:, i:i + 1]))  # [E, 2C]
        coff_embeds = torch.cat(coff_embeds, dim=-1)  # [E, 6C]
        coff_embeds = self.coff_mlp(coff_embeds)

        return coff_embeds

    def forward(self, solvent_repr, solute_repr, data):
        
        expanded_solvent_repr = solvent_repr[data.solvent_2d_idx[data.solute == 0]]
        
        # Get representation of solute that solvent assigned
        assigned_solute_repr = solute_repr[data.solute_2d_idx[data.solute == 0]]

        row, col = data.edge_index
        edge_attr_2D = torch.cat([expanded_solvent_repr[row], expanded_solvent_repr[col]], dim=-1)
        edge_attr_2D = self.edge_2D_emb(edge_attr_2D)

        solvent_pos = data.pos[data.solute == 0]
        
        # construct geometric features
        coord_diff, coord_cross, coord_vertical = coord2basis_SDE(solvent_pos, row, col)  # [num_edge, 3] * 3
        equivariant_basis = [coord_diff, coord_cross, coord_vertical]
        edge_basis = torch.cat([coord_diff.unsqueeze(1), coord_cross.unsqueeze(1), coord_vertical.unsqueeze(1)], dim=1)  # [num_edge, 3, 3]
        r_i, r_j = solvent_pos[row], solvent_pos[col]  # [num_edge, 3]
        coff_i = torch.matmul(edge_basis, r_i.unsqueeze(-1)).squeeze(-1)  # [num_edge, 3]
        coff_j = torch.matmul(edge_basis, r_j.unsqueeze(-1)).squeeze(-1)  # [num_edge, 3]
        coff_i[:, 1] = torch.abs(coff_i[:, 1].clone())
        coff_j[:, 1] = torch.abs(coff_j[:, 1].clone())
        coff_mul = coff_i * coff_j  # [num_edge, 3]
        coff_i_norm = coff_i.norm(dim=-1, keepdim=True) # [num_edge, 1]
        coff_j_norm = coff_j.norm(dim=-1, keepdim=True) # [num_edge, 1]
        pseudo_cos = coff_mul.sum(dim=-1, keepdim=True) / (coff_i_norm + EPSILON) / (coff_j_norm + EPSILON)
        pseudo_sin = torch.sqrt(1 - pseudo_cos ** 2)
        pseudo_angle = torch.cat([pseudo_sin, pseudo_cos], dim=-1)  # [num_edge, 2]
        embed_i = self.get_embedding(coff_i)  # [num_edge, C]
        embed_j = self.get_embedding(coff_j)  # [num_edge, C]
        edge_embed = torch.cat([pseudo_angle, embed_i, embed_j], dim=-1)
        edge_attr_3D_frame_invariant = self.project(edge_embed)

        edge_attr = edge_attr_2D + edge_attr_3D_frame_invariant

        # Incorporate solute representation into the node attribute
        node_attr = torch.cat([expanded_solvent_repr, assigned_solute_repr], dim = 1)

        # match dimension
        node_attr = self.node_emb(node_attr)

        # estimate scores
        output = self.score_network(data.edge_index, node_attr, edge_attr, equivariant_basis)
        scores = output["gradient"]

        pos_noise = data.pos_target[data.solute == 0]
        loss_pos = torch.sum((scores - pos_noise) ** 2, -1)
        
        return loss_pos.mean()


class AccumulatedNormalization(nn.Module):
    """Running normalization of a tensor."""
    def __init__(self, accumulator_shape: Tuple[int, ...], epsilon: float = 1e-8):
        super(AccumulatedNormalization, self).__init__()

        self._epsilon = epsilon
        self.register_buffer("acc_sum", torch.zeros(accumulator_shape))
        self.register_buffer("acc_squared_sum", torch.zeros(accumulator_shape))
        self.register_buffer("acc_count", torch.zeros((1,)))
        self.register_buffer("num_accumulations", torch.zeros((1,)))

    def update_statistics(self, batch: torch.Tensor):
        batch_size = batch.shape[0]
        self.acc_sum += batch.sum(dim=0)
        self.acc_squared_sum += batch.pow(2).sum(dim=0)
        self.acc_count += batch_size
        self.num_accumulations += 1

    @property
    def acc_count_safe(self):
        return self.acc_count.clamp(min=1)

    @property
    def mean(self):
        return self.acc_sum / self.acc_count_safe

    @property
    def std(self):
        return torch.sqrt(
            (self.acc_squared_sum / self.acc_count_safe) - self.mean.pow(2)
        ).clamp(min=self._epsilon)

    def forward(self, batch: torch.Tensor):
        if self.training:
            self.update_statistics(batch)
        return ((batch - self.mean) / self.std)



if __name__ == "__main__":

    from utils import argument

    args, unknown = argument.pretrain_parse_args()

    from datasets_pretrain import ChromophoreAll
    dataset = ChromophoreAll(args.data_path, args.rotation, args.radius, args.fixed_direction, args.sample)

    from pretrainers import MRL3D
    embedder = MRL3D_PreTrainer(args, dataset)

    best_mse, best_mae = embedder.train()