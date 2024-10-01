import abc
import numpy as np
import torch
from torch import nn, Tensor
from torch_scatter import scatter_mean
import torch.nn as nn
import torch.nn.functional as F

from typing import Union
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size

from torch_sparse import SparseTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import TransformerConv

EPSILON = 1e-6


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation="relu", dropout=0):
        super(MultiLayerPerceptron, self).__init__()

        self.dims = [input_dim] + hidden_dims
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None
        # print("activation in MultiLayerPerceptron", self.activation)
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))

        self.reset_parameters()

    def reset_parameters(self):
        for i, layer in enumerate(self.layers):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.)

    def forward(self, input):
        x = input
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                if self.activation:
                    x = self.activation(x)
                if self.dropout:
                    x = self.dropout(x)
        return x


class GATLayer(nn.Module):
    def __init__(self, n_head, hidden_dim, dropout=0.2):
        super(GATLayer, self).__init__()

        assert hidden_dim % n_head == 0
        self.MHA = TransformerConv(
            in_channels=hidden_dim,
            out_channels=int(hidden_dim // n_head),
            heads=n_head,
            dropout=dropout,
            edge_dim=hidden_dim,
        )
        self.FFN = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, edge_index, node_attr, edge_attr):
        x = self.MHA(node_attr, edge_index, edge_attr)
        node_attr = node_attr + self.norm1(x)
        x = self.FFN(node_attr)
        node_attr = node_attr + self.norm2(x)
        
        return node_attr


class EquiLayer(MessagePassing):
    def __init__(self, eps=0., train_eps=False, activation="silu", **kwargs):
        super(EquiLayer, self).__init__(aggr='mean', **kwargs)
        self.initial_eps = eps

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None   

        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # Node and edge feature dimensionalites need to match.
        if isinstance(edge_index, Tensor):
            assert edge_attr is not None
            # assert x[0].size(-1) == edge_attr.size(-1)
        elif isinstance(edge_index, SparseTensor):
            assert x[0].size(-1) == edge_index.size(-1)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        return out

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor: 
        if self.activation:
            return self.activation(x_j + edge_attr)
        else: # TODO: we are mostly using False for activation
            return edge_attr

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class EquivariantScoreNetwork(torch.nn.Module):
    def __init__(self, hidden_dim, hidden_coff_dim=64, activation="silu", short_cut=False, concat_hidden=False):
        super(EquivariantScoreNetwork, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = 2
        self.num_convs = 2
        self.short_cut = short_cut
        self.num_head = 8
        self.dropout = 0.1
        self.concat_hidden = concat_hidden
        self.hidden_coff_dim = hidden_coff_dim

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None 
        
        self.gnn_layers = nn.ModuleList()
        self.equi_modules = nn.ModuleList()
        self.basis_mlp_modules = nn.ModuleList()
        for _ in range(self.num_layers):
            trans_convs = nn.ModuleList()
            for _ in range(self.num_convs):
                trans_convs.append(GATLayer(self.num_head, self.hidden_dim, dropout=self.dropout))
            self.gnn_layers.append(trans_convs)

            self.equi_modules.append(EquiLayer(activation=False))

            self.basis_mlp_modules.append(
                nn.Sequential(
                nn.Linear(2 * self.hidden_dim, self.hidden_coff_dim),
                # nn.Softplus(),
                nn.SiLU(),
                nn.Linear(self.hidden_coff_dim, 3))
            )

    def forward(self, edge_index, node_attr, edge_attr, equivariant_basis):
        """
        Args:
            edge_index: edge connection (num_node, 2)
            node_attr: node feature tensor with shape (num_node, hidden)
            edge_attr: edge feature tensor with shape (num_edge, hidden)
            equivariant_basis: an equivariant basis coord_diff, coord_cross, coord_vertical
        Output:
            gradient (score)
        """
        hiddens = []
        conv_input = node_attr # (num_node, hidden)
        coord_diff, coord_cross, coord_vertical = equivariant_basis

        for module_idx, gnn_layers in enumerate(self.gnn_layers):

            for conv_idx, gnn in enumerate(gnn_layers):
                hidden = gnn(edge_index, conv_input, edge_attr)

                if conv_idx < len(gnn_layers) - 1 and self.activation is not None:
                    hidden = self.activation(hidden)
                assert hidden.shape == conv_input.shape                
                if self.short_cut and hidden.shape == conv_input.shape:
                    hidden += conv_input

                hiddens.append(hidden)
                conv_input = hidden

            if self.concat_hidden:
                node_feature = torch.cat(hiddens, dim=-1)
            else:
                node_feature = hiddens[-1]

            h_row, h_col = node_feature[edge_index[0]], node_feature[edge_index[1]] # (num_edge, hidden)
            edge_feature = torch.cat([h_row + h_col, edge_attr], dim=-1)  # (num_edge, 2 * hidden)

            # generate gradient
            dynamic_coff = self.basis_mlp_modules[module_idx](edge_feature)  # (num_edge, 3)
            basis_mix = dynamic_coff[:, :1] * coord_diff + dynamic_coff[:, 1:2] * coord_cross + dynamic_coff[:, 2:3] * coord_vertical  # (num_edge, 3)

            if module_idx == 0:
                gradient = self.equi_modules[module_idx](node_feature, edge_index, basis_mix)
            else:
                gradient += self.equi_modules[module_idx](node_feature, edge_index, basis_mix)

        return {
            "node_feature": node_feature,
            "gradient": gradient
        }


class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, N):
        """Construct an SDE.
        Args:
            N: number of discretization time steps.
        """
        super().__init__()
        self.N = N

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""
        pass

    @abc.abstractmethod
    def sde(self, x, t):
        pass

    @abc.abstractmethod
    def marGINal_prob(self, x, t):
        """Parameters to determine the marGINal distribution of the SDE, $p_t(x)$."""
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape):
        """Generate one sample from the prior distribution, $p_T(x)$."""
        pass

    @abc.abstractmethod
    def prior_logp(self, z):
        """Compute log-density of the prior distribution.
        Useful for computing the log-likelihood via probability flow ODE.
        Args:
            z: latent code
        Returns:
            log probability density
        """
        pass

    def discretize(self, x, t):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.
        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.
        Args:
            x: a torch tensor
            t: a torch float representing the time step (from 0 to `self.T`)
        Returns:
            f, G
        """
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return f, G

    def reverse(self, score_fn, probability_flow=False):
        """Create the reverse-time SDE/ODE.
        Args:
            score_fn: A time-dependent score-based model that takes x and t and returns the score.
            probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        N = self.N
        T = self.T
        sde_fn = self.sde
        discretize_fn = self.discretize

        # -------- Build the class for reverse-time SDE --------
        class RSDE(self.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, representation, data, t):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""
                drift, diffusion = sde_fn(x, t)
                score = score_fn.get_score(representation, data, x, None, t)
                drift = drift - diffusion[:, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
                # -------- Set the diffusion function to zero for ODEs. --------
                diffusion = 0. if self.probability_flow else diffusion
                return drift, diffusion

            def discretize(self, x, representation, data, t):
                """Create discretized iteration rules for the reverse diffusion sampler."""
                f, G = discretize_fn(x, t)
                score = score_fn.get_score(representation, data, x, None, t)
                rev_f = f - G[:, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()


class VPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        """Construct a Variance Preserving SDE.
        Args:
            beta_min: value of beta(0)
            beta_max: value of beta(1)
            N: number of discretization steps
        """
        super().__init__(N)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    # -------- mean, std of the perturbation kernel --------
    def marGINal_prob(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        #mean = torch.exp(log_mean_coeff[:, None, None]) * x
        mean = torch.exp(log_mean_coeff[:, None]) * x
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_sampling_sym(self, shape):
        x = torch.randn(*shape).triu(1)
        return x + x.transpose(-1,-2)

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        loGPS = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2)) / 2.
        return loGPS

    def discretize(self, x, t):
        """DDPM discretization."""
        timestep = (t * (self.N - 1) / self.T).long()
        beta = self.discrete_betas.to(x.device)[timestep]
        alpha = self.alphas.to(x.device)[timestep]
        sqrt_beta = torch.sqrt(beta)
        f = torch.sqrt(alpha)[:, None] * x - x
        G = sqrt_beta
        return f, G

    def transition(self, x, t, dt):
        # -------- negative timestep dt --------
        log_mean_coeff = 0.25 * dt * (2*self.beta_0 + (2*t + dt)*(self.beta_1 - self.beta_0) )
        mean = torch.exp(-log_mean_coeff[:, None, None]) * x
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std


class VESDE(SDE):
    def __init__(self, sigma_min=0.01, sigma_max=50, N=1000):
        """Construct a Variance Exploding SDE.
        Args:
            sigma_min: smallest sigma.
            sigma_max: largest sigma.
            N: number of discretization steps
        """
        super().__init__(N)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))
        self.N = N

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = torch.zeros_like(x)
        diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)), device=t.device))
        return drift, diffusion

    def marGINal_prob(self, x, t):
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = x
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_sampling_sym(self, shape):
        x = torch.randn(*shape).triu(1)
        x = x + x.transpose(-1,-2)
        return x

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=(1, 2, 3)) / (2 * self.sigma_max ** 2)

    def discretize(self, x, t):
        """SMLD(NCSN) discretization."""
        timestep = (t * (self.N - 1) / self.T).long()
        sigma = self.discrete_sigmas.to(t.device)[timestep]
        adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t),
                                                                 self.discrete_sigmas[timestep - 1].to(t.device))
        f = torch.zeros_like(x)
        G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
        return f, G

    def transition(self, x, t, dt):
        # -------- negative timestep dt --------
        std = torch.square(self.sigma_min * (self.sigma_max / self.sigma_min) ** t) - \
                    torch.square(self.sigma_min * (self.sigma_max / self.sigma_min) ** (t + dt))
        std = torch.sqrt(std)
        mean = x
        return mean, std


class subVPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        """Construct the sub-VP SDE that excels at likelihoods.
        Args:
            beta_min: value of beta(0)
            beta_max: value of beta(1)
            N: number of discretization steps
        """
        super().__init__(N)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1. - self.discrete_betas

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None] * x
        discount = 1. - torch.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2)
        diffusion = torch.sqrt(beta_t * discount)
        return drift, diffusion

    def marGINal_prob(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff)[:, None, None] * x
        std = 1 - torch.exp(2. * log_mean_coeff)
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_sampling_sym(self, shape):
        x = torch.randn(*shape).triu(1)
        return x + x.transpose(-1,-2)

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.


def get_beta_schedule(beta_schedule, *, beta_min, beta_max, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_min**0.5, beta_max**0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_min, beta_max, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_max * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_max - beta_min) + beta_min
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    betas = torch.from_numpy(betas).float()
    return betas


def coord2basis(pos, row, col):
    coord_diff = pos[row] - pos[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    coord_cross = torch.cross(pos[row], pos[col])

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


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, embedding_size, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class SDEModel2Dto3D_01(torch.nn.Module):
    def __init__(
        self, emb_dim, hidden_dim,
        beta_schedule, beta_min, beta_max, num_diffusion_timesteps, SDE_type="VE",
        short_cut=False, concat_hidden=False, use_extend_graph=False):

        super(SDEModel2Dto3D_01, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.SDE_type = SDE_type
        self.use_extend_graph = use_extend_graph

        self.node_emb = MultiLayerPerceptron(self.emb_dim, [self.hidden_dim], activation="silu")
        self.edge_2D_emb = nn.Sequential(nn.Linear(self.emb_dim*2, self.emb_dim), nn.BatchNorm1d(self.emb_dim), nn.ReLU(), nn.Linear(self.emb_dim, self.hidden_dim))

        self.coff_gaussian_fourier = GaussianFourierProjection(embedding_size=self.hidden_dim, scale=1)
        self.coff_mlp = nn.Linear(4 * self.hidden_dim, self.hidden_dim)
        self.project = MultiLayerPerceptron(2 * self.hidden_dim + 2, [self.hidden_dim, self.hidden_dim], activation="silu")

        self.score_network = EquivariantScoreNetwork(hidden_dim=self.hidden_dim, hidden_coff_dim=128, activation="silu", short_cut=short_cut, concat_hidden=concat_hidden)

        if self.SDE_type in ["VE", "VE_test"]:
            self.sde_pos = VESDE(sigma_min=beta_min, sigma_max=beta_max, N=num_diffusion_timesteps)
        elif self.SDE_type in ["VP", "VP_test"]:
            self.sde_pos = VPSDE(beta_min=beta_min, beta_max=beta_max, N=num_diffusion_timesteps)
        elif self.SDE_type == "discrete_VE":
            betas = get_beta_schedule(
                beta_schedule=beta_schedule,
                beta_min=beta_min,
                beta_max=beta_max,
                num_diffusion_timesteps=num_diffusion_timesteps,
            )
            self.betas = nn.Parameter(betas, requires_grad=False)
            # variances
            alphas = (1. - betas).cumprod(dim=0)
            self.alphas = nn.Parameter(alphas, requires_grad=False)
            # print("betas used in 2D to 3D diffusion model", self.betas)
            # print("alphas used in 2D to 3D diffusion model", self.alphas)

        self.num_diffusion_timesteps = num_diffusion_timesteps
        return

    def get_embedding(self, coff_index):
        coff_embeds = []
        for i in [0, 2]:  # if i=1, then x=0
            coff_embeds.append(self.coff_gaussian_fourier(coff_index[:, i:i + 1]))  # [E, 2C]
        coff_embeds = torch.cat(coff_embeds, dim=-1)  # [E, 6C]
        coff_embeds = self.coff_mlp(coff_embeds)

        return coff_embeds

    def forward(self, node_2D_repr, data, anneal_power):
        pos = data.positions
        pos.requires_grad = True

        # data = self.get_distance(data)
        node2graph = data.batch
        if self.use_extend_graph:
            extended_edge_index = data.extended_edge_index
        else:
            extended_edge_index = data.edge_index

        # Perterb pos
        pos_noise = torch.randn_like(pos)

        # sample variances
        time_step = torch.randint(0, self.num_diffusion_timesteps, size=(data.num_graphs // 2 + 1,), device=pos.device)
        time_step = torch.cat([time_step, self.num_diffusion_timesteps - time_step - 1], dim=0)[:data.num_graphs]  # (num_graph, )

        if self.SDE_type in ["VE", "VP"]:
            time_step = time_step / self.num_diffusion_timesteps * (1 - EPSILON) + EPSILON  # normalize to [0, 1]
            time_step = time_step.squeeze(-1)
            t_pos = time_step.index_select(0, node2graph)  # (num_graph, )
            mean_pos, std_pos = self.sde_pos.marGINal_prob(pos, t_pos)
            pos_perturbed = mean_pos + std_pos[:, None] * pos_noise
            
        elif self.SDE_type in ["VE_test", "VP_test"]:
            time_step = time_step.squeeze(-1)
            t_pos = time_step.index_select(0, node2graph)  # (num_graph, )
            mean_pos, std_pos = self.sde_pos.marGINal_prob(pos, t_pos)
            pos_perturbed = mean_pos + std_pos[:, None] * pos_noise

        elif self.SDE_type == "discrete_VE":
            a = self.alphas.index_select(0, time_step)  # (num_graph, )
            a_pos = a.index_select(0, node2graph).unsqueeze(-1)  # (num_nodes, 1)
            pos_perturbed = pos + pos_noise * (1.0 - a_pos).sqrt() / a_pos.sqrt()
        
        # edge_attr from 2D represenattion node_2D_repr
        row, col = extended_edge_index
        edge_attr_2D = torch.cat([node_2D_repr[row], node_2D_repr[col]], dim=-1)
        edge_attr_2D = self.edge_2D_emb(edge_attr_2D)
        
        # construct geometric features
        coord_diff, coord_cross, coord_vertical = coord2basis(pos_perturbed, row, col)  # [num_edge, 3] * 3
        equivariant_basis = [coord_diff, coord_cross, coord_vertical]
        edge_basis = torch.cat([coord_diff.unsqueeze(1), coord_cross.unsqueeze(1), coord_vertical.unsqueeze(1)], dim=1)  # [num_edge, 3, 3]
        r_i, r_j = pos_perturbed[row], pos_perturbed[col]  # [num_edge, 3]
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

        # match dimension
        node_attr = self.node_emb(node_2D_repr)

        # estimate scores
        output = self.score_network(extended_edge_index, node_attr, edge_attr, equivariant_basis)
        scores = output["gradient"]
        if anneal_power == 0:
            loss_pos = torch.sum((scores - pos_noise) ** 2, -1)  # (num_node)
        else:
            annealed_std = std_pos ** anneal_power  # (num_node)
            annealed_std = annealed_std.unsqueeze(1,)  # (num_node,1)
            loss_pos = torch.sum((scores - pos_noise) ** 2 * annealed_std, -1)  # (num_node)
        loss_pos = scatter_mean(loss_pos, node2graph)  # (num_graph)

        loss_dict = {
            'position': loss_pos.mean(),
        }
        return loss_dict

    @torch.no_grad()
    def get_score(self, node_2D_repr, data, pos_perturbed, sigma, t_pos):
        node_attr = self.node_emb(node_2D_repr)
        
        if self.use_extend_graph:
            extended_edge_index = data.extended_edge_index
        else:
            extended_edge_index = data.edge_index
        
        # edge_attr from 2D represenattion node_2D_repr
        row, col = extended_edge_index        
        edge_attr_2D = torch.cat([node_2D_repr[row], node_2D_repr[col]], dim=-1)
        edge_attr_2D = self.edge_2D_emb(edge_attr_2D)
        
        # construct geometric features
        coord_diff, coord_cross, coord_vertical = coord2basis(pos_perturbed, row, col)  # [num_edge, 3] * 3
        equivariant_basis = [coord_diff, coord_cross, coord_vertical]
        edge_basis = torch.cat([coord_diff.unsqueeze(1), coord_cross.unsqueeze(1), coord_vertical.unsqueeze(1)], dim=1)  # [num_edge, 3, 3]
        r_i, r_j = pos_perturbed[row], pos_perturbed[col]  # [num_edge, 3]
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
        
        # match dimension
        node_attr = self.node_emb(node_2D_repr)
        
        # estimate scores
        output = self.score_network(extended_edge_index, node_attr, edge_attr, equivariant_basis)
        output = output["gradient"]
        scores = -output

        _, std_pos = self.sde_pos.marGINal_prob(pos_perturbed, t_pos)
        scores = scores / std_pos[:, None]
        # print(t_pos, std_pos)
        return scores


class SDEModel2Dto3D_02(torch.nn.Module):
    def __init__(
        self, emb_dim, hidden_dim,
        beta_schedule, beta_min, beta_max, num_diffusion_timesteps, SDE_type="VE",
        short_cut=False, concat_hidden=False, use_extend_graph=False):

        super(SDEModel2Dto3D_02, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.SDE_type = SDE_type
        self.use_extend_graph = use_extend_graph

        self.node_emb = MultiLayerPerceptron(self.emb_dim, [self.hidden_dim], activation="silu")
        self.edge_2D_emb = nn.Sequential(nn.Linear(self.emb_dim*2, self.emb_dim), nn.BatchNorm1d(self.emb_dim), nn.ReLU(), nn.Linear(self.emb_dim, self.hidden_dim))

        self.dist_gaussian_fourier = GaussianFourierProjection(embedding_size=self.hidden_dim, scale=1)
        self.input_mlp = MultiLayerPerceptron(2*self.hidden_dim, [self.hidden_dim], activation="silu")

        self.coff_gaussian_fourier = GaussianFourierProjection(embedding_size=self.hidden_dim, scale=1)
        self.coff_mlp = nn.Linear(4 * self.hidden_dim, self.hidden_dim)
        self.project = MultiLayerPerceptron(2 * self.hidden_dim + 2, [self.hidden_dim, self.hidden_dim], activation="silu")

        self.score_network = EquivariantScoreNetwork(hidden_dim=self.hidden_dim, hidden_coff_dim=128, activation="silu", short_cut=short_cut, concat_hidden=concat_hidden)

        if self.SDE_type in ["VE", "VE_test"]:
            self.sde_pos = VESDE(sigma_min=beta_min, sigma_max=beta_max, N=num_diffusion_timesteps)
        elif self.SDE_type in ["VP", "VP_test"]:
            self.sde_pos = VPSDE(beta_min=beta_min, beta_max=beta_max, N=num_diffusion_timesteps)
        elif self.SDE_type == "discrete_VE":
            betas = get_beta_schedule(
                beta_schedule=beta_schedule,
                beta_min=beta_min,
                beta_max=beta_max,
                num_diffusion_timesteps=num_diffusion_timesteps,
            )
            self.betas = nn.Parameter(betas, requires_grad=False)
            # variances
            alphas = (1. - betas).cumprod(dim=0)
            self.alphas = nn.Parameter(alphas, requires_grad=False)
            # print("betas used in 2D to 3D diffusion model", self.betas)
            # print("alphas used in 2D to 3D diffusion model", self.alphas)

        self.num_diffusion_timesteps = num_diffusion_timesteps
        return

    def get_embedding(self, coff_index):
        coff_embeds = []
        for i in [0, 2]:  # if i=1, then x=0
            coff_embeds.append(self.coff_gaussian_fourier(coff_index[:, i:i + 1]))  # [E, 2C]
        coff_embeds = torch.cat(coff_embeds, dim=-1)  # [E, 6C]
        coff_embeds = self.coff_mlp(coff_embeds)

        return coff_embeds

    def forward(self, node_2D_repr, data, anneal_power):
        pos = data.pos
        pos.requires_grad = True

        # data = self.get_distance(data)
        node2graph = data.batch
        if self.use_extend_graph:
            extended_edge_index = data.extended_edge_index
        else:
            extended_edge_index = data.edge_index
        
        # Perterb pos
        pos_noise = torch.randn_like(pos)

        # sample variances
        time_step = torch.randint(0, self.num_diffusion_timesteps, size=(data.num_graphs // 2 + 1,), device=pos.device)
        time_step = torch.cat([time_step, self.num_diffusion_timesteps - time_step - 1], dim=0)[:data.num_graphs]  # (num_graph, )

        if self.SDE_type in ["VE", "VP"]:
            time_step = time_step / self.num_diffusion_timesteps * (1 - EPSILON) + EPSILON  # normalize to [0, 1]
            time_step = time_step.squeeze(-1)
            t_pos = time_step.index_select(0, node2graph)  # (num_nodes, )
            mean_pos, std_pos = self.sde_pos.marGINal_prob(pos, t_pos)
            pos_perturbed = mean_pos + std_pos[:, None] * pos_noise
            
        elif self.SDE_type in ["VE_test", "VP_test"]:
            time_step = time_step.squeeze(-1)
            t_pos = time_step.index_select(0, node2graph)  # (num_graph, )
            mean_pos, std_pos = self.sde_pos.marGINal_prob(pos, t_pos)
            pos_perturbed = mean_pos + std_pos[:, None] * pos_noise

        elif self.SDE_type == "discrete_VE":
            a = self.alphas.index_select(0, time_step)  # (num_graph, )
            a_pos = a.index_select(0, node2graph).unsqueeze(-1)  # (num_nodes, 1)
            pos_perturbed = pos + pos_noise * (1.0 - a_pos).sqrt() / a_pos.sqrt()
        
        distance_perturbed = get_perturb_distance(pos_perturbed, extended_edge_index)

        # edge_attr should come from 2D represenattion x
        row, col = extended_edge_index
        edge_attr_2D = torch.cat([node_2D_repr[row], node_2D_repr[col]], dim=-1)
        edge_attr_2D = self.edge_2D_emb(edge_attr_2D)
        
        distance_perturbed_emb = self.dist_gaussian_fourier(distance_perturbed)  # (num_edge, hidden*2)
        edge_attr_3D_invariant = self.input_mlp(distance_perturbed_emb)  # (num_edge, hidden)

        # construct geometric features
        coord_diff, coord_cross, coord_vertical = coord2basis(pos_perturbed, row, col)  # [num_edge, 3] * 3
        equivariant_basis = [coord_diff, coord_cross, coord_vertical]
        edge_basis = torch.cat([coord_diff.unsqueeze(1), coord_cross.unsqueeze(1), coord_vertical.unsqueeze(1)], dim=1)  # [num_edge, 3, 3]
        r_i, r_j = pos_perturbed[row], pos_perturbed[col]  # [num_edge, 3]
        coff_i = torch.matmul(edge_basis, r_i.unsqueeze(-1)).squeeze(-1)  # [num_edge, 3]
        coff_j = torch.matmul(edge_basis, r_j.unsqueeze(-1)).squeeze(-1)  # [num_edge, 3]
        coff_i[:, 1] = torch.abs(coff_i[:, 1].clone())
        coff_j[:, 1] = torch.abs(coff_j[:, 1].clone())
        coff_mul = coff_i * coff_j  # [num_edge, 3]
        coff_i_norm = coff_i.norm(dim=-1, keepdim=True) # [num_edge, 1]
        coff_j_norm = coff_j.norm(dim=-1, keepdim=True) # [num_edge, 1]
        pseudo_cos = coff_mul.sum(dim=-1, keepdim=True) / (coff_i_norm + EPSILON) / (coff_j_norm + EPSILON)
        pseudo_sin = torch.sqrt(1 - pseudo_cos ** 2)
        pseudo_angle = torch.cat([pseudo_sin, pseudo_cos], dim=-1)
        embed_i = self.get_embedding(coff_i)  # [num_edge, C]
        embed_j = self.get_embedding(coff_j)  # [num_edge, C]
        edge_embed = torch.cat([pseudo_angle, embed_i, embed_j], dim=-1)
        edge_attr_3D_frame_invariant = self.project(edge_embed)
        
        edge_attr = edge_attr_3D_invariant * edge_attr_2D + edge_attr_3D_frame_invariant

        # match dimension
        node_attr = self.node_emb(node_2D_repr)

        # estimate scores
        output = self.score_network(extended_edge_index, node_attr, edge_attr, equivariant_basis)
        scores = output["gradient"]
        if anneal_power == 0:
            loss_pos = torch.sum((scores - pos_noise) ** 2, -1)  # (num_node)
        else:
            annealed_std = std_pos ** anneal_power  # (num_node)
            annealed_std = annealed_std.unsqueeze(1,)  # (num_node,1)
            loss_pos = torch.sum((scores - pos_noise) ** 2 * annealed_std, -1)  # (num_node)
        loss_pos = scatter_mean(loss_pos, node2graph)  # (num_graph)

        loss_dict = {
            'position': loss_pos.mean(),
        }
        return loss_dict

    @torch.no_grad()
    def get_score(self, node_2D_repr, data, pos_perturbed, sigma, t_pos):
        node_attr = self.node_emb(node_2D_repr)
        
        if self.use_extend_graph:
            extended_edge_index = data.extended_edge_index
        else:
            extended_edge_index = data.edge_index
        
        distance_perturbed = get_perturb_distance(pos_perturbed, extended_edge_index)

        # edge_attr from 2D represenattion node_2D_repr
        row, col = extended_edge_index        
        edge_attr_2D = torch.cat([node_2D_repr[row], node_2D_repr[col]], dim=-1)
        edge_attr_2D = self.edge_2D_emb(edge_attr_2D)
        
        distance_perturbed_emb = self.dist_gaussian_fourier(distance_perturbed)  # (num_edge, hidden*2)
        edge_attr_3D_invariant = self.input_mlp(distance_perturbed_emb)  # (num_edge, hidden)

        # construct geometric features
        coord_diff, coord_cross, coord_vertical = coord2basis(pos_perturbed, row, col)  # [num_edge, 3] * 3
        equivariant_basis = [coord_diff, coord_cross, coord_vertical]
        edge_basis = torch.cat([coord_diff.unsqueeze(1), coord_cross.unsqueeze(1), coord_vertical.unsqueeze(1)], dim=1)  # [num_edge, 3, 3]
        r_i, r_j = pos_perturbed[row], pos_perturbed[col]  # [num_edge, 3]
        coff_i = torch.matmul(edge_basis, r_i.unsqueeze(-1)).squeeze(-1)  # [num_edge, 3]
        coff_j = torch.matmul(edge_basis, r_j.unsqueeze(-1)).squeeze(-1)  # [num_edge, 3]
        coff_i[:, 1] = torch.abs(coff_i[:, 1].clone())
        coff_j[:, 1] = torch.abs(coff_j[:, 1].clone())
        coff_mul = coff_i * coff_j  # [num_edge, 3]
        coff_i_norm = coff_i.norm(dim=-1, keepdim=True) # [num_edge, 1]
        coff_j_norm = coff_j.norm(dim=-1, keepdim=True) # [num_edge, 1]
        pseudo_cos = coff_mul.sum(dim=-1, keepdim=True) / (coff_i_norm + EPSILON) / (coff_j_norm + EPSILON)
        pseudo_sin = torch.sqrt(1 - pseudo_cos ** 2)
        pseudo_angle = torch.cat([pseudo_sin, pseudo_cos], dim=-1)
        embed_i = self.get_embedding(coff_i)  # [num_edge, C]
        embed_j = self.get_embedding(coff_j)  # [num_edge, C]
        edge_embed = torch.cat([pseudo_angle, embed_i, embed_j], dim=-1)
        edge_attr_3D_frame_invariant = self.project(edge_embed)
        
        edge_attr = edge_attr_3D_invariant * edge_attr_2D + edge_attr_3D_frame_invariant
        
        # match dimension
        node_attr = self.node_emb(node_2D_repr)
        
        # estimate scores
        output = self.score_network(extended_edge_index, node_attr, edge_attr, equivariant_basis)
        output = output["gradient"]
        scores = -output

        _, std_pos = self.sde_pos.marGINal_prob(pos_perturbed, t_pos)
        scores = scores / std_pos[:, None]
        # print(t_pos, std_pos)
        return scores


class SDEModel2Dto3D_03(torch.nn.Module):
    def __init__(
        self, emb_dim, hidden_dim,
        beta_schedule, beta_min, beta_max, num_diffusion_timesteps, SDE_type="VE",
        short_cut=False, concat_hidden=False, use_extend_graph=False):

        super(SDEModel2Dto3D_03, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.SDE_type = SDE_type
        self.use_extend_graph = use_extend_graph

        self.node_emb = MultiLayerPerceptron(self.emb_dim, [self.hidden_dim], activation="silu")
        self.edge_2D_emb = nn.Sequential(nn.Linear(self.emb_dim*2, self.emb_dim), nn.BatchNorm1d(self.emb_dim), nn.ReLU(), nn.Linear(self.emb_dim, self.hidden_dim))
        self.edge_2D_emb = nn.Linear(self.emb_dim*2, self.hidden_dim)
        # TODO: will hack
        self.edge_emb = torch.nn.Embedding(100, self.hidden_dim)

        self.coff_gaussian_fourier = GaussianFourierProjection(embedding_size=self.hidden_dim, scale=1)
        self.coff_mlp = nn.Linear(4 * self.hidden_dim, self.hidden_dim)
        self.project = MultiLayerPerceptron(2 * self.hidden_dim + 2, [self.hidden_dim, self.hidden_dim], activation="silu")

        self.score_network = EquivariantScoreNetwork(hidden_dim=self.hidden_dim, hidden_coff_dim=128, activation="silu", short_cut=short_cut, concat_hidden=concat_hidden)

        if self.SDE_type in ["VE", "VE_test"]:
            self.sde_pos = VESDE(sigma_min=beta_min, sigma_max=beta_max, N=num_diffusion_timesteps)
        elif self.SDE_type in ["VP", "VP_test"]:
            self.sde_pos = VPSDE(beta_min=beta_min, beta_max=beta_max, N=num_diffusion_timesteps)
        elif self.SDE_type == "discrete_VE":
            betas = get_beta_schedule(
                beta_schedule=beta_schedule,
                beta_min=beta_min,
                beta_max=beta_max,
                num_diffusion_timesteps=num_diffusion_timesteps,
            )
            self.betas = nn.Parameter(betas, requires_grad=False)
            # variances
            alphas = (1. - betas).cumprod(dim=0)
            self.alphas = nn.Parameter(alphas, requires_grad=False)
            # print("betas used in 2D to 3D diffusion model", self.betas)
            # print("alphas used in 2D to 3D diffusion model", self.alphas)

        self.num_diffusion_timesteps = num_diffusion_timesteps
        return

    def get_embedding(self, coff_index):
        coff_embeds = []
        for i in [0, 2]:  # if i=1, then x=0
            coff_embeds.append(self.coff_gaussian_fourier(coff_index[:, i:i + 1]))  # [E, 2C]
        coff_embeds = torch.cat(coff_embeds, dim=-1)  # [E, 6C]
        coff_embeds = self.coff_mlp(coff_embeds)

        return coff_embeds

    def forward(self, node_2D_repr, data, anneal_power):
        pos = data.positions
        pos.requires_grad = True

        node2graph = data.batch
        if self.use_extend_graph:
            extended_edge_index = data.extended_edge_index
        else:
            extended_edge_index = data.edge_index   

        # Perterb pos
        pos_noise = torch.randn_like(pos)

        # sample variances
        time_step = torch.randint(0, self.num_diffusion_timesteps, size=(data.num_graphs // 2 + 1,), device=pos.device)
        time_step = torch.cat([time_step, self.num_diffusion_timesteps - time_step - 1], dim=0)[:data.num_graphs]  # (num_graph, )

        if self.SDE_type in ["VE", "VP"]:
            time_step = time_step / self.num_diffusion_timesteps * (1 - EPSILON) + EPSILON  # normalize to [0, 1]
            time_step = time_step.squeeze(-1)
            t_pos = time_step.index_select(0, node2graph)  # (num_graph, )
            mean_pos, std_pos = self.sde_pos.marGINal_prob(pos, t_pos)
            pos_perturbed = mean_pos + std_pos[:, None] * pos_noise
            
        elif self.SDE_type in ["VE_test", "VP_test"]:
            time_step = time_step.squeeze(-1)
            t_pos = time_step.index_select(0, node2graph)  # (num_graph, )
            mean_pos, std_pos = self.sde_pos.marGINal_prob(pos, t_pos)
            pos_perturbed = mean_pos + std_pos[:, None] * pos_noise

        elif self.SDE_type == "discrete_VE":
            a = self.alphas.index_select(0, time_step)  # (num_graph, )
            a_pos = a.index_select(0, node2graph).unsqueeze(-1)  # (num_nodes, 1)
            pos_perturbed = pos + pos_noise * (1.0 - a_pos).sqrt() / a_pos.sqrt()
        
        # edge_attr from 2D represenattion node_2D_repr
        row, col = extended_edge_index
        edge_attr_2D = torch.cat([node_2D_repr[row], node_2D_repr[col]], dim=-1)
        edge_attr_input = self.edge_emb(data.extended_edge_attr) # (num_edge, hidden)
        edge_attr_2D = self.edge_2D_emb(edge_attr_2D) + edge_attr_input
        
        # construct geometric features
        coord_diff, coord_cross, coord_vertical = coord2basis(pos_perturbed, row, col)  # [num_edge, 3] * 3
        equivariant_basis = [coord_diff, coord_cross, coord_vertical]
        edge_basis = torch.cat([coord_diff.unsqueeze(1), coord_cross.unsqueeze(1), coord_vertical.unsqueeze(1)], dim=1)  # [num_edge, 3, 3]
        r_i, r_j = pos_perturbed[row], pos_perturbed[col]  # [num_edge, 3]
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

        # match dimension
        node_attr = self.node_emb(node_2D_repr)

        # estimate scores
        output = self.score_network(extended_edge_index, node_attr, edge_attr, equivariant_basis)
        scores = output["gradient"]
        if anneal_power == 0:
            loss_pos = torch.sum((scores - pos_noise) ** 2, -1)  # (num_node)
        else:
            annealed_std = std_pos ** anneal_power  # (num_node)
            annealed_std = annealed_std.unsqueeze(1,)  # (num_node,1)
            loss_pos = torch.sum((scores - pos_noise) ** 2 * annealed_std, -1)  # (num_node)
        loss_pos = scatter_mean(loss_pos, node2graph)  # (num_graph)

        loss_dict = {
            'position': loss_pos.mean(),
        }
        return loss_dict

    @torch.no_grad()
    def get_score(self, node_2D_repr, data, pos_perturbed, sigma, t_pos):
        node_attr = self.node_emb(node_2D_repr)
        
        if self.use_extend_graph:
            extended_edge_index = data.extended_edge_index
        else:
            extended_edge_index = data.edge_index
        
        # edge_attr from 2D represenattion node_2D_repr
        row, col = extended_edge_index        
        edge_attr_2D = torch.cat([node_2D_repr[row], node_2D_repr[col]], dim=-1)
        edge_attr_input = self.edge_emb(data.extended_edge_attr) # (num_edge, hidden)   
        edge_attr_2D = self.edge_2D_emb(edge_attr_2D) + edge_attr_input
        
        # construct geometric features
        coord_diff, coord_cross, coord_vertical = coord2basis(pos_perturbed, row, col)  # [num_edge, 3] * 3
        equivariant_basis = [coord_diff, coord_cross, coord_vertical]
        edge_basis = torch.cat([coord_diff.unsqueeze(1), coord_cross.unsqueeze(1), coord_vertical.unsqueeze(1)], dim=1)  # [num_edge, 3, 3]
        r_i, r_j = pos_perturbed[row], pos_perturbed[col]  # [num_edge, 3]
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
        
        # match dimension
        node_attr = self.node_emb(node_2D_repr)
        
        # estimate scores
        output = self.score_network(extended_edge_index, node_attr, edge_attr, equivariant_basis)
        output = output["gradient"]
        scores = -output

        _, std_pos = self.sde_pos.marGINal_prob(pos_perturbed, t_pos)
        scores = scores / std_pos[:, None]
        # print(t_pos, std_pos)
        return scores


class SDEModel2Dto3D_04(torch.nn.Module):
    def __init__(
        self, emb_dim, hidden_dim,
        beta_schedule, beta_min, beta_max, num_diffusion_timesteps, SDE_type="VE",
        short_cut=False, concat_hidden=False, use_extend_graph=False):

        super(SDEModel2Dto3D_04, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.SDE_type = SDE_type
        self.use_extend_graph = use_extend_graph

        self.node_emb = MultiLayerPerceptron(self.emb_dim, [self.hidden_dim], activation="silu")
        self.edge_2D_emb = nn.Sequential(nn.Linear(self.emb_dim*2, self.emb_dim), nn.BatchNorm1d(self.emb_dim), nn.ReLU(), nn.Linear(self.emb_dim, self.hidden_dim))
        self.edge_2D_emb = nn.Linear(self.emb_dim*2, self.hidden_dim)
        # TODO: will hack
        self.edge_emb = torch.nn.Embedding(100, self.hidden_dim)

        self.coff_gaussian_fourier = GaussianFourierProjection(embedding_size=self.hidden_dim, scale=1)
        self.coff_mlp = nn.Linear(4 * self.hidden_dim, self.hidden_dim)
        self.project = MultiLayerPerceptron(2 * self.hidden_dim + 2, [self.hidden_dim, self.hidden_dim], activation="silu")

        self.score_network = EquivariantScoreNetwork(hidden_dim=self.hidden_dim, hidden_coff_dim=128, activation="silu", short_cut=short_cut, concat_hidden=concat_hidden)

        if self.SDE_type in ["VE", "VE_test"]:
            self.sde_pos = VESDE(sigma_min=beta_min, sigma_max=beta_max, N=num_diffusion_timesteps)
        elif self.SDE_type in ["VP", "VP_test"]:
            self.sde_pos = VPSDE(beta_min=beta_min, beta_max=beta_max, N=num_diffusion_timesteps)
        elif self.SDE_type == "discrete_VE":
            betas = get_beta_schedule(
                beta_schedule=beta_schedule,
                beta_min=beta_min,
                beta_max=beta_max,
                num_diffusion_timesteps=num_diffusion_timesteps,
            )
            self.betas = nn.Parameter(betas, requires_grad=False)
            # variances
            alphas = (1. - betas).cumprod(dim=0)
            self.alphas = nn.Parameter(alphas, requires_grad=False)
            # print("betas used in 2D to 3D diffusion model", self.betas)
            # print("alphas used in 2D to 3D diffusion model", self.alphas)

        self.num_diffusion_timesteps = num_diffusion_timesteps
        return

    def get_embedding(self, coff_index):
        coff_embeds = []
        for i in [0, 2]:  # if i=1, then x=0
            coff_embeds.append(self.coff_gaussian_fourier(coff_index[:, i:i + 1]))  # [E, 2C]
        coff_embeds = torch.cat(coff_embeds, dim=-1)  # [E, 6C]
        coff_embeds = self.coff_mlp(coff_embeds)

        return coff_embeds

    def forward(self, node_2D_repr, data, anneal_power):
        pos = data.positions
        pos.requires_grad = True

        node2graph = data.batch
        if self.use_extend_graph:
            extended_edge_index = data.extended_edge_index
        else:
            extended_edge_index = data.edge_index   

        # Perterb pos
        pos_noise = torch.randn_like(pos)

        # sample variances
        time_step = torch.randint(0, self.num_diffusion_timesteps, size=(data.num_graphs // 2 + 1,), device=pos.device)
        time_step = torch.cat([time_step, self.num_diffusion_timesteps - time_step - 1], dim=0)[:data.num_graphs]  # (num_graph, )

        if self.SDE_type in ["VE", "VP"]:
            time_step = time_step / self.num_diffusion_timesteps * (1 - EPSILON) + EPSILON  # normalize to [0, 1]
            time_step = time_step.squeeze(-1)
            t_pos = time_step.index_select(0, node2graph)  # (num_graph, )
            mean_pos, std_pos = self.sde_pos.marGINal_prob(pos, t_pos)
            pos_perturbed = mean_pos + std_pos[:, None] * pos_noise
            
        elif self.SDE_type in ["VE_test", "VP_test"]:
            time_step = time_step.squeeze(-1)
            t_pos = time_step.index_select(0, node2graph)  # (num_graph, )
            mean_pos, std_pos = self.sde_pos.marGINal_prob(pos, t_pos)
            pos_perturbed = mean_pos + std_pos[:, None] * pos_noise

        elif self.SDE_type == "discrete_VE":
            a = self.alphas.index_select(0, time_step)  # (num_graph, )
            a_pos = a.index_select(0, node2graph).unsqueeze(-1)  # (num_nodes, 1)
            pos_perturbed = pos + pos_noise * (1.0 - a_pos).sqrt() / a_pos.sqrt()
        
        # edge_attr from 2D represenattion node_2D_repr
        row, col = extended_edge_index
        edge_attr_2D = torch.cat([node_2D_repr[row] * node_2D_repr[col], node_2D_repr[row] + node_2D_repr[col]], dim=-1)
        edge_attr_input = self.edge_emb(data.extended_edge_attr) # (num_edge, hidden)
        edge_attr_2D = self.edge_2D_emb(edge_attr_2D) + edge_attr_input
        
        # construct geometric features
        coord_diff, coord_cross, coord_vertical = coord2basis(pos_perturbed, row, col)  # [num_edge, 3] * 3
        equivariant_basis = [coord_diff, coord_cross, coord_vertical]
        edge_basis = torch.cat([coord_diff.unsqueeze(1), coord_cross.unsqueeze(1), coord_vertical.unsqueeze(1)], dim=1)  # [num_edge, 3, 3]
        r_i, r_j = pos_perturbed[row], pos_perturbed[col]  # [num_edge, 3]
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

        # match dimension
        node_attr = self.node_emb(node_2D_repr)

        # estimate scores
        output = self.score_network(extended_edge_index, node_attr, edge_attr, equivariant_basis)
        scores = output["gradient"]
        if anneal_power == 0:
            loss_pos = torch.sum((scores - pos_noise) ** 2, -1)  # (num_node)
        else:
            annealed_std = std_pos ** anneal_power  # (num_node)
            annealed_std = annealed_std.unsqueeze(1,)  # (num_node,1)
            loss_pos = torch.sum((scores - pos_noise) ** 2 * annealed_std, -1)  # (num_node)
        loss_pos = scatter_mean(loss_pos, node2graph)  # (num_graph)

        loss_dict = {
            'position': loss_pos.mean(),
        }
        return loss_dict

    @torch.no_grad()
    def get_score(self, node_2D_repr, data, pos_perturbed, sigma, t_pos):
        node_attr = self.node_emb(node_2D_repr)
        
        if self.use_extend_graph:
            extended_edge_index = data.extended_edge_index
        else:
            extended_edge_index = data.edge_index
        
        # edge_attr from 2D represenattion node_2D_repr
        row, col = extended_edge_index        
        edge_attr_2D = torch.cat([node_2D_repr[row] * node_2D_repr[col], node_2D_repr[row] + node_2D_repr[col]], dim=-1)
        edge_attr_input = self.edge_emb(data.extended_edge_attr) # (num_edge, hidden)   
        edge_attr_2D = self.edge_2D_emb(edge_attr_2D) + edge_attr_input
        
        # construct geometric features
        coord_diff, coord_cross, coord_vertical = coord2basis(pos_perturbed, row, col)  # [num_edge, 3] * 3
        equivariant_basis = [coord_diff, coord_cross, coord_vertical]
        edge_basis = torch.cat([coord_diff.unsqueeze(1), coord_cross.unsqueeze(1), coord_vertical.unsqueeze(1)], dim=1)  # [num_edge, 3, 3]
        r_i, r_j = pos_perturbed[row], pos_perturbed[col]  # [num_edge, 3]
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
        
        # match dimension
        node_attr = self.node_emb(node_2D_repr)
        
        # estimate scores
        output = self.score_network(extended_edge_index, node_attr, edge_attr, equivariant_basis)
        output = output["gradient"]
        scores = -output

        _, std_pos = self.sde_pos.marGINal_prob(pos_perturbed, t_pos)
        scores = scores / std_pos[:, None]
        # print(t_pos, std_pos)
        return scores