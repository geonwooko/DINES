import torch
import torch.nn.functional as F
import torch.nn as nn

class InitDisenLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_factors, act_fn):
        """
        Build a initial disentangling layer

        Args:
            in_dim: input features dimension (d_in)
            out_dim: output embedding dimension (d_0)
            num_factors: number of factors (K)
            act_fn: torch activation function
        """
        super(InitDisenLayer, self).__init__()
        self.d_in = in_dim
        self.d_0 = out_dim
        self.K = num_factors
        self.act_fn = act_fn
        self.setup_layers()


    def setup_layers(self):
        """
        Set up layers 
        """
        self.disen_weights = nn.Parameter(torch.randn(self.K, self.d_0, self.d_in//self.K))
        self.disen_bias = nn.Parameter(torch.zeros(1, self.K, self.d_in//self.K))
        torch.nn.init.xavier_uniform_(self.disen_weights)


    def forward(self, X):
        """
        Disentangle input features into K vectors
        
        The einsum function used in forward operates as follows:
            (Input Features (N, d_in) X FC Weights (K, d_in, d_0)) + FC biases (1, d_in, d_0)
                -> Initial disentangled embedding (N, K, d_0)
        It is equivalent to pass input features into K fully-connected layers
        
        Args:
            X: input node features

        Returns:
            f_0: Initial disentangled node embedding
        """
        f_0 = torch.einsum("ij,kjl->ikl", X, self.disen_weights) + self.disen_bias
        f_0 = F.normalize(self.act_fn(f_0))
        return f_0


class SDNConv(nn.Module):
    def __init__(self, in_dim, out_dim, num_factors, act_fn):
        """
        Build a SDNConv layer
        
        Args:
            in_dim: input embedding dimension (d_(l-1))
            out_dim: output embedding dimension (d_l)
            num_factors: number of factors (K)
            act_fn: torch activation function 
        """
        super(SDNConv, self).__init__()
        self.K = num_factors
        self.d_in = in_dim
        self.d_out = out_dim
        self.K = num_factors
        self.act_fn = act_fn
        self.setup_layers()


    def setup_layers(self):
        """
        Set up layers
        """
        self.disen_aggregate_weights = nn.Parameter(torch.randn(self.K, 5*self.d_in//self.K, self.d_out//self.K))
        self.disen_aggregate_bias = nn.Parameter(torch.zeros(1, self.K, self.d_in//self.K))
        torch.nn.init.xavier_uniform_(self.disen_aggregate_weights)


    def forward(self, Ep, En, f_in):
        """
        For each factor, aggregate the neighbors' embedding and combine the aggregated embeddings.
        
        The einsum function used in forward operates as follows:
            (Concatenation of messages of each factor (N, K, 5*d_in/K) X FC Weights (K, 5*d_in/K, d_out/K)) + FC biases (1, K, d_out/K)
                -> Initial disentangled embedding (N, K, d_out/K)

        Args:
            Ep: positive edge list
            En: negative edge list
            f_in: disentangled node embeddings of before layer

        Returns:
            f_out: aggregated disentangled node embeddings
        """
        m_p_out, m_p_in, m_n_out, m_n_in = self.aggregate(Ep, En, f_in)
        m_concat = torch.cat([m_p_out, m_p_in, f_in, m_n_out, m_n_in], dim=2)
        f_out = torch.einsum("ijk,jkl->ijl", m_concat, self.disen_aggregate_weights) + self.disen_aggregate_bias 
        f_out = F.normalize(self.act_fn(f_out))
        return f_out


    def aggregate(self, Ep, En, f_in):
        """
        Aggregate messsages for each factor by considering neighbor types based on sign and directions

        Args:
            Ep: positive edge list
            En: negative edge list
            f_in: disentangled node embeddings of before layer

        Returns:
            m_p_out: aggregated meesages of positive out-neighbors
            m_p_in: aggregated meesages of positive in-neighbors
            m_n_out: aggregated meesages of negative out-neighbors
            m_n_in: aggregated meesages of negative in-neighbors
        """
        Ep_src, Ep_dst = Ep[:, 0], Ep[:, 1]
        En_src, En_dst = En[:, 0], En[:, 1]
        zeros = torch.zeros_like(f_in)
        m_p_out = zeros.index_add(0, Ep_src, f_in[Ep_dst])
        m_p_in = zeros.index_add(0, Ep_dst, f_in[Ep_src])
        m_n_out = zeros.index_add(0, En_src, f_in[En_dst])
        m_n_in = zeros.index_add(0, En_dst, f_in[En_src])
        return m_p_out, m_p_in, m_n_out, m_n_in


class DINESEncoder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        num_factors,
        num_layers=2,
        act_fn=torch.tanh
    ):
        """
        Build a DINES Encoder

        Args:
            in_dim: input feature dimension (d_in)
            out_dim: output embedding dimension (d_out)
            num_factors: number of factors (K)
            num_layers: number of layers (L)
            act_fn: torch activation function
        """

        super(DINESEncoder, self).__init__()
        self.d_in = in_dim
        self.d_out = out_dim
        self.K = num_factors
        self.L = num_layers
        self.act_fn = act_fn

        self.setup_layers()


    def setup_layers(self):
        """
        Set up layers for DINES Encoder
        """
        self.init_disen = InitDisenLayer(self.d_in, self.d_out, self.K, self.act_fn)
        self.conv_layers = nn.ModuleList()
        for _ in range(self.L):
            self.conv_layers.append(SDNConv(self.d_out, self.d_out, self.K, self.act_fn))


    def forward(self, X, Ep, En):
        """
        Generate disentangled node representation
        Args:
            X: input node features
            Ep: positive edge list
            En: negative edge list

        Returns:
            Z: disentangled node embeddings
        """
        f_0 = self.init_disen(X)

        f_l = f_0
        for sdn_conv in self.conv_layers:
            f_l = sdn_conv(Ep, En, f_l)

        Z = f_l 
        return Z # (N, K, d_out/K)