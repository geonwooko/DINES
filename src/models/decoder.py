import torch
import torch.nn as nn
import torch.nn.functional as F

class PariwiseCorrelationDecoder(nn.Module):
    def __init__(self, num_factors, out_dim):
        """
        Build a pairwise factor correlation decoder

        Args:
            num_fators: number of factors
            factor_dim: dimension of a factor embedding
        """
        super(PariwiseCorrelationDecoder, self).__init__()

        self.K = num_factors
        self.d_out = out_dim
        
        # Classifier for link sign prediction
        self.sign_predictor = nn.Sequential(
            nn.Linear(self.K**2, 1, bias=False),
            nn.Sigmoid(),
        )
        torch.nn.init.xavier_uniform_(self.sign_predictor[0].weight)


        # Factor classifier for self-supervised learning
        self.factor_discriminator = nn.Sequential(
            nn.Linear(self.d_out//self.K, self.K),
            nn.Softmax(dim=1),
        )
        torch.nn.init.xavier_uniform_(self.factor_discriminator[0].weight)
        torch.nn.init.zeros_(self.factor_discriminator[0].bias)


    def forward(self, Z, edges):
        """
        Generate factor-correlation map of edges and forward them into predicted probability

        Args:
            Z: disentangled node embeddings
            edges: edge list

        Returns:
            prob: predicted probability of edges ( if prob[i] >= 0.5, then predicted label of node i is 1(positive) )
        """
        M = len(edges)
        src, dst = edges[:, 0], edges[:, 1]
        src_emb = Z[src] # (M, K, D/K)
        dst_emb = torch.transpose(Z[dst], 1, 2) # (M, D/K, K)
        H = torch.bmm(src_emb, dst_emb)  # (M, K, K)
        prob = self.sign_predictor(H.reshape(M, -1))
        return prob


    def sign_prediction_loss(self, Z, edges, y):
        """
        Calculate link sign prediction loss

        Args:
            Z: disentangled node embeddings
            edges: edge list
            y: label list

        Returns:
            loss_bce: binary cross entropy loss
        """
        prob = self.forward(Z, edges)
        loss_bce = F.binary_cross_entropy(prob.squeeze(), y.float())
        return loss_bce


    def factor_discriminatve_loss(self, Z): 
        """
        Calculate self-supervised factor discriminative loss for enhancing disentanglement

        Args:
            Z: disentangled node embeddings

        Returns:
            loss_disc: cross entropy loss
        """
        pseudo_labels = torch.arange(start=0, end=self.K, step=1,
                                     device=Z.device, dtype=torch.long).repeat(len(Z))
        Z = Z.reshape(-1, self.d_out//self.K) # (N*K, d_out/K)
        probs = self.factor_discriminator(Z)
        loss_disc = F.nll_loss(torch.log(probs), pseudo_labels)
        return loss_disc