import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
from fire import Fire
from utils import print_params, fix_seed, create_directory, calculate_metrics
from data import load_and_setup_data
from torchmetrics.functional import f1_score, auroc
from tqdm import tqdm
from loguru import logger
from itertools import chain
from torch_scatter import scatter_add
import warnings
warnings.filterwarnings(action='ignore')


class MFA(nn.Module): # multi-facted attention
    def __init__(self, num_factors, in_dim):
        super().__init__()
        self.K = num_factors
        self.d_in = in_dim
        self.d_K = self.d_in // self.K
        self.linear_embedding = nn.Linear(self.d_K, self.d_K, bias=False)
        torch.nn.init.xavier_uniform_(self.linear_embedding.weight)
        self.attention_embedding = nn.Linear(2*self.d_K, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.attention_embedding.weight)
        self.act = F.leaky_relu
        
    def forward(self, h, edges):
        h_lin = self.linear_embedding(h) # N,K,d_K
        
        attn = self.attention(h_lin, edges) # N,K,1 
        return attn
      
    def attention(self, h, edges):
        src_node, dst_node = edges[:, 0], edges[:, 1]
        device = h.device
        src_embedding, dst_embedding = h[src_node], h[dst_node]
        K_range = torch.arange(self.K).to(device) 
        src_k_index = torch.repeat_interleave(K_range, self.K) 
        dst_k_index = K_range.repeat(self.K)
        src_k_embedding = src_embedding[:, src_k_index, :]
        dst_k_embedding = dst_embedding[:, dst_k_index, :]
        score = torch.exp(self.act(
            self.attention_embedding(
                torch.concat([src_k_embedding, dst_k_embedding]
                                , dim=2)
                ))).reshape(-1, self.K, self.K).sum(dim=1)
        attn = score / score.sum(dim=1, keepdim=True)
        return attn.unsqueeze(2)
      

class MNA(nn.Module): # multi-order neighbor aggregatoon
    def __init__(self, num_factors, in_dim):
        super().__init__()
        self.K = num_factors
        self.d_in = in_dim
        self.d_K = self.d_in // self.K
        self.mfa = MFA(self.K, self.d_in)
        self.mix = nn.Linear(2*self.d_in, self.d_in, bias=False)
        torch.nn.init.xavier_uniform_(self.mix.weight)
        
    def forward(self, h_BU0, h_BU, balanced_edges, unbalanced_edges):
        attn_B = self.mfa(h_BU, balanced_edges)
        attn_U = self.mfa(h_BU, unbalanced_edges)

        out = h_BU0.new_zeros(h_BU0.shape)
        
        B_src, B_dst = balanced_edges[:, 0], balanced_edges[:, 1]
        h_B = h_BU + scatter_add(h_BU0[B_dst] * attn_B, B_src, out=out,dim=0)
        
        U_src, U_dst = unbalanced_edges[:, 0], unbalanced_edges[:, 1]
        h_U = h_BU + scatter_add(h_BU0[U_dst] * attn_U, U_src, out=out, dim=0)

        h_BU = self.mix(torch.concat([torch.flatten(h_B, start_dim=1), 
                                       torch.flatten(h_U, start_dim=1)],
                                      dim=1)).reshape(-1, self.K, self.d_K)
        h_BU = torch.tanh(h_BU)
        return h_BU

        
class MUSE(nn.Module):
    def __init__(self, num_factors, in_dim, out_dim, num_nodes):
        super().__init__()
        self.K = num_factors
        self.d_in = in_dim
        self.d_out = out_dim
        self.d_K = self.d_in // self.K
        self.num_nodes = num_nodes

        self.h0 = nn.Parameter(torch.empty((num_nodes , self.K, self.d_K)))
        torch.nn.init.xavier_uniform_(self.h0)
            
        self.mna1 = MNA(self.K, self.d_out)
        self.mna2 = MNA(self.K, self.d_out)
        
    def forward(self, h, balanced_edges, unbalanced_edges):
        h_BU0 = self.h0
        
        h_BU1 = self.mna1(h_BU0, h_BU0, balanced_edges[0], unbalanced_edges[0])
        
        h_BU2 = self.mna2(h_BU0, h_BU1, balanced_edges[1], unbalanced_edges[1])
        
        h = torch.flatten(h_BU2, start_dim=1)
        
        return h


class Decoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.classifier = nn.Linear(self.out_dim * 2, 1)
        self.bce = nn.BCEWithLogitsLoss()

        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def forward(self, node_embedding, edges):
        src, dst = edges[:, 0], edges[:, 1]
        src_embedding = node_embedding[src]
        dst_embedding = node_embedding[dst]
        merged_embedding = torch.hstack([src_embedding, dst_embedding])
        logits = self.classifier(merged_embedding)
        return logits

    def bce_loss(self, node_embedding, edges, y):
        logit = self.forward(node_embedding, edges).squeeze(1)
        loss = self.bce(logit, y.float())

        return loss

    def evaluate(self, node_embedding, edges, y):
        logit = self.forward(node_embedding, edges)
        y=y.long()
        prob = torch.sigmoid(logit)
        probs = torch.cat([1-prob, prob], dim=1)
        predictions = torch.argmax(probs, dim=1)
        metrics = DotMap()
        metrics.AUC = auroc(probs, y, num_classes=2).cpu().detach().item()
        metrics.F1_MACRO = (
            f1_score(predictions, y, average="macro", num_classes=2)
            .cpu()
            .detach()
            .item()
        )
        metrics.F1_MICRO = (
            f1_score(predictions, y, average="micro", num_classes=2)
            .cpu()
            .detach()
            .item()
        )
        binary = (
            f1_score(predictions, y, average="None", num_classes=2)
            .cpu()
            .detach()
            .tolist()
        )
        metrics.F1_BINARY = binary[1]
        metrics.MEAN_AUC_MACRO = (metrics.AUC + metrics.F1_MACRO) / 2

        return metrics

    def structure_loss(self, node_embedding, edges, y):
        pos_edges = edges[y==1]
        pos_src, pos_dst = pos_edges[:, 0], pos_edges[:, 1]
        src_embedding = node_embedding[pos_src]
        dst_embedding = node_embedding[pos_dst]
        L_STP = (torch.norm(src_embedding - dst_embedding, p=2, dim=1)).mean()
        
        neg_edges = edges[y!=1]
        neg_src, neg_dst = neg_edges[:, 0], neg_edges[:, 1]
        src_embedding = node_embedding[neg_src]
        dst_embedding = node_embedding[neg_dst]
        L_STN = (torch.norm(src_embedding - dst_embedding, p=2,dim=1)).mean()
        L_ST = L_STP - L_STN
        
        return L_ST

def extract_second_order_edges(train_adj, device):
    logger.info('Extract second order edges')
    A_squared = train_adj ** 2
    
    # Remove one-hop neighbors
    row, col = train_adj.nonzero()
    for r,c in zip(row, col):
        A_squared[r,c] = 0
    
    second_order_edges = np.vstack(A_squared.nonzero()).T
    second_order_weight = A_squared.data[A_squared.data != 0]
    second_order_balanced_edges = second_order_edges[second_order_weight > 0]
    second_order_unbalanced_edges = second_order_edges[second_order_weight < 0]
    
    second_order_balanced_edges = torch.tensor(second_order_balanced_edges, device=device, dtype=torch.long)
    second_order_unbalanced_edges = torch.tensor(second_order_unbalanced_edges, device=device, dtype=torch.long)
    logger.info('Second order neighbors are extracted!')
    return second_order_balanced_edges, second_order_unbalanced_edges
    

class Trainer:
    def __init__(self, encoder, decoder, config):
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.lambda_st = config['LAMBDA_ST']
        self.optimizer = torch.optim.Adam(chain(self.encoder.parameters(), self.decoder.parameters()),
                                          lr=config['LR'],
                                          weight_decay=config['WEIGHT_DECAY'])
        
        
    def train_model(self, data, num_epochs=100):
        logger.info("Start training...")
        pbar = tqdm(range(num_epochs), desc="Epoch...")
        self.encoder.train()
        self.decoder.train()
        for epoch in pbar:
            self.optimizer.zero_grad()
            Z = self.encoder.forward(data['X'], data['balanced_edges'], data['unbalanced_edges'])
            loss_bce = self.decoder.bce_loss(Z, data['train_edges'], data['train_y'])
            loss_st = self.decoder.structure_loss(Z, data['train_edges'], data['train_y'])
            
            loss_total = loss_bce + loss_st * self.lambda_st
            loss_total.backward()
            self.optimizer.step()
            pbar.set_description(f"Epoch : {epoch}  train_loss : {loss_total.item():.3f}")
        logger.info("Training is end!")
    

def evaluate(encoder, decoder, data):
    with torch.no_grad():
        encoder.eval()
        decoder.eval()
        Z = encoder.forward(data['X'], data['balanced_edges'], data['unbalanced_edges'])
        test_prob = decoder.forward(Z, data['test_edges'])
        AUC, Macro_F1 = calculate_metrics(test_prob, data['test_y'])
    return AUC, Macro_F1 
   
 
if __name__ == '__main__':
    def main(
        
        # dataset and preprocessing 
        dataset = "BC_ALPHA",
        data_dir = "./data",
        test_ratio = 0.2,
        gpu_id = None,
        seed = None,
        
        # general hyperparamters
        in_dim = 64,
        out_dim = 64,
        num_epochs = 100,
        lr =  0.001,
        weight_decay = 0,
        
        # model hyperparameters
        num_factors = 4,
        lambda_st = 0.01,
    ):
        # Build configuration
        config = dict()
        config['DATASET'] = dataset
        config['DATA_DIR'] = data_dir
        config['SEED'] = seed
        config['TEST_RATIO'] = test_ratio
        config['IN_DIM'] = in_dim
        config['OUT_DIM'] = out_dim
        config['NUM_EPOCHS'] = num_epochs
        config['LR'] = lr
        config['WEIGHT_DECAY'] = weight_decay
        config['NUM_FACTORS'] = num_factors
        config['LAMBDA_ST'] = lambda_st
        config['DEVICE'] = "cpu" if gpu_id is None else f"cuda:{gpu_id}"
        
        print_params(config)
        
        # Load and preprocess dataset
        data = load_and_setup_data(config)
        
        # Calculate balanced and unbalanced edges based on balance theory
        second_order_balanced_edges, second_order_unbalanced_edges = extract_second_order_edges(data['train_adj'], data['X'].device)
        data['balanced_edges'] = [data['train_pos_edges'], second_order_balanced_edges]
        data['unbalanced_edges'] = [data['train_neg_edges'], second_order_unbalanced_edges]
        
        # Build encoder and decoder
        encoder = MUSE(
            in_dim=config['IN_DIM'],
            out_dim=config['OUT_DIM'],
            num_factors=config['NUM_FACTORS'],
            num_nodes=data['num_nodes']
        ).to(config['DEVICE'])
        
        decoder = Decoder(
            out_dim=config['OUT_DIM']
        ).to(config['DEVICE'])
        
        # Build trainer
        trainer = Trainer(encoder=encoder, decoder=decoder, config=config)

        # Train MUSE
        trainer.train_model(data=data, num_epochs=config['NUM_EPOCHS'])
        
        # Evaluate
        AUC, Macro_F1 = evaluate(encoder, decoder, data)
        
        logger.info(f"test AUC: {AUC:.3f}   Macro-F1: {Macro_F1:.3f}")


if __name__ == "__main__":
    sys.exit(Fire(main))
