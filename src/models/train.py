import torch
from tqdm import tqdm
from loguru import logger
from itertools import chain

class Trainer:
    def __init__(self, encoder, decoder, config):
        """
        Build trainer for DINES encoder and decoder

        Args:
            encoder: DINES encoder
            decoder: DINES decoder 
            config: configuaritions
        """
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.lambda_disc = config['LAMBDA_DISC']
        self.optimizer = torch.optim.Adam(chain(self.encoder.parameters(), self.decoder.parameters()),
                                          lr=config['LR'],
                                          weight_decay=config['WEIGHT_DECAY'])
        
        
    def train_model(self, data, num_epochs=100):
        """
        Train DINES encoder and decoder

        Args:
            data: data dictionary
            num_epochs: number of training epochs 
        """
        logger.info("Start training...")
        pbar = tqdm(range(num_epochs), desc="Epoch...")
        self.encoder.train()
        self.decoder.train()
        for epoch in pbar:
            self.optimizer.zero_grad()
            Z = self.encoder.forward(data['X'], data['train_edges_each_type'])
            loss_bce = self.decoder.sign_prediction_loss(Z, data['train_edges'], data['train_y'])
            loss_disc = self.decoder.factor_discriminatve_loss(Z)
            loss_total = loss_bce + loss_disc * self.lambda_disc
            loss_total.backward()
            self.optimizer.step()
            pbar.set_description(f"Epoch : {epoch}  train_loss : {loss_total.item():.3f}")
        logger.info("Training is end!")
    
    
    def save_models(self, save_dir):
        """
        Save the trained encoder and decoder parameters as file

        Args:
            save_dir: save directory path
        """
        encoder_save_path = f"{save_dir}/encoder.pt"
        decoder_save_path = f"{save_dir}/decoder.pt"
        torch.save(self.encoder.cpu().state_dict(), encoder_save_path)
        torch.save(self.decoder.cpu().state_dict(), decoder_save_path)
        logger.info(f"Trained encoder saved in {encoder_save_path}")
        logger.info(f"Trained decoder saved in {decoder_save_path}")