import sys
import json
import torch
from fire import Fire
from loguru import logger
from utils import fix_seed, calculate_metrics
from data import load_and_setup_data
from models.dines import DINESEncoder
from models.decoder import PariwiseCorrelationDecoder

def evaluate(encoder, decoder, data):
    """
    Evaluate the trained DINES

    Args:
        encoder: DINES encoder
        decoder: DINES decoder
        data: data dictionary

    Returns:
        AUC, Macro-F1
    """
    with torch.no_grad():
        encoder.eval()
        decoder.eval()
        Z = encoder.forward(data['X'], data['train_edges_each_type'])
        test_prob = decoder.forward(Z, data['test_edges'])
        AUC, Macro_F1 = calculate_metrics(test_prob, data['test_y'])
    return AUC, Macro_F1 
   
   
def main(
    dataset = "BC_ALPHA",
    input_dir = "./output",
    gpu_id = None
):
    """
    Evaluate pre-trained DINES

    Args:
        dataset: dataset name
        input_dir: directory path where pre-trained DINES is stored
        gpu_id: GPU id. If None, use CPU
    """
    save_dir = f"{input_dir}/{dataset}"
    config_path = f"{save_dir}/config.json"
    encoder_pretrained_path = f'{save_dir}/encoder.pt'
    decoder_pretrained_path = f'{save_dir}/decoder.pt'
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    config['DEVICE'] = "cpu" if gpu_id is None else f"cuda:{gpu_id}"

    if config['SEED'] is not None:
        fix_seed(config['SEED'])
    
    # Load and preprocess dataset
    data = load_and_setup_data(config)

    # Load pre-trained DINES encoder
    encoder = DINESEncoder(
        in_dim=config['IN_DIM'],
        out_dim=config['OUT_DIM'],
        num_layers=config['NUM_LAYERS'],
        num_factors=config['NUM_FACTORS'],
        aggr_type=config['AGGR_TYPE']
    ).to(config['DEVICE'])
    encoder.load_state_dict(torch.load(encoder_pretrained_path, map_location=config['DEVICE']))
    
    # Load pre-trained DINES decoder
    decoder = PariwiseCorrelationDecoder(
        out_dim=config['OUT_DIM'],
        num_factors=config['NUM_FACTORS']
    ).to(config['DEVICE'])
    decoder.load_state_dict(torch.load(decoder_pretrained_path, map_location=config['DEVICE']))
    
    # Evaluate model
    logger.info(f"Evaluate DINES on {dataset}")
    AUC, Macro_F1 = evaluate(encoder, decoder, data)   
    logger.info(f"test AUC: {AUC:.3f}   Macro-F1: {Macro_F1:.3f}")
    
    
if __name__ == "__main__":
    sys.exit(Fire(main))
