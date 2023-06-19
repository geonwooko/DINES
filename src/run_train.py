import sys
import json
import torch
from fire import Fire
from utils import print_params, fix_seed, create_directory
from data import load_and_setup_data
from models.dines import DINESEncoder
from models.train import Trainer
from models.decoder import PariwiseCorrelationDecoder


def main(
    # whether to load the configuration used in a pre-trained model	
    load_config = False,
    
    # dataset and preprocessing 
    dataset = "BC_ALPHA",
    data_dir = "./data",
    output_dir = "./output",
    test_ratio = 0.2,
    gpu_id = None,
    seed = None,
    
    # general hyperparamters
    in_dim = 64,
    out_dim = 64,
    num_epochs = 100,
    lr =  0.005,
    weight_decay = 0.005,
    
    # model hyperparameters
    num_factors = 8,
    num_layers = 2,
    lambda_disc = 0.1,
    aggr_type='sum'
):
    """
    Train DINES with hyperparameters

    Args:
        load_config: whether to load the configuration used in a pre-trained model	
        dataset: dataset name
        data_dir: data directory path
        output_dir: output directory path
        test_ratio: ratio of test edges
        gpu_id: GPU id; If None, a CPU is used
        seed: random seed; If None, the seed is not fixed
        in_dim: input feature dimension (d_in)
        out_dim: output embedding dimension (d_out)
        num_epochs: number of epochs
        lr: learning rate (eta)
        weight_decay: strength of L2 regularization (lambda_reg)
        num_factors: number of factors (K)
        num_layers: number of layers (L)
        lambda_disc: strength of the discriminative loss (lambda_disc)
        aggr_type: aggregator type ['sum', 'max', 'attn', 'mean']
    """
    
    # Build configuration
    if load_config:
        with open(f"./pretrained/{dataset}/config.json", 'r') as f:
            config = json.load(f)
    else:
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
        config['NUM_LAYERS'] = num_layers
        config['NUM_FACTORS'] = num_factors
        config['LAMBDA_DISC'] = lambda_disc
        config['AGGR_TYPE'] = aggr_type
    config['OUTPUT_DIR'] = output_dir
    config['DEVICE'] = "cpu" if gpu_id is None else f"cuda:{gpu_id}"
    
    print_params(config)
    
    # If seed is not None, fix the seed
    if config['SEED'] is not None:
        fix_seed(config['SEED'])
        # Note that `torch.index_add_`function cannot be run deterministically on the GPU.
        # For a stric reproducibility, we force this code to run deterministically if you use a CPU.
        # If you want this to run non-deterministically for speed on the CPU, remove the below.
        if  config['DEVICE'] == 'cpu':
            torch.use_deterministic_algorithms(True)
    
    # Load and preprocess dataset
    data = load_and_setup_data(config)

    # Build encoder and decoder
    encoder = DINESEncoder(
        in_dim=config['IN_DIM'],
        out_dim=config['OUT_DIM'],
        num_layers=config['NUM_LAYERS'],
        num_factors=config['NUM_FACTORS'],
        aggr_type=config['AGGR_TYPE']
    ).to(config['DEVICE'])
    
    decoder = PariwiseCorrelationDecoder(
        out_dim=config['OUT_DIM'],
        num_factors=config['NUM_FACTORS']
    ).to(config['DEVICE'])
    
    # Build trainer
    trainer = Trainer(encoder=encoder, decoder=decoder, config=config)

    # Train DINES
    trainer.train_model(data=data, num_epochs=config['NUM_EPOCHS'])
    
    # Save the models and config
    save_dir = f"{output_dir}/{dataset}"
    create_directory(save_dir)
    trainer.save_models(save_dir)
    with open(f"{save_dir}/config.json", 'w') as f:
        json.dump(config, f, indent=4)


if __name__ == "__main__":
    sys.exit(Fire(main))
