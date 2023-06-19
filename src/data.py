import torch
import numpy as np
from loguru import logger
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import coo_matrix


def load_and_setup_data(config):
    """
    Load and set up the data 

    Args:
        config: configuaritions

    Returns:
        data: dataset dictionary
    """
    logger.info("Start loading the signed network...")
    edge_path = f"{config['DATA_DIR']}/{config['DATASET']}/edges.csv"
    num_nodes, edges = load_edges(edge_path=edge_path)
    train_edges, test_edges, train_y, test_y = split_edges(edges, test_ratio=config['TEST_RATIO'])
    train_pos_edges = train_edges[train_y == 1]
    train_neg_edges = train_edges[train_y == 0]
    
    logger.info("Extract input features by TSVD...")
    train_adj = gen_adjacency_matrix(train_edges, train_y, num_nodes)
    X = extract_input_features(train_adj, config['IN_DIM'])
    
    data = dict()
    device = config['DEVICE']
    data['train_adj'] = train_adj
    data['train_edges'] = torch.from_numpy(train_edges).long().to(device)
    data['test_edges'] = torch.from_numpy(test_edges).long().to(device)
    data['train_pos_edges'] = torch.from_numpy(train_pos_edges).long().to(device)
    data['train_neg_edges'] = torch.from_numpy(train_neg_edges).long().to(device)
    data['train_y'] = torch.from_numpy(train_y).to(device).long().to(device)
    data['test_y'] = torch.from_numpy(test_y).to(device).long().to(device)
    data['X'] = torch.from_numpy(X).to(device).float().to(device)
    data['num_nodes'] = num_nodes
    
    data['train_edges_each_type'] = [
        data['train_pos_edges'], # +, ->
        data['train_neg_edges'], # -, ->
        data['train_pos_edges'][:, [1,0]], # +, <-
        data['train_neg_edges'][:, [1,0]]  # -, <-
    ]
    
    return data


def gen_adjacency_matrix(edges, y, num_nodes):
    """
    Generate adjacency matrix
    
    Args:
        edges: edge list
        num_nodes: number of nodes

    Returns:
        adj: adjacency matrix
    """
    y = np.where(y>0, 1, -1)
    adj = coo_matrix((y, (edges[:, 0], edges[:, 1])),
                     shape=(num_nodes, num_nodes))
                    #  dtype=np.float32)
    
    return adj


def extract_input_features(train_adj, in_dim):
    """
    Generated node features by tsvd

    Args:
        train_adj: adjacency matrix of train edges
        in_dim: TSVD input feature dimension

    Returns:
        X: input node features
    """
    X = TruncatedSVD(n_components=in_dim, n_iter=30).fit_transform(train_adj)
    X = StandardScaler().fit_transform(X)
    return X


def load_edges(edge_path):
    """
    Load the edges

    Args:
        edge_path: edge path

    Returns:
        num_nodes: number of nodes
        edges: raw edge list
    """
    edges = np.loadtxt(edge_path, dtype="int", delimiter=",")
    num_nodes = np.amax(edges[:, :2]) + 1
    return num_nodes, edges


def split_edges(edges, test_ratio):
    """
    Split the edges

    Args:
        edges: raw edge list
        test_ratio: edge ratio for test

    Returns:
        num_nodes: number of nodes
        train_edges: train edge list
        test_edges: test edge list
        train_y: train label list
        test_y: test label list
    """
    train_edges, test_edges = train_test_split(edges, test_size=test_ratio)
    train_y = np.where(train_edges[:, 2] > 0, 1, 0)
    test_y = np.where(test_edges[:, 2] > 0, 1, 0)
    train_edges = train_edges[:, :2]
    test_edges = test_edges[:, :2]
    return train_edges, test_edges, train_y, test_y