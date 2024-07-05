import numpy as np
import pickle

def save_model(model, save_path):
    """
    Save a trained model to a file using pickle.

    Parameters:
    - model: Trained model object
    - save_path: File path to save the model
    """
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)


def load_model(load_path):
    """
    Load a trained model from a file using pickle.

    Parameters:
    - load_path: File path of the saved model

    Returns:
    - model: Trained model object
    """
    with open(load_path, 'rb') as f:
        model = pickle.load(f)
    return model