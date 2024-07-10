import numpy as np
import pickle
import math

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


# Calculate the best number of sentences for the summary
def best_len_of_summary(sentences: list[str]) -> int:
    # if the number of sentences is less than or equal to 3 then return the number of sentences
    if len(sentences) <= 3:
        return len(sentences)
    # else return 1.3 times the log of the number of sentences
    return round(1.3 * math.log(len(sentences)))