import json
import os
import torch


# default directory for outputting values
OUT_DIR = 'out/experiments/'
MODEL_DIR = 'out/models/'


def write_as_json(filename, data):
    """
    Dumps the given data into the specified file using JSON formatting. The file
    is created if it does not exist.

    Args:
        filename: path to file to dump JSON into
        data: numeric data to dump
    """
    if not os.path.exists(os.path.dirname(OUT_DIR + filename)):
        print('creating ...')
        os.makedirs(os.path.dirname(OUT_DIR + filename))

    with open(OUT_DIR + filename, "w") as f:
        json.dump(data, f)


def save_model(model, filename):
    if not os.path.exists(os.path.dirname(MODEL_DIR)):
        print('creating ...')
        os.makedirs(os.path.dirname(MODEL_DIR))

    torch.save(model, MODEL_DIR + filename)


def load_model(filename):
    if not os.path.exists(os.path.dirname(MODEL_DIR)):
        raise FileNotFoundError()
    return torch.load(MODEL_DIR + filename)
