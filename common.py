import os
import sys
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader
import numpy as np
import einops
import wandb
import plotly.express as px
from pathlib import Path
import itertools
import random
from IPython.display import display
import wandb
from jaxtyping import Float, Int, Bool, Shaped, jaxtyped
from typing import Union, Optional, Tuple, Callable, Dict
import typeguard
from functools import partial
import copy
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets
from IPython.display import HTML
import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
from tqdm.notebook import tqdm
from dataclasses import dataclass
from rich import print as rprint
import pandas as pd
import torch

import circuits.eval_sae_as_classifier as eval_sae
import circuits.analysis as analysis
import circuits.eval_board_reconstruction as eval_board_reconstruction
import circuits.get_eval_results as get_eval_results
import circuits.f1_analysis as f1_analysis
import circuits.utils as utils
import circuits.pipeline_config as pipeline_config
from circuits.dictionary_learning.dictionary import AutoEncoder, GatedAutoEncoder, AutoEncoderNew


from huggingface_hub import hf_hub_download
import chess_utils

import pickle
with open('/root/train_ChessGPT/data/lichess_hf_dataset/meta.pkl', 'rb') as picklefile:
    meta = pickle.load(picklefile)

device = torch.device("cuda:0")

autoencoder_group_path = "/root/chessgpt_git/chessgpt_git/SAE_BoardGameEval/autoencoders/testing_chess/"
autoencoder_path = "/root/chessgpt_git/chessgpt_git/SAE_BoardGameEval/autoencoders/testing_chess/trainer4/"


def get_dataset(device):
    train_dataset_name = f"chess_train_dataset.pkl"

    if os.path.exists(train_dataset_name):
        print("Loading statistics aggregation dataset")
        with open(train_dataset_name, "rb") as f:
            train_data = pickle.load(f)
    else:
        train_data = eval_sae.construct_dataset(
            False,
            [chess_utils.board_to_check_state],
            10000,
            split="train",
            device=device,
            precompute_dataset=True,
        )
        with open(train_dataset_name, "wb") as f:
            pickle.dump(train_data, f)
    return train_data

def load_autoencoder(device):
    return AutoEncoder.from_pretrained(Path(autoencoder_path + 'ae.pt'), device=device)

def load_model(device):
    model = AutoModelForCausalLM.from_pretrained("adamkarvonen/8LayerChessGPT2")
    model.to(device)
    return model

def get_indexing_function():
    indexing_functions = eval_sae.get_recommended_indexing_functions(False)
    return indexing_functions[0]


def get_aggregation_output_location(eval_sae_n_inputs):
    indexing_function = get_indexing_function()
    expected_aggregation_output_location = eval_sae.get_output_location(
        autoencoder_path,
        n_inputs=eval_sae_n_inputs,
        indexing_function=indexing_function,
    )
    return expected_aggregation_output_location

analysis_device = device

torch.cuda.empty_cache()
def get_expected_feature_labels_output_location(eval_sae_n_inputs):
    expected_aggregation_output_location = get_aggregation_output_location(eval_sae_n_inputs)
    expected_feature_labels_output_location = expected_aggregation_output_location.replace(
        "results.pkl", "feature_labels.pkl"
    )
    return expected_feature_labels_output_location

def get_feature_labels(eval_sae_n_inputs):
    expected_feature_labels_output_location = get_expected_feature_labels_output_location(eval_sae_n_inputs)
    with open(expected_feature_labels_output_location, "rb") as f:
        feature_labels = pickle.load(f)
    feature_labels = utils.to_device(feature_labels, analysis_device)
    return feature_labels

def rc_to_square_notation(row, col):
    letters = "ABCDEFGH"
    number = row + 1
    letter = letters[col]
    return f"{letter}{number}"


def get_aggregation_results(eval_sae_n_inputs):
    expected_aggregation_output_location = get_aggregation_output_location(eval_sae_n_inputs)
    with open(expected_aggregation_output_location, "rb") as f:
        aggregation_results = pickle.load(f)
    aggregation_results = utils.to_device(aggregation_results, device)
    return aggregation_results

def get_formatted_results(aggregation_results):
    custom_functions = pipeline_config.Config().chess_functions
    formatted_results = analysis.add_off_tracker(aggregation_results, custom_functions, analysis_device)

    formatted_results = analysis.normalize_tracker(
        formatted_results,
        "on",
        custom_functions,
        analysis_device,
    )

    formatted_results = analysis.normalize_tracker(
        formatted_results,
        "off",
        custom_functions,
        analysis_device,
    )
    return formatted_results

def get_feature_indices_in_alive_space(formatted_results, function_name):
    indices = (formatted_results[function_name]['on_normalized'][2] > 0.999).nonzero()[:, 0]
    return indices

def get_true_feature_indices(formatted_results, function_name):
    latent_indices = get_feature_indices_in_alive_space(formatted_results, function_name)
    return formatted_results['alive_features'][latent_indices]
