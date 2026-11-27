"""
Runs of the methods from the literature (PatchCore, PaDiM). Separate from run_experiment.py because the protocol
is different: no training, the statistics of the backbone features are fitted with a single pass over the training data.
Data and evaluation are the same as for the custom models (build_dataloaders, engine.evaluate).
"""

from models.literature_models import PatchCoreWrapper, PaDiMWrapper
from run_experiment import build_dataloaders
from utils.helpers import get_device, set_seed
from engine import evaluate
from pathlib import Path
import torch
import json
import gc


def build_literature_model(config: dict, device: torch.device, memory_bank_path=None):
    if config['model_name'] == 'PatchCore':
        model = PatchCoreWrapper(
            backbone=config['backbone_name'],
            layers=config['feature_layers'],
            coreset_sampling_ratio=config['coreset_sampling_ratio'],
            num_neighbors=config['num_neighbors'],
        ).to(device)
    elif config['model_name'] == 'PaDiM':
        model = PaDiMWrapper(
            backbone=config['backbone_name'],
            layers=config['feature_layers'],
            n_features=config['n_features'],
        ).to(device)
    else:
        raise ValueError(f"Unknown model name: {config['model_name']}")

    if memory_bank_path:
        # restores a fitted PatchCore memory bank without fitting again
        model.model.memory_bank = torch.load(memory_bank_path, map_location=device)
    return model


def fit_literature_model(class_name: str, config: dict):
    """
    Fits the model on the training set of the class and returns it with the test loader. Also used to get back a fitted
    PaDiM (not saved on disk).
    """
    set_seed(config['seed'])
    device = get_device(config['device'])

    train_loader, _, test_loader = build_dataloaders(class_name, config)
    # same images as train_loader, but without drop_last: the fit must see every training image
    fit_loader = torch.utils.data.DataLoader(
        train_loader.dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False,
    )

    model = build_literature_model(config, device)
    model.fit(fit_loader, device)
    return model, test_loader


def run_literature_single_class(class_name: str, config: dict):

    '''
    Entry point of the script. Orchestrates the instantiation of the model, fitting, evaluation, 
    and saving of results for a single class.
    '''

    print(f"\n{'='*30}  Starting class: {class_name.upper()}  {'='*30}")

    out_dir = Path(config['output_dir']) / config['experiment_name'] / class_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir.parent / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Fit the model on the training set 
    model, test_loader = fit_literature_model(class_name, config)
    if config['model_name'] == 'PatchCore':  # PaDiM is not saved: fit it again with fit_literature_model
        torch.save(model.model.memory_bank, out_dir / "memory_bank.pt")

    # Evaluate the model on the test set and save metrics
    metrics = evaluate(model, test_loader, get_device(config['device']), save_dir=out_dir)

    # Clean up to free memory
    del model, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics
