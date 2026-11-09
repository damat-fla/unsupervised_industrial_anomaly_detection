from models.feature_autoencoder import FeatureReconstructionModel
from models.pixel_autoencoder import Autoencoder
from torchvision.transforms import v2
from utils.helpers import get_device, set_seed
from dataset import MVTec2DDataset
from engine import evaluate, train
from pathlib import Path
import torch
import json
import gc


def build_dataloaders(class_name: str, config: dict):
    # ================== TRANSFORMS ==================
    
    # define transforms for the dataset
    transforms = v2.Compose([
        v2.Resize(config['img_size']),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),  # valori in [0, 1]
    ])

    # define transforms for the target (anomaly binary segmentation) images
    target_transforms = v2.Compose([
        v2.Resize(config['img_size'], interpolation=v2.InterpolationMode.NEAREST),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),   
    ])

    # ================== DATASETS AND DATALOADERS ==================

    # create datasets and dataloaders for training, validation, and testing
    train_dataset = MVTec2DDataset(
        transforms=transforms,
        split='train',
        class_name=class_name,
        cache=config.get('cache_data', False)  # Use caching if specified in config
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True 
    )

    val_dataset = MVTec2DDataset(
        transforms=transforms,
        split='validation',
        class_name=class_name,
        cache=config.get('cache_data', False)  # Use caching if specified in config
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=False
    )

    test_dataset = MVTec2DDataset(
        transforms=transforms,
        target_transforms=target_transforms,
        split='test',
        class_name=class_name,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def build_model(config: dict, device: torch.device, checkpoint_path=None):
    if config['model_name'] == 'FeatureReconstructionModel':
        model = FeatureReconstructionModel(
            backbone_name=config['backbone_name'], 
            feature_layers=config['feature_layers']
        ).to(device)
    elif config['model_name'] == 'Autoencoder':
        model = Autoencoder(
            in_channels=3, 
            base_channels=config['base_channels'],
            latent_dim=config.get('latent_dim', None),
            block_type=config.get('block_type', 'conv'),  # Default to 'conv' if not specified
            img_size=config['img_size'][0],
        ).to(device)
    else:
        raise ValueError(f"Unknown model name: {config['model_name']}")

    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded model weights from {checkpoint_path}")
    return model

def run_experiment_single_class(class_name: str, config: dict):
    print(f"\n{'='*30}  Starting class: {class_name.upper()}  {'='*30}")

    set_seed(config['seed'])
    DEVICE = get_device(config['device'])

    # ================== DATA AND DATALOADERS ==================
    train_loader, val_loader, test_loader = build_dataloaders(class_name, config)
    

    # ================== MODEL ==================
    model = build_model(config, DEVICE)

    # ================== LOSS, OPTIMIZER ==================
    criterion = config['loss_fn']()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # ================== TRAINING ==================
    out_dir = Path(config['output_dir']) / config['experiment_name'] / class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    config_file = Path(config['output_dir']) / config['experiment_name'] / "config.json"
    if not config_file.exists():
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2, default=lambda o: getattr(o, "__name__", str(o)))


    print(f"Training {config['model_name']} for class '{class_name}' on device: {DEVICE}")
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=config['num_epochs'],
        patience=config['patience'],
        device=DEVICE,
        save_dir=out_dir,
    )


    # ================== EVALUATION ==================
    # Load the best model for evaluation
    best_model_path = out_dir / "checkpoints" / "best_model.pth"
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    
    metrics = evaluate(model, test_loader, DEVICE, save_dir=out_dir)

    del model, train_loader, val_loader, test_loader, optimizer, criterion
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return history, metrics
