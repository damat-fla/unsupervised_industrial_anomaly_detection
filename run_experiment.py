from models.feature_autoencoder import FeatureReconstructionModel
from utils.helpers import set_seed, get_device
from models.pixel_autoencoder import Autoencoder
from torchvision.transforms import v2
from dataset import MVTec2DDataset
from engine import evaluate, train
from pathlib import Path
import torch


def run_experiment_single_class(class_name: str, config: dict):
    print(f"\n{'='*30}\n  Starting class: {class_name.upper()}\n{'='*30}")

    set_seed(config['seed'])
    DEVICE = get_device()

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
        class_name=class_name
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
    )

    val_dataset = MVTec2DDataset(
        transforms=transforms,
        split='validation',
        class_name=class_name,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
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

    # ================== MODEL ==================
    if config['model_name'] == 'FeatureReconstructionModel':
        model = FeatureReconstructionModel(
            backbone_name=config['backbone_name'], 
            feature_layers=config['feature_layers']
        ).to(DEVICE)
    elif config['model_name'] == 'Autoencoder':
        model = Autoencoder(
            in_channels=3, 
            base_channels=config['base_channels']
        ).to(DEVICE)
    else:
        raise ValueError(f"Unknown model name: {config['model_name']}")

    # ================== LOSS, OPTIMIZER ==================
    criterion = config['loss_fn']()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # ================== TRAINING ==================
    out_dir = Path(config['output_dir']) / config['experiment_name'] / class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training {config['model_name']} for class '{class_name}' on device: {DEVICE}")
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=config['num_epochs'],
        device=DEVICE,
        save_dir=out_dir,
    )


    # ================== EVALUATION ==================
    # Load the best model for evaluation
    best_model_path = out_dir / "best_model.pth"
    model.load_state_dict(torch.load(best_model_path))
    
    metrics = evaluate(model, test_loader, DEVICE)
 
        
    return history, metrics
