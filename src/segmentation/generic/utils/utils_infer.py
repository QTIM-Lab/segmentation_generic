import lightning.pytorch as pl
import yaml

from src.segmentation.generic.augmentations.augmentations_train_val import get_train_and_val_transform
import torch
from transformers import SamProcessor
import yaml
import torch
import lightning.pytorch as pl

from src.segmentation.generic.augmentations.augmentations_train_val import get_train_and_val_transform
from src.segmentation.generic.utils.utils_train import get_dataloader_from_csv
from src.segmentation.medsam.models.models_medsam import SegmentationMedSAM


def infer_model(config_augmentations_path, holdout_csv_path, weights_path, model_arch, image_root_dir, label_root_dir,
                gpu_id=0, num_workers=4, label_bbox_option='label', csv_img_path_col='image', csv_label_path_col='mask'):
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else "cpu")
    # # Need for output size etc.
    # config_augmentations = '/home/kindersc/repos/segmentation_generic/yamls/augmentations/medsam/medium_augs.yaml'
    # holdout_csv = '/sddata/data/geographic_atrophy/nj_110/csvs/holdout.csv'
    # weights_path = '/sddata/projects/segmentation_ga/models_official/medsam/ga_segmentation_medsam.pt'
    # Initialize Preprocessor
    preprocess = SamProcessor.from_pretrained("flaviagiammarino/medsam-vit-base")

    # Initialize Augmentations
    with open(config_augmentations_path, 'r') as file:
        augmentation_params = yaml.safe_load(file)
    _, val_transform = get_train_and_val_transform(augmentation_params)

    # Get dataset
    holdout_dataloader = get_dataloader_from_csv(
        model_arch=model_arch,
        csv_path=holdout_csv_path,
        csv_img_path_col=csv_img_path_col,
        csv_label_path_col=csv_label_path_col,
        image_root_dir=image_root_dir,
        label_root_dir=label_root_dir,
        transform=val_transform,
        preprocessor=preprocess,
        batch_size=1,
        num_workers=num_workers,
        label_bbox_option=label_bbox_option
    )

    configs = {
        'lr': 0,
        'optimizer_name': 'adam',
        'scheduler_name': 'none',
        'adamw_weight_decay': 0.0001,
        'sgd_momentum': 0.1
    }

    model = SegmentationMedSAM.load_from_checkpoint(
        checkpoint_path=weights_path,
        map_location=device,
        configs=dict(configs),
        num_classes=1,
        preprocessor=preprocess
    )

    trainer = pl.Trainer(
        # devices='auto',
        devices=[gpu_id],
        accelerator='gpu',
    )

    # Evaluate the model on the test set
    test_acc = trainer.test(model, dataloaders=holdout_dataloader)[0]['test_acc']

    return test_acc
