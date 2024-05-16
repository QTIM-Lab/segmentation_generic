import pandas as pd
from torch.utils.data import DataLoader

from src.segmentation.mask2former.data_modules.datasets_train import ImageSegmentationDataset


def get_dataloaders(config, train_transform, val_transform, preprocess):
    data_root_dir = config.data_dir

    # get paths
    image_root_dir = data_root_dir + 'images'
    label_root_dir = data_root_dir + 'labels'
    train_csv = data_root_dir + 'csvs/train.csv'
    val_csv = data_root_dir + 'csvs/val.csv'
    test_csv = data_root_dir + 'csvs/test.csv'

    # get column names
    # TODO: fix hardcode?
    csv_img_path_col = 'image'
    csv_label_path_col = 'mask'

    # read csvs
    train_data_df = pd.read_csv(train_csv)
    val_data_df = pd.read_csv(val_csv)
    test_data_df = pd.read_csv(test_csv)

    # get list of image paths
    train_image_paths = train_data_df[csv_img_path_col].tolist()
    train_mask_paths = train_data_df[csv_label_path_col].tolist()
    val_image_paths = val_data_df[csv_img_path_col].tolist()
    val_mask_paths = val_data_df[csv_label_path_col].tolist()
    test_image_paths = test_data_df[csv_img_path_col].tolist()
    test_mask_paths = test_data_df[csv_label_path_col].tolist()

    # get datasets
    train_dataset = ImageSegmentationDataset(
        train_image_paths,
        train_mask_paths,
        transform=train_transform,
        preprocess=preprocess,
        image_root_dir=image_root_dir,
        label_root_dir=label_root_dir
    )
    val_dataset = ImageSegmentationDataset(
        val_image_paths,
        val_mask_paths,
        transform=val_transform,
        preprocess=preprocess,
        image_root_dir=image_root_dir,
        label_root_dir=label_root_dir
    )
    test_dataset = ImageSegmentationDataset(
        test_image_paths,
        test_mask_paths,
        transform=val_transform,
        preprocess=preprocess,
        image_root_dir=image_root_dir,
        label_root_dir=label_root_dir
    )

    # get dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=config.gpu_max_batch_size, shuffle=True, num_workers=config.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=config.gpu_max_batch_size, shuffle=False, num_workers=config.num_workers//2)
    test_dataloader = DataLoader(test_dataset, batch_size=config.gpu_max_batch_size, shuffle=False, num_workers=config.num_workers)

    return train_dataloader, val_dataloader, test_dataloader


def get_first_n_batches(input_dataloader, n=5):
    # Initialize an empty list to store the first 5 batches
    first_n_batches = []

    # Use enumerate to iterate over the DataLoader with an index
    for batch_idx, batch in enumerate(input_dataloader):
        if batch_idx < n:
            first_n_batches.append(batch)
        else:
            break  # Stop after getting the first n

    return first_n_batches
