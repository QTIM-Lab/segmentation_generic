import argparse
import os

from src.segmentation.generic.utils.utils_infer import infer_model

# output = {
#     '0345wdf': 0.984,
#     '123424d': 0.977,
#     ...
#     30
#     10 for each x 3 models per
# }



def parse_args():
    parser = argparse.ArgumentParser(description="Segmentation")
    parser.add_argument('--all_models_root_dir', type=str, required=True, help="Path to where all models are (i.e. /output_logger)")
    parser.add_argument('--config_augmentations_path', type=str, required=True, help="Path to some config file")
    parser.add_argument('--holdout_csv_path', type=str, required=True, help="Path to the test (or holdout) set csv")
    parser.add_argument('--model_arch', type=str, choices=['mask2former', 'medsam'], required=True, help="Model architecture")
    parser.add_argument('--image_root_dir', type=str, required=True, help="Directory where test images are stored")
    parser.add_argument('--label_root_dir', type=str, required=True, help="Directory where test labels are stored")
    parser.add_argument('--num_workers', type=int, default=1, help="Number of workers to use")
    parser.add_argument('--gpu_id', type=int, default=0, help="Which GPU to run on (distributed not available yet)")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    all_models_root_dir = args.all_models_root_dir
    config_augmentations_path = args.config_augmentations_path
    holdout_csv_path = args.holdout_csv_path
    model_arch = args.model_arch
    image_root_dir = args.image_root_dir
    label_root_dir = args.label_root_dir
    num_workers = args.num_workers
    gpu_id = args.gpu_id

    outputs = {}

    # Walk through the root directory
    for root, dirs, files in os.walk(all_models_root_dir):  # iterate over /output_logger/... folder
        folder_name = os.path.basename(root)
        for file in files:
            # Check if the file ends with .pt or .ckpt
            if file.endswith('.pt') or file.endswith('.ckpt'):
                # Print the full path to the file
                file_path = os.path.join(root, file)
                print(f"weights file: {file_path}")
                test_val = infer_model(
                    config_augmentations_path=config_augmentations_path,
                    holdout_csv_path=holdout_csv_path,
                    weights_path=file_path,
                    model_arch=model_arch,
                    image_root_dir=image_root_dir,
                    label_root_dir=label_root_dir,
                    gpu_id=0,
                    num_workers=4
                )

                outputs[folder_name] = test_val

    print('output:')
    print(outputs)
