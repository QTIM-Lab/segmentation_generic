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
    parser.add_argument('--label_bbox_option', type=str, default='label', help="Which bounding box type do you want ['label','image','padded','yolo']")

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
    for dir in os.listdir(all_models_root_dir):  # iterate over /output_logger/... folder
        root = os.path.join(all_models_root_dir, f"{dir}/wandb-lightning/{dir.split('_')[0]}/checkpoints")
        for file in os.listdir(root):
            # Check if the file ends with .pt or .ckpt
            if file.endswith('.pt') or file.endswith('.ckpt'):
                # import pdb; pdb.set_trace()
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
                    num_workers=4,
                    label_bbox_option = args.label_bbox_option
                )

                outputs[dir] = test_val
                print("outputs: ", outputs)

    print('output:')
    import pdb; pdb.set_trace()
    print(outputs)
