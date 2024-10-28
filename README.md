# Segmentation Generic


![Tests](https://github.com/QTIM-Lab/segmentation_generic/actions/workflows/tests.yaml/badge.svg)

By Scott Kinder (scott.kinder@cuanschutz.edu)

## Zero-to-run script


### Environment Setup

```sh
git clone https://github.com/QTIM-Lab/segmentation_generic.git

cd segmentation_generic

pip install virtualenv

# create venv
python3 -m virtualenv venv

# activate venv
source venv/bin/activate

pip install --upgrade pip

pip install -r requirements.txt

pip install -r requirements_dev.txt

# Allows you to see modules properly i.e. src.segmentation.etc
# Another way to think is: its like pip install <package>, but your package pip install segmentation_generic
# from segmentation_generic.src.segmentation
pip install -e .

```

### Data setup

Data should be in a folder, say /data/all/

Then, you need:
- /data/all/csvs, with train.csv, val.csv, test.csv. The csv's need columns: image, mask, which have the filename for the original image and binary segmentation mask
- /data/all/images, with images, any size any format should work. Match with the csv of course
- /data/all/labels, with binary labels, matching name on the label col of csv

### Run train

```sh
# Run train
python src/segmentation/generic/run/run_train.py \
    --model_arch medsam \
    --train_yaml /path/to/repo/segmentation_generic/yamls/training/sweeps/medsam/miccai_experiments/my_example.yaml \
    --system_yaml /path/to/repo/segmentation_generic/yamls/system/my_system.yaml \
    --gpu_id 0

```