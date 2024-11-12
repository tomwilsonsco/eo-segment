# segment eo imagery using ai
Currently implemented fastai process using fastai's `unet_learner()`.

Makes use of [rschip](https://github.com/tomwilsonsco/rs-chip) for image preprocessing and creating normaliser.

Completely reproducible simple example including 4 band (B,G,R,NiR) Sentinel 2 image and mask image to classify water bodies.

## Setup

1. Clone this repository.

2. `cd` to the repo and build the docker image:

```bash
docker build . --file .devcontainer/Dockerfile -t segment
```

3. Run the docker image ensuring access to gpu:

```bash
docker run --rm --gpus all -i -t -p 127.0.0.1:8888:8888 -w /app \
--mount type=bind,src="$(pwd)",target=/app segment
```

## Prepare imagery
 
```bash
python src/preprocess/tile.py
```
To generate image chips for use with `unet_learner`.


## Train a model
```bash
python src/fast_ai/train.py --image-path inputs/chips_img
```
Many more training options with defaults. Run `python src/fast_ai/train.py -h` to see them.

## Make predictions for full image extent:
```bash
python src/fast_ai/predict.py \
--input-image inputs/s2_flow_country_2023_06_16.tif \
--trained-model models/fastai_unet_31_10_2024_1119 \
--normaliser-scaler inputs/chips_img/s2_flow_country_2023_06_16_normaliser.pkl \
--boundary-remove
```


