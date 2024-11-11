# segment eo imagery using ai
Currently implemented fastai process using fastai's `unet_learner()`.

Makes use of [rschip](https://github.com/tomwilsonsco/rs-chip) for image preprocessing and creating normaliser.

Run 
```bash
python src/preprocess/tile.py
```
To generate image chips for use with `unet_learner`.

Then train a model:
```bash
python src/fast_ai/train.py --image-path inputs/chips_img
```
Many more training options with defaults. Run `python src/fast_ai/train.py -h` to see them.

And make predictions for full image extent:
```bash
python src/fast_ai/predict.py --input-image inputs/s2_flow_country_2023_06_16.tif --trained-model 
models/fastai_unet_31_10_2024_1119 --normaliser-scaler outputs/chips_img/s2_flow_country_2023_06_16_normaliser.pkl
```


