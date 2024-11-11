import rasterio as rio
from rasterio.windows import Window
import numpy as np
from pathlib import Path
import argparse
from fastai.vision.all import load_learner
from rschip import ImageChip
import pickle
from train import open_tif_rasterio, get_x, get_y   # need this part of the data load for prediction


def write_prediction(image_path, output_path, pred_arr):
    with rio.open(image_path) as f:
        prof = f.profile
    prof.update(count=1, dtype=np.int8)
    with rio.open(output_path, "w", **prof) as f:
       f.write(pred_arr)

def _generate_windows(src, tile_size, offset):
    for y in range(0, src.height, offset):
        for x in range(0, src.width, offset):
            window = Window(x, y, tile_size, tile_size)
            yield window

def apply_normaliser_scaler(img_arr, normaliser_scaler):
    if "normaliser" in normaliser_scaler:
        return ImageChip.apply_normaliser(img_arr, normaliser_scaler["normaliser"])
    else:
        return ImageChip.apply_scaler(img_arr, normaliser_scaler["scaler"])

def centre_only_prediction(pred_arr, win, src):
    tile_size = pred_arr.shape[0]
    mask = np.zeros((tile_size, tile_size), dtype=np.int8)
    # Default central region
    start_x, start_y = (tile_size // 8, tile_size // 8)
    end_x, end_y = (tile_size - start_x, tile_size - start_y)

    # Adjust the mask for image edge
    if win.col_off == 0:  # Left edge
        start_x = 0
    if win.row_off == 0:  # Top edge
        start_y = 0
    if win.col_off + win.width >= src.width:  # Right edge
        end_x = win.width
    if win.row_off + win.height >= src.height:  # Bottom edge
        end_y = win.height

    # Create the mask with the defined region
    mask[start_y:end_y, start_x:end_x] = 1

    # Apply the mask to the prediction array
    return pred_arr * mask


def read_predict_window(image_path, window, model, normaliser_scaler):
    with rio.open(image_path) as f:
        img_arr = f.read(window=window, boundless=True, fill_value=0)
    if normaliser_scaler:
        img_arr = apply_normaliser_scaler(img_arr, normaliser_scaler)
    pred = model.predict(img_arr)[0].numpy()   # predict method returns (label, _, probabilities)
    return np.squeeze(pred)

def win_ranges(win):
    ymin, ymax = (win.row_off, win.row_off + win.height)
    xmin, xmax = (win.col_off, win.col_off + win.width)
    return ymin, ymax, xmin, xmax

def init_pred_array(pred_full, win, tile_size):
    ymin, ymax, xmin, xmax = win_ranges(win)
    init_arr = np.zeros((tile_size, tile_size))
    value_ext = pred_full[0, ymin:ymax, xmin:xmax]
    value_ht, value_wd = value_ext.shape
    init_arr[0:value_ht, 0:value_wd] = value_ext
    return init_arr, value_ht, value_wd

def predict_image(image_path, output_path, model, normaliser_scaler, tile_size, offset):
    with (rio.open(image_path) as src):
        windows = list(_generate_windows(src, tile_size, offset))
        pred_full = np.zeros((1, src.height, src.width))
        for i, win in enumerate(windows):
            init, val_ht, val_wd = init_pred_array(pred_full, win, tile_size)
            pred = read_predict_window(image_path, win, model, normaliser_scaler)
            pred = pred.astype(np.uint8)
            pred = centre_only_prediction(pred, win, src)
            pred = np.maximum(pred, init)
            pred = pred[0:val_ht, 0:val_wd]
            ymin, ymax, xmin, xmax = win_ranges(win)
            pred_full[0, ymin:ymax, xmin:xmax] = pred
            if i != 0 and i % 100 ==0:
                print(f"predicted {i} of {len(windows)}")
    write_prediction(image_path, output_path, pred_full)
    print(f"Created {output_path}")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Predict a full image using trained fastai unet model"
    )

    parser.add_argument(
        "--input-image",
        type=str,
        required=True,
        help="Full path to the image for prediction.",
    )
    parser.add_argument(
        "--trained-model",
        type=str,
        required=True,
        help="Full path to the trained fastai model file.",
    )
    parser.add_argument(
        "--normaliser-scaler",
        type=str,
        required=False,
        help="Full path to the rschip.ImageChip.normaliser or standard_scaler pickle file.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=128,
        help="Size of the tiles to split the image into.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=64,
        help="Offset for sliding windows.",
    )


    args = parser.parse_args()

    # Convert paths to Pathlib objects
    image_path = Path(args.input_image)
    model_path = Path(args.trained_model)
    output_path = Path("outputs/", f"{image_path.stem}_{model_path.stem}_pred.tif")

    if args.normaliser_scaler:
        normaliser_scaler_fp = Path(args.normaliser_scaler)
        if not normaliser_scaler_fp.exists():
            raise FileNotFoundError("Normaliser or scaler path does not exist.")
        with open(normaliser_scaler_fp, "rb") as f:
            n_s = pickle.load(f)
            if "normaliser" in normaliser_scaler_fp.name:
                normaliser_scaler = {"normaliser": n_s}
            elif "scaler" in normaliser_scaler_fp.name:
                normaliser_scaler = {"scaler": n_s}
            else:
                raise ValueError(f"{normaliser_scaler_fp.name} not valid - must have normaliser or scaler in pickle file "
                                 f"name.")
    else:
        normaliser_scaler = None

    # Verify that paths exist
    if not image_path.exists():
        raise FileNotFoundError("Image path does not exist.")

    if not model_path.exists():
        raise FileNotFoundError("Model path does not exist.")

    if output_path.exists():
        try:
            Path.unlink(output_path)
        except:
            raise FileExistsError("Prediction image already exists. Cannot overwrite.")

    model = load_learner(model_path, cpu=False)

    predict_image(image_path, output_path, model, normaliser_scaler, args.tile_size, args.offset)

if __name__ == "__main__":
    main()
