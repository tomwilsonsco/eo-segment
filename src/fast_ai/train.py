import argparse
from datetime import datetime
import rasterio as rio
import numpy as np
from fastai.vision.all import *
from pathlib import Path


def open_tif_rasterio(fn):
    """
    Open a TIF file using rasterio and convert it to float32.
    """
    with rio.open(fn) as src:
        img = src.read()  # Reads as (channels, height, width)
    img = img.astype(np.float32)  # Ensure the datatype is float32
    return img


def get_x(file_path):
    return file_path


def get_y(file_path):
    return file_path.parent.parent / "chips_seg" / file_path.name


def prepare_data_loaders(image_path, batch_size):
    """
    Prepare data loaders for training and validation.
    """
    tif_block = TransformBlock(type_tfms=open_tif_rasterio)

    dblock = DataBlock(
        blocks=(
            tif_block,
            MaskBlock(codes=None),
        ),  # Use MaskBlock for multi-class segmentation if needed
        get_items=get_image_files,
        get_x=get_x,
        get_y=get_y,
        splitter=RandomSplitter(valid_pct=0.2),
        item_tfms=None,  # No scaling or resizing by default
        batch_tfms=None,  # Optionally add augmentations here
    )

    return dblock.dataloaders(image_path, bs=batch_size)


def train_model(
    image_path,
    batch_size,
    epochs,
    save_path,
    num_classes=2,
    arch="resnet34",
    pretrained=True,
    **learner_kwargs,
):
    """
    Train the segmentation model with the provided parameters.
    """
    # Prepare data loaders
    dls = prepare_data_loaders(image_path, batch_size)

    # Create learner
    arch_func = globals().get(arch, resnet34)
    learn = unet_learner(
        dls,
        arch_func,
        n_in=4,
        n_out=num_classes,
        pretrained=pretrained,
        metrics=[Dice()],
        normalize=False,
        **learner_kwargs,
    )

    # Move model to GPU if available
    if torch.cuda.is_available():
        learn.model.cuda()

    # Train the model
    learn.fine_tune(epochs)

    # Save the model if a save path is provided
    if save_path:
        learn.export(save_path)
        print(f"Model saved at: {save_path}")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a segmentation model using fastai and rasterio."
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the folder containing input images.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training."
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of epochs for training."
    )
    parser.add_argument(
        "--save_model",
        type=str,
        default=f'models/fastai_unet_{datetime.now().strftime("%d_%m_%Y_%H%M")}',
        help="Path to save the trained model.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=2,
        help="Number of classes for segmentation (default is binary).",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet34",
        help="Model architecture to use (default is resnet34). Available options: resnet18, resnet34, resnet50, resnet101, resnet152.",
    )
    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Specify to not use a pre-trained model.",
    )

    args = parser.parse_args()

    # Convert paths to Pathlib objects
    image_path = Path(args.image_path)

    # Verify that paths exist
    if not image_path.exists():
        raise FileNotFoundError("Image path does not exist.")

    # Train the model
    learner_kwargs = vars(args).copy()
    learner_kwargs = {
        k: learner_kwargs[k]
        for k in learner_kwargs
        if k
        not in [
            "image_path",
            "mask_path",
            "batch_size",
            "epochs",
            "save_model",
            "num_classes",
            "arch",
            "no_pretrained",
        ]
    }
    pretrained = not args.no_pretrained
    train_model(
        image_path,
        args.batch_size,
        args.epochs,
        args.save_model,
        args.num_classes,
        args.arch,
        pretrained,
        **learner_kwargs,
    )


if __name__ == "__main__":
    main()
