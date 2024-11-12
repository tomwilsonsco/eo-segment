import argparse
from datetime import datetime
import rasterio as rio
import numpy as np
from fastai.vision.all import *
from pathlib import Path


def open_tif_rasterio(fn):
    """
    Open a TIF file using rasterio and convert it to float32.

    Args:
        fn (str): File path to the TIF image.

    Returns:
        numpy.ndarray: The image data as a NumPy array with dtype float32.
    """
    with rio.open(fn) as src:
        img = src.read()  # Reads as (channels, height, width)
    img = img.astype(np.float32)  # Ensure the datatype is float32
    return img


def get_x(file_path):
    """
    Get the file path for the input image.

    Args:
        file_path (Path): Path to the image file.

    Returns:
        Path: The file path itself.
    """
    return file_path


def get_y(file_path):
    """
    Get the corresponding mask file path for the input image.

    Args:
        file_path (Path): Path to the image file.

    Returns:
        Path: The file path to the mask image corresponding to the input image.
    """
    return file_path.parent.parent / "chips_seg" / file_path.name


def prepare_data_loaders(image_path, batch_size):
    """
    Prepare data loaders for training and validation.

    Args:
        image_path (Path): Path to the directory containing input images.
        batch_size (int): Batch size for training and validation.

    Returns:
        DataLoaders: Fastai DataLoaders object for training and validation.
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

    Args:
        image_path (Path): Path to the directory containing input images.
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs for training.
        save_path (str): Path to save the trained model.
        num_classes (int, optional): Number of classes for segmentation. Defaults to 2.
        arch (str, optional): Model architecture to use. Defaults to "resnet34".
        pretrained (bool, optional): Whether to use a pre-trained model. Defaults to True.
        **learner_kwargs: Additional keyword arguments for the learner.

    Returns:
        None
    """
    # Prepare data loaders
    dls = prepare_data_loaders(image_path, batch_size)

    # Create learner
    arch_dict = {
        "resnet18": resnet18,
        "resnet34": resnet34,
        "resnet50": resnet50,
        "resnet101": resnet101,
        "resnet152": resnet152,
    }
    arch_func = arch_dict.get(arch, resnet34)
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
    """
    Parse command-line arguments and start model training.

    Args:
        None

    Returns:
        None
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a segmentation model using fastai and rasterio."
    )
    parser.add_argument(
        "--image-path",
        type=str,
        required=True,
        help="Path to the folder containing input images.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training. Defaults to 16.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs for training. Defaults to 1.",
    )
    parser.add_argument(
        "--save-model",
        type=str,
        default=f'models/fastai_unet_{datetime.now().strftime("%d_%m_%Y_%H%M")}',
        help="Path to save the trained model. Default name uses datetime",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="Number of classes for segmentation (default is 2 for binary).",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet34",
        help="Model architecture to use (default is resnet34). Available options: resnet18, resnet34, resnet50, resnet101, \
             resnet152.",
    )
    parser.add_argument(
        "--no-pretrained",
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
