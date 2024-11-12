from rschip import ImageChip
from rschip import RemoveBackgroundOnly
from pathlib import Path


def main():
    output_dir = Path("inputs") / "chips_img"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the ImageChip instance for 128 by 128 tiles
    image_chipper = ImageChip(
        input_image_path="inputs/s2_flow_country_2023_06_16.tif",
        output_path=output_dir,
        pixel_dimensions=128,
        offset=64,
        output_format="tif",
        max_batch_size=1000,
    )

    image_chipper.set_normaliser(
        min_val=[250, 250, 250, 0], max_val=[2500, 2500, 2500, 7000]
    )

    # Generate chips
    image_chipper.chip_image()

    output_dir = Path("inputs") / "chips_seg"

    image_chipper = ImageChip(
        input_image_path="inputs/water_class_img.tif",
        output_path=output_dir,
        output_name="s2_flow_country_2023_06_16",
        pixel_dimensions=128,
        offset=64,
        output_format="tif",
    )
    image_chipper.chip_image()

    remover = RemoveBackgroundOnly(background_val=0, non_background_min=1000)

    # Remove chips with only background
    remover.remove_background_only_files(
        class_chips_dir="inputs/chips_seg", image_chips_dir="inputs/chips_img"
    )


if __name__ == "__main__":
    main()
