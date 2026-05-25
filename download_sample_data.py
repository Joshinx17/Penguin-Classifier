import random
from pathlib import Path
from PIL import Image, ImageDraw


def generate_synthetic_penguin(species, path, index):
    # Base size 224x224
    img = Image.new("RGB", (224, 224), color=(220, 225, 230))  # Ice background
    draw = ImageDraw.Draw(img)

    # Apply minor random variations to help the model learn
    body_shift_x = random.randint(-5, 5)
    body_shift_y = random.randint(-5, 5)
    head_shift_x = random.randint(-3, 3)
    head_shift_y = random.randint(-3, 3)

    # Draw body (black/charcoal oval)
    body_color = (20, 24, 30)
    draw.ellipse(
        [
            40 + body_shift_x,
            60 + body_shift_y,
            180 + body_shift_x,
            210 + body_shift_y,
        ],
        fill=body_color,
    )

    # Draw chest (white oval)
    draw.ellipse(
        [
            65 + body_shift_x,
            90 + body_shift_y,
            155 + body_shift_x,
            210 + body_shift_y,
        ],
        fill=(240, 242, 245),
    )

    # Draw head
    draw.ellipse(
        [
            72 + head_shift_x,
            20 + head_shift_y,
            152 + head_shift_x,
            100 + head_shift_y,
        ],
        fill=body_color,
    )

    # Draw beak/bill
    beak_color = (255, 120, 0) if species == "Gentoo" else (40, 30, 20)
    draw.polygon(
        [
            (112 + head_shift_x, 60 + head_shift_y),
            (145 + head_shift_x, 65 + head_shift_y),
            (112 + head_shift_x, 70 + head_shift_y),
        ],
        fill=beak_color,
    )

    if species == "Adelie":
        # White eye ring
        draw.ellipse(
            [
                98 + head_shift_x,
                48 + head_shift_y,
                108 + head_shift_x,
                58 + head_shift_y,
            ],
            fill=(255, 255, 255),
        )
        draw.ellipse(
            [
                101 + head_shift_x,
                51 + head_shift_y,
                105 + head_shift_x,
                55 + head_shift_y,
            ],
            fill=(0, 0, 0),
        )  # pupil
    elif species == "Chinstrap":
        # Black strap under the chin
        draw.line(
            [
                (80 + head_shift_x, 80 + head_shift_y),
                (112 + head_shift_x, 90 + head_shift_y),
                (144 + head_shift_x, 80 + head_shift_y),
            ],
            fill=(0, 0, 0),
            width=3,
        )
        # Eye
        draw.ellipse(
            [
                98 + head_shift_x,
                48 + head_shift_y,
                106 + head_shift_x,
                56 + head_shift_y,
            ],
            fill=(0, 0, 0),
        )
    elif species == "Gentoo":
        # White patch on the head (bonnet)
        draw.polygon(
            [
                (85 + head_shift_x, 30 + head_shift_y),
                (112 + head_shift_x, 22 + head_shift_y),
                (135 + head_shift_x, 30 + head_shift_y),
                (120 + head_shift_x, 50 + head_shift_y),
                (100 + head_shift_x, 50 + head_shift_y),
            ],
            fill=(255, 255, 255),
        )
        # Eye
        draw.ellipse(
            [
                98 + head_shift_x,
                48 + head_shift_y,
                106 + head_shift_x,
                56 + head_shift_y,
            ],
            fill=(0, 0, 0),
        )

    # Random slight rotation
    img = img.rotate(random.uniform(-5, 5), fillcolor=(220, 225, 230))
    img.save(path)


def main():
    data_dir = Path("data/penguins")
    data_dir.mkdir(parents=True, exist_ok=True)

    print("--- Generating Stylized Synthetic Penguin Dataset ---")
    print("This generator creates distinct 224x224 penguin vectors for local training.")

    species_list = ["Adelie", "Chinstrap", "Gentoo"]
    images_per_class = 15  # Good size for fast transfer training + validation split

    for species in species_list:
        species_dir = data_dir / species
        species_dir.mkdir(exist_ok=True)
        print(f"Generating {images_per_class} images for {species}...")
        for i in range(1, images_per_class + 1):
            dest_path = species_dir / f"sample-{i}.jpg"
            generate_synthetic_penguin(species, dest_path, i)

    # Save a test image
    test_image_path = Path("data/test_penguin.jpg")
    print("\nGenerating separate Chinstrap test image...")
    generate_synthetic_penguin("Chinstrap", test_image_path, 999)

    print("\n--- Generation Completed! ---")
    print(f"Dataset generated at: {data_dir.resolve()}")
    print("You can train the model now by running:")
    print("  python train_image_classifier.py --data-dir data/penguins --epochs 5")


if __name__ == "__main__":
    main()
