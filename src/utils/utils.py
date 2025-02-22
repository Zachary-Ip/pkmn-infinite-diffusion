import os
from datetime import datetime

from PIL import Image
from torchvision import utils


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


def save_images(generated_images, epoch, args, contexts=None):
    images = generated_images["sample"]
    images_processed = (images * 255).round().astype("uint8")

    current_date = datetime.today().strftime("%Y%m%d_%H%M%S")
    out_dir = f"./{args.samples_dir}/{current_date}_{args.dataset_name}_{epoch}/"
    os.makedirs(out_dir)
    for idx, image in enumerate(images_processed):
        image = Image.fromarray(image)
        if contexts:
            image.save(f"{out_dir}/{epoch}_{contexts[idx]}_{idx}.jpeg")
        else:
            image.save(f"{out_dir}/{epoch}_{idx}.jpeg")

    utils.save_image(
        generated_images["sample_pt"],
        f"{out_dir}/{epoch}_grid.jpeg",
        nrow=args.eval_batch_size // 4,
    )
