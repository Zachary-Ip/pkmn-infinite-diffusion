import argparse
import ast
import configparser
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from diffusers.optimization import get_scheduler
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

from src.models.unet import UNet
from src.training.ddim import DDIMScheduler
from src.training.ema import EMA
from src.utils.dataset import PokemonDataset
from src.utils.utils import save_images

SEED = 123

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def main(args):

    # Set up model objects
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        3,
        image_size=args.resolution,
        hidden_dims=args.hidden_dims,
        use_flash_attn=args.use_flash_attn,
    ).to(device)

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=args.n_timesteps, beta_schedule="cosine"
    )

    if args.pretrained_model_path:
        pretrained = torch.load(args.pretrained_model_path)["model_state"]
        model.load_state_dict(pretrained)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
    )

    transform = transforms.Compose(
        [
            transforms.Resize((args.resolution, args.resolution)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5,), (0.5,)
            ),  # normalize to [-1, 1] for faster convergence and numerical stability
        ]
    )
    dataset = PokemonDataset(args.dataset_path, transform=transform)
    train_dataloader = DataLoader(dataset, batch_size=args.train_batch_size)
    steps_per_epoch = len(train_dataloader)
    total_num_steps = (
        steps_per_epoch * args.num_epochs // args.gradient_accumulation_steps
    )
    total_num_steps += int(total_num_steps * 10 / 100)
    gamma = args.gamma
    ema = EMA(model, gamma, total_num_steps)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=total_num_steps,
    )

    # summary(model, [(1, 3, args.resolution, args.resolution), (1,)], verbose=1)
    scaler = GradScaler(enabled=args.fp16_precision)
    global_step = 0
    losses = []

    # Begin training
    for epoch in range(args.num_epochs):
        progress_bar = tqdm(total=steps_per_epoch)
        progress_bar.set_description(f"Epoch {epoch}")
        losses_log = 0
        for step, batch in enumerate(train_dataloader):
            orig_images = batch["image"].to(device)
            metadata = batch["metadata"].to(device)

            batch_size = orig_images.shape[0]
            noise = torch.randn(orig_images.shape).to(device)
            timesteps = torch.randint(
                0, noise_scheduler.num_train_timesteps, (batch_size,), device=device
            ).long()
            noisy_images = noise_scheduler.add_noise(orig_images, noise, timesteps)
            optimizer.zero_grad()
            with autocast(enabled=args.fp16_precision, device_type=device.type):
                noise_pred = model(noisy_images, timesteps, metadata)["sample"]
                loss = F.l1_loss(noise_pred, noise)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ema.update_params(gamma)
            gamma = ema.update_gamma(global_step)
            if args.use_clip_grad:
                clip_grad_norm_(model.parameters(), 1.0)

            lr_scheduler.step()
            progress_bar.update(1)
            losses_log += loss.detach().item()
            logs = {
                "loss_avg": losses_log / (step + 1),
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
                "gamma": gamma,
            }
            progress_bar.set_postfix(**logs)
            global_step += 1

            # Generate sample images for visual inspection
            if global_step % args.save_model_steps == 0:
                ema.ema_model.eval()
                with torch.no_grad():
                    # has to be instantiated every time, because of reproducibility
                    generator = torch.manual_seed(0)
                    generated_images = noise_scheduler.generate(
                        ema.ema_model,
                        num_inference_steps=n_inference_timesteps,
                        generator=generator,
                        eta=1.0,
                        batch_size=args.eval_batch_size,
                    )

                    save_images(generated_images, epoch, args)
                    out_dir = Path(args.output_dir)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = os.path.join(
                        args.output_dir, f"checkpoint_{epoch}_{global_step}.pth"
                    )

                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "ema_model_state": ema.ema_model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                        },
                        out_path,
                    )

            progress_bar.close()
            losses.append(losses_log / (step + 1))


if __name__ == "__main__":
    # Initialize argument parser for command-line arguments
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config_file",
        type=str,
        default="configs/config.ini",
        help="Path to the config file",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Display model info",
    )
    args = parser.parse_args()

    # Read the config file
    config = configparser.ConfigParser()
    config.read(args.config_file)
    # Load parameters from the config file
    dataset_name = config.get(
        "settings", "dataset_name", fallback="pkmn-infinite-fusion-sprites"
    )
    dataset_path = config.get("settings", "dataset_path", fallback="./data/raw/")
    hidden_dims = ast.literal_eval(
        config.get("settings", "hidden_dims", fallback="[64, 128, 256]")
    )
    n_timesteps = config.getint("settings", "n_timesteps", fallback=1000)
    n_inference_timesteps = config.getint(
        "settings", "n_inference_timesteps", fallback=250
    )
    resolution = config.getint("settings", "resolution", fallback=32)
    output_dir = config.get("settings", "output_dir", fallback="trained_models/")
    samples_dir = config.get("settings", "samples_dir", fallback="test_samples/")
    loss_logs_dir = config.get("settings", "loss_logs_dir", fallback="training_logs")
    cache_dir = config.get("settings", "cache_dir", fallback=None)
    train_batch_size = config.getint("settings", "train_batch_size", fallback=16)
    eval_batch_size = config.getint("settings", "eval_batch_size", fallback=16)
    num_epochs = config.getint("settings", "num_epochs", fallback=1)
    save_model_steps = config.getint("settings", "save_model_steps", fallback=1000)
    gradient_accumulation_steps = config.getint(
        "settings", "gradient_accumulation_steps", fallback=1
    )
    learning_rate = config.getfloat("settings", "learning_rate", fallback=1e-4)
    lr_scheduler = config.get("settings", "lr_scheduler", fallback="cosine")
    lr_warmup_steps = config.getint("settings", "lr_warmup_steps", fallback=100)
    adam_beta1 = config.getfloat("settings", "adam_beta1", fallback=0.9)
    adam_beta2 = config.getfloat("settings", "adam_beta2", fallback=0.99)
    adam_weight_decay = config.getfloat("settings", "adam_weight_decay", fallback=0.0)
    use_clip_grad = config.getboolean("settings", "use_clip_grad", fallback=False)
    use_flash_attn = config.getboolean("settings", "use_flash_attn", fallback=False)
    logging_dir = config.get("settings", "logging_dir", fallback="logs")
    pretrained_model_path = ast.literal_eval(
        config.get("settings", "pretrained_model_path", fallback=None)
    )
    fp16_precision = config.getboolean("settings", "fp16_precision", fallback=False)
    gamma = config.getfloat("settings", "gamma", fallback=0.996)
    # If both are not provided in the config file, raise an error
    if dataset_name is None and dataset_path is None:
        raise ValueError(
            "You must specify either a dataset name from the hub or a train data directory."
        )

    # Replace the arguments with the values from the config file
    # Define Args class

    class Args:
        def __init__(
            self,
            dataset_name,
            dataset_path,
            hidden_dims,
            n_timesteps,
            n_inference_timesteps,
            resolution,
            output_dir,
            samples_dir,
            loss_logs_dir,
            cache_dir,
            train_batch_size,
            eval_batch_size,
            num_epochs,
            save_model_steps,
            gradient_accumulation_steps,
            learning_rate,
            lr_scheduler,
            lr_warmup_steps,
            adam_beta1,
            adam_beta2,
            adam_weight_decay,
            use_clip_grad,
            use_flash_attn,
            logging_dir,
            pretrained_model_path,
            fp16_precision,
            gamma,
        ):
            self.dataset_name = dataset_name
            self.dataset_path = dataset_path
            self.hidden_dims = hidden_dims
            self.n_timesteps = n_timesteps
            self.n_inference_timesteps = n_inference_timesteps
            self.resolution = resolution
            self.output_dir = output_dir
            self.samples_dir = samples_dir
            self.loss_logs_dir = loss_logs_dir
            self.cache_dir = cache_dir
            self.train_batch_size = train_batch_size
            self.eval_batch_size = eval_batch_size
            self.num_epochs = num_epochs
            self.save_model_steps = save_model_steps
            self.gradient_accumulation_steps = gradient_accumulation_steps
            self.learning_rate = learning_rate
            self.lr_scheduler = lr_scheduler
            self.lr_warmup_steps = lr_warmup_steps
            self.adam_beta1 = adam_beta1
            self.adam_beta2 = adam_beta2
            self.adam_weight_decay = adam_weight_decay
            self.use_clip_grad = use_clip_grad
            self.use_flash_attn = use_flash_attn
            self.logging_dir = logging_dir
            self.pretrained_model_path = pretrained_model_path
            self.fp16_precision = fp16_precision
            self.gamma = gamma

    # Initialize and pass arguments to main
    config_args = Args(
        dataset_name,
        dataset_path,
        hidden_dims,
        n_timesteps,
        n_inference_timesteps,
        resolution,
        output_dir,
        samples_dir,
        loss_logs_dir,
        cache_dir,
        train_batch_size,
        eval_batch_size,
        num_epochs,
        save_model_steps,
        gradient_accumulation_steps,
        learning_rate,
        lr_scheduler,
        lr_warmup_steps,
        adam_beta1,
        adam_beta2,
        adam_weight_decay,
        use_clip_grad,
        use_flash_attn,
        logging_dir,
        pretrained_model_path,
        fp16_precision,
        gamma,
    )
    main(config_args)
