import math

import torch
from einops.layers.torch import Rearrange
from torch import nn
from torch.nn import functional as F

from src.models.attention import Attention
from torch.utils.checkpoint import checkpoint


def get_downsample_layer(in_dim, hidden_dim, is_last):
    if not is_last:
        return nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
            nn.Conv2d(in_dim * 4, hidden_dim, 1),
        )
    else:
        return nn.Conv2d(in_dim, hidden_dim, 3, padding=1)


def get_attn_layer(in_dim, use_full_attn, use_flash_attn):
    if use_full_attn:
        return Attention(in_dim, use_flash_attn=use_flash_attn)
    else:
        return nn.Identity()


def get_upsample_layer(in_dim, hidden_dim, is_last):
    if not is_last:
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_dim, hidden_dim, 3, padding=1),
        )
    else:
        return nn.Conv2d(in_dim, hidden_dim, 3, padding=1)


def sinusoidal_embedding(timesteps, dim):
    half_dim = dim // 2
    exponent = -math.log(10000) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32
    )
    exponent = exponent / (half_dim - 1.0)

    emb = torch.exp(exponent).to(device=timesteps.device)
    emb = timesteps[:, None].float() * emb[None, :]

    return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        temb_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=8,
    ):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.time_emb_proj = nn.Sequential(
            nn.SiLU(), torch.nn.Linear(temb_channels, out_channels)
        )

        self.residual_conv = (
            nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.norm1 = nn.GroupNorm(num_channels=out_channels, num_groups=groups)
        self.norm2 = nn.GroupNorm(num_channels=out_channels, num_groups=groups)
        self.nonlinearity = nn.SiLU()

    def forward(self, x, temb):
        residual = self.residual_conv(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.nonlinearity(x)

        temb = self.time_emb_proj(temb)
        x += temb[:, :, None, None]

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.nonlinearity(x)

        return x + residual


class UNet(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_dims=[
            64,
            128,
            256,
        ],  # Defines the number of feature channels at different levels
        image_size=64,
        metadata_dim=18,  # Size of the metadata one-hot vector (adjust as needed)
        use_flash_attn=False,  # Whether to use efficient attention mechanism
    ):
        """
        UNet architecture for diffusion models with metadata conditioning.

        Args:
            in_channels (int): Number of input channels (e.g., 3 for RGB images).
            hidden_dims (list): Feature map sizes at different stages of the U-Net.
            image_size (int): Image resolution.
            metadata_dim (int): Dimension of metadata one-hot vector (e.g., type & egg group).
            use_flash_attn (bool): Flag to use flash attention for efficiency.
        """
        super(UNet, self).__init__()

        self.sample_size = image_size
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims

        timestep_input_dim = hidden_dims[0]  # Base feature size for time embeddings
        time_embed_dim = timestep_input_dim * 4  # Expanded embedding size

        # Time embedding network (maps timesteps to a learned representation)
        self.time_embedding = nn.Sequential(
            nn.Linear(timestep_input_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Metadata embedding network (maps metadata one-hot vector to a latent space)
        self.metadata_embedding = nn.Sequential(
            nn.Linear(metadata_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Initial convolutional layer to process input image
        self.init_conv = nn.Conv2d(
            in_channels, out_channels=hidden_dims[0], kernel_size=3, stride=1, padding=1
        )

        # Downsampling (encoder) blocks
        down_blocks = []
        in_dim = hidden_dims[0]  # First layer has the same channels as init_conv output
        for idx, hidden_dim in enumerate(hidden_dims[1:]):
            is_last = idx >= (len(hidden_dims) - 2)  # Is this the last down block?
            is_first = idx == 0  # Is this the first down block?
            use_attn = (
                True if use_flash_attn else not is_first
            )  # Apply attention except for first block
            down_blocks.append(
                nn.ModuleList(
                    [
                        ResidualBlock(
                            in_dim, in_dim, time_embed_dim
                        ),  # Residual block 1
                        ResidualBlock(
                            in_dim, in_dim, time_embed_dim
                        ),  # Residual block 2
                        get_attn_layer(
                            in_dim, use_attn, use_flash_attn
                        ),  # Attention layer
                        get_downsample_layer(
                            in_dim, hidden_dim, is_last
                        ),  # Downsample layer
                    ]
                )
            )
            in_dim = hidden_dim  # Update feature size for next block

        self.down_blocks = nn.ModuleList(down_blocks)

        # Bottleneck (middle block)
        mid_dim = hidden_dims[-1]
        self.mid_block1 = ResidualBlock(
            mid_dim, mid_dim, time_embed_dim
        )  # Residual block
        self.mid_attn = Attention(mid_dim)  # Self-attention layer
        self.mid_block2 = ResidualBlock(
            mid_dim, mid_dim, time_embed_dim
        )  # Another residual block

        # Upsampling (decoder) blocks
        up_blocks = []
        in_dim = mid_dim  # Start with highest feature size
        for idx, hidden_dim in enumerate(list(reversed(hidden_dims[:-1]))):
            is_last = idx >= (len(hidden_dims) - 2)  # Is this the last up block?
            use_attn = (
                True if use_flash_attn else not is_last
            )  # Apply attention except last block
            up_blocks.append(
                nn.ModuleList(
                    [
                        ResidualBlock(
                            in_dim + hidden_dim, in_dim, time_embed_dim
                        ),  # Residual block 1
                        ResidualBlock(
                            in_dim + hidden_dim, in_dim, time_embed_dim
                        ),  # Residual block 2
                        get_attn_layer(
                            in_dim, use_attn, use_flash_attn
                        ),  # Attention layer
                        get_upsample_layer(
                            in_dim, hidden_dim, is_last
                        ),  # Upsample layer
                    ]
                )
            )
            in_dim = hidden_dim  # Update feature size for next block

        self.up_blocks = nn.ModuleList(up_blocks)

        # Final residual block and output convolution
        self.out_block = ResidualBlock(
            hidden_dims[0] * 2, hidden_dims[0], time_embed_dim
        )
        self.conv_out = nn.Conv2d(
            hidden_dims[0], out_channels=3, kernel_size=1
        )  # Output RGB image

    def forward(self, sample, timesteps, metadata=None, guidance_scale=1.0):
        """
        Forward pass of the U-Net with optional Classifier-Free Guidance and gradient checkpointing.

        Args:
            sample (Tensor): Input image tensor (B, C, H, W).
            timesteps (Tensor): Current diffusion timestep (B,).
            metadata (Tensor, optional): One-hot encoded metadata (B, metadata_dim). Can be None for CFG.
            guidance_scale (float): Strength of classifier-free guidance (default 1.0, no guidance).

        Returns:
            dict: Output sample tensor with generated image.
        """

        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=sample.device
            )

        timesteps = torch.flatten(timesteps)
        timesteps = timesteps.broadcast_to(sample.shape[0])

        # Compute time embedding
        t_emb = sinusoidal_embedding(timesteps, self.hidden_dims[0])
        t_emb = self.time_embedding(t_emb)

        if metadata is not None:
            # Compute metadata embedding and add to time embedding
            m_emb = self.metadata_embedding(metadata)
            t_emb = t_emb + m_emb  # Feature addition for conditioning

        # Initial convolution to process the input image
        x = self.init_conv(sample)
        r = x.clone()  # Save for residual connection

        skips = []

        # Encoder (downsampling) with gradient checkpointing
        for block1, block2, attn, downsample in self.down_blocks:
            # Apply checkpointing to the first block
            def create_custom_forward_block1(module, t_embedding):
                def custom_forward(x_tensor):
                    return module(x_tensor, t_embedding)

                return custom_forward

            x = checkpoint(create_custom_forward_block1(block1, t_emb), x)
            skips.append(x)

            # Apply checkpointing to the second block
            def create_custom_forward_block2(module, t_embedding):
                def custom_forward(x_tensor):
                    return module(x_tensor, t_embedding)

                return custom_forward

            x = checkpoint(create_custom_forward_block2(block2, t_emb), x)

            # Apply attention with checkpointing
            def custom_forward_attn(x_tensor):
                return attn(x_tensor)

            x = checkpoint(custom_forward_attn, x)
            skips.append(x)

            # Downsampling is typically less memory-intensive, can skip checkpointing
            x = downsample(x)

        # Bottleneck processing with checkpointing
        def create_custom_forward_mid(module, t_embedding):
            def custom_forward(x_tensor):
                return module(x_tensor, t_embedding)

            return custom_forward

        x = checkpoint(create_custom_forward_mid(self.mid_block1, t_emb), x)

        def custom_forward_mid_attn(x_tensor):
            return self.mid_attn(x_tensor)

        x = checkpoint(custom_forward_mid_attn, x)
        x = checkpoint(create_custom_forward_mid(self.mid_block2, t_emb), x)

        # Decoder (upsampling) with gradient checkpointing
        for block1, block2, attn, upsample in self.up_blocks:
            # Get skip connection
            skip_connection = skips.pop()

            # Custom checkpoint function for first block with concatenation
            def create_custom_forward_up1(module, skip, t_embedding):
                def custom_forward(x_tensor):
                    x_cat = torch.cat((x_tensor, skip), dim=1)
                    return module(x_cat, t_embedding)

                return custom_forward

            x = checkpoint(create_custom_forward_up1(block1, skip_connection, t_emb), x)

            # Get second skip connection
            skip_connection = skips.pop()

            # Custom checkpoint function for second block with concatenation
            def create_custom_forward_up2(module, skip, t_embedding):
                def custom_forward(x_tensor):
                    x_cat = torch.cat((x_tensor, skip), dim=1)
                    return module(x_cat, t_embedding)

                return custom_forward

            x = checkpoint(create_custom_forward_up2(block2, skip_connection, t_emb), x)

            # Apply attention with checkpointing
            def custom_forward_up_attn(x_tensor):
                return attn(x_tensor)

            x = checkpoint(custom_forward_up_attn, x)

            # Upsampling is typically less memory-intensive
            x = upsample(x)

        # Final processing
        def create_custom_forward_out(module, residual, t_embedding):
            def custom_forward(x_tensor):
                x_cat = torch.cat((x_tensor, residual), dim=1)
                return module(x_cat, t_embedding)

            return custom_forward

        x = checkpoint(create_custom_forward_out(self.out_block, r, t_emb), x)
        out = self.conv_out(x)

        if guidance_scale > 1.0:
            # For guidance, we need to compute the unconditioned output
            # Note: We can't use checkpointing for the recursive call
            with torch.no_grad():  # Use no_grad to save more memory
                # Enable evaluation mode to disable dropout, etc.
                self.eval()
                # Compute unconditioned output - store in CPU if needed to save GPU memory
                unconditioned_out = self.forward(
                    sample, timesteps, metadata=None, guidance_scale=1.0
                )["sample"]
                # Return to training mode
                self.train()

            # CFG formula: mix conditioned and unconditioned predictions
            out = unconditioned_out + guidance_scale * (out - unconditioned_out)

        return {"sample": out}
