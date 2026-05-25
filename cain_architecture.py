"""
CAIN-GAN Architecture Implementation
Based on "Automated site planning using CAIN-GAN model" (Automation in Construction, 2024)

Key Components:
- Contextual Attention Mechanism
- Dual-path Bottleneck (Hallucination + Attention)
- Residual Blocks
- Spectral Normalization
- Two-stage progressive generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SpectralNorm(nn.Module):
    """Spectral Normalization for weight matrices."""

    def __init__(self, module: nn.Module, name: str = 'weight', n_power_iterations: int = 1):
        super().__init__()
        self.module = module
        self.name = name
        self.n_power_iterations = n_power_iterations

        if not self._is_parametrized():
            self._parametrize()

    def _is_parametrized(self) -> bool:
        try:
            getattr(self.module, f'{self.name}_u')
            return True
        except AttributeError:
            return False

    def _parametrize(self):
        """Initialize u vector for power iteration."""
        w = getattr(self.module, self.name)
        h, w_shape = w.shape[0], w.shape[1:]
        u = torch.randn(1, h, requires_grad=False)
        self.register_buffer(f'{self.name}_u', u)

    def forward(self, module: Optional[nn.Module] = None) -> nn.Module:
        if module is not None:
            self.module = module
        self._update_weights()
        return self.module

    def _update_weights(self):
        """Update weight matrix using power iteration."""
        w = getattr(self.module, self.name)
        batch_size = w.shape[0]
        w_mat = w.reshape(batch_size, -1)
        u = getattr(self.module, f'{self.name}_u')

        for _ in range(self.n_power_iterations):
            v = F.normalize(u @ w_mat, dim=1)
            u = F.normalize(v @ w_mat.t(), dim=1)

        sigma = (u @ w_mat.t() @ v.t()).squeeze()
        w_normalized = w / (sigma + 1e-12)

        setattr(self.module, self.name, w_normalized)
        setattr(self.module, f'{self.name}_u', u)


class ConvBlock(nn.Module):
    """Convolutional block with spectral normalization and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_spectral_norm: bool = True,
        activation: str = "leaky_relu",
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )

        if use_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

        if activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.activation(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block for feature extraction."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_spectral_norm: bool = True,
    ):
        super().__init__()

        self.conv1 = ConvBlock(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_spectral_norm=use_spectral_norm,
            activation="relu",
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        if use_spectral_norm:
            self.conv2 = nn.utils.spectral_norm(self.conv2)

        self.skip_connection = (in_channels == out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.skip_connection:
            out = out + residual
        else:
            out = out + F.interpolate(residual, size=out.shape[-2:], mode='bilinear', align_corners=False)

        return out


class ContextualAttention(nn.Module):
    """
    Contextual Attention Mechanism.
    Extracts characteristic textures from surrounding built environment.
    """

    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()

        self.query_conv = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, foreground: torch.Tensor, background: torch.Tensor) -> torch.Tensor:
        """
        Args:
            foreground: Feature maps from main path (B, C, H, W)
            background: Feature maps from context (B, C, H, W)

        Returns:
            Attention-weighted combination (B, C, H, W)
        """
        # Get shapes
        batch_size, channels, height, width = foreground.shape

        # Compute attention maps
        proj_query = self.query_conv(foreground).view(batch_size, -1, width * height)  # (B, C', N)
        proj_key = self.key_conv(background).view(batch_size, -1, width * height)      # (B, C', N)
        proj_value = self.value_conv(background).view(batch_size, -1, width * height)  # (B, C, N)

        # Attention weights
        attention = torch.bmm(proj_key.permute(0, 2, 1), proj_query)  # (B, N, N)
        attention = F.softmax(attention, dim=1)

        # Apply attention to values
        out = torch.bmm(proj_value, attention)  # (B, C, N)
        out = out.view(batch_size, channels, height, width)

        # Residual connection with learned weight
        out = self.gamma * out + foreground

        return out


class DualPathBottleneck(nn.Module):
    """
    Dual-path bottleneck combining:
    - Top path: Hallucination using residual blocks
    - Bottom path: Contextual attention on background features
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()

        # Top path: Hallucination
        self.residual_block1 = ResidualBlock(channels, channels)
        self.residual_block2 = ResidualBlock(channels, channels)

        # Bottom path: Contextual attention
        self.attention = ContextualAttention(channels, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (B, C, H, W)

        Returns:
            Combined features from both paths (B, C, H, W)
        """
        # Top path: hallucination with residual blocks
        top_path = self.residual_block1(x)
        top_path = self.residual_block2(top_path)

        # Bottom path: contextual attention
        bottom_path = self.attention(x, x)

        # Concatenate and combine
        combined = top_path + bottom_path

        return combined


class CAINGeneratorFootprint(nn.Module):
    """
    CAIN-GAN Generator for Footprint Construction (Stage 1).
    Predicts building footprints conditioned on context and planning guidance.
    """

    def __init__(
        self,
        conditional_channels: int = 5,  # site_context + planning_guidance + footprints + mask
        output_channels: int = 1,
        ngf: int = 64,  # Number of generator filters
    ):
        super().__init__()

        self.ngf = ngf

        # Encoder
        self.encoder = nn.Sequential(
            ConvBlock(conditional_channels, ngf, kernel_size=7, padding=3, activation="relu"),
            ConvBlock(ngf, ngf * 2, kernel_size=4, stride=2, padding=1, activation="relu"),
            ConvBlock(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1, activation="relu"),
            ConvBlock(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1, activation="relu"),
        )

        # Dual-path bottleneck
        self.bottleneck = DualPathBottleneck(ngf * 8)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf, output_channels, kernel_size=7, padding=3),
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Conditional inputs (B, C, 256, 256)
               - Site context, planning guidance, neighboring footprints, mask

        Returns:
            Predicted footprint (B, 1, 256, 256)
        """
        # Encode
        encoded = self.encoder(x)

        # Bottleneck with dual paths
        bottleneck_out = self.bottleneck(encoded)

        # Decode
        footprint = self.decoder(bottleneck_out)

        return footprint


class CAINDiscriminatorFootprint(nn.Module):
    """
    CAIN-GAN Discriminator for Footprint Construction.
    Distinguishes real from generated footprints.
    Uses spectral normalization for training stability.
    """

    def __init__(
        self,
        input_channels: int = 1,
        ndf: int = 64,  # Number of discriminator filters
    ):
        super().__init__()

        self.features = nn.Sequential(
            # ConvBlock 1
            ConvBlock(
                input_channels,
                ndf,
                kernel_size=4,
                stride=2,
                padding=1,
                use_spectral_norm=True,
                activation="leaky_relu",
            ),
            # ConvBlock 2
            ConvBlock(
                ndf,
                ndf * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                use_spectral_norm=True,
                activation="leaky_relu",
            ),
            # ConvBlock 3
            ConvBlock(
                ndf * 2,
                ndf * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                use_spectral_norm=True,
                activation="leaky_relu",
            ),
            # ConvBlock 4
            ConvBlock(
                ndf * 4,
                ndf * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                use_spectral_norm=True,
                activation="leaky_relu",
            ),
        )

        # Classification layer
        self.classifier = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input footprint (B, 1, 256, 256)

        Returns:
            Probability prediction (B, 1, 1, 1)
        """
        features = self.features(x)
        output = self.classifier(features)
        return output


class CAINGeneratorHeight(nn.Module):
    """
    CAIN-GAN Generator for Height Completion (Stage 2).
    Predicts building heights based on constructed footprints and context.
    """

    def __init__(
        self,
        conditional_channels: int = 5,  # site_context + planning_guidance + mask + constructed footprint
        output_channels: int = 1,
        ngf: int = 64,
    ):
        super().__init__()

        self.ngf = ngf

        # Encoder
        self.encoder = nn.Sequential(
            ConvBlock(conditional_channels, ngf, kernel_size=7, padding=3, activation="relu"),
            ConvBlock(ngf, ngf * 2, kernel_size=4, stride=2, padding=1, activation="relu"),
            ConvBlock(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1, activation="relu"),
            ConvBlock(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1, activation="relu"),
        )

        # Dual-path bottleneck
        self.bottleneck = DualPathBottleneck(ngf * 8)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf, output_channels, kernel_size=7, padding=3),
            nn.ReLU(),  # Heights are non-negative
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Conditional inputs (B, C, 256, 256)
               - Site context, planning guidance, mask, constructed footprints

        Returns:
            Predicted heights (B, 1, 256, 256)
        """
        encoded = self.encoder(x)
        bottleneck_out = self.bottleneck(encoded)
        heights = self.decoder(bottleneck_out)

        return heights


class CAINDiscriminatorHeight(nn.Module):
    """
    CAIN-GAN Discriminator for Height Completion.
    """

    def __init__(
        self,
        input_channels: int = 1,
        ndf: int = 64,
    ):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(
                input_channels,
                ndf,
                kernel_size=4,
                stride=2,
                padding=1,
                use_spectral_norm=True,
                activation="leaky_relu",
            ),
            ConvBlock(
                ndf,
                ndf * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                use_spectral_norm=True,
                activation="leaky_relu",
            ),
            ConvBlock(
                ndf * 2,
                ndf * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                use_spectral_norm=True,
                activation="leaky_relu",
            ),
            ConvBlock(
                ndf * 4,
                ndf * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                use_spectral_norm=True,
                activation="leaky_relu",
            ),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        output = self.classifier(features)
        return output


class CAINGANModel(nn.Module):
    """
    Complete CAIN-GAN Model with two-stage progressive generation.
    """

    def __init__(
        self,
        conditional_channels: int = 5,
        ngf: int = 64,
        ndf: int = 64,
    ):
        super().__init__()

        # Stage 1: Footprint construction
        self.generator_footprint = CAINGeneratorFootprint(
            conditional_channels=conditional_channels,
            output_channels=1,
            ngf=ngf,
        )
        self.discriminator_footprint = CAINDiscriminatorFootprint(
            input_channels=1,
            ndf=ndf,
        )

        # Stage 2: Height completion
        self.generator_height = CAINGeneratorHeight(
            conditional_channels=conditional_channels,
            output_channels=1,
            ngf=ngf,
        )
        self.discriminator_height = CAINDiscriminatorHeight(
            input_channels=1,
            ndf=ndf,
        )

    def forward_footprint(self, conditional: torch.Tensor) -> torch.Tensor:
        """Generate building footprints."""
        return self.generator_footprint(conditional)

    def forward_height(self, conditional: torch.Tensor) -> torch.Tensor:
        """Generate building heights."""
        return self.generator_height(conditional)

    def forward(
        self,
        conditional: torch.Tensor,
        stage: str = "footprint",
    ) -> torch.Tensor:
        """
        Args:
            conditional: Input features (B, C, 256, 256)
            stage: "footprint" or "height"

        Returns:
            Generated output
        """
        if stage == "footprint":
            return self.forward_footprint(conditional)
        elif stage == "height":
            return self.forward_height(conditional)
        else:
            raise ValueError(f"Unknown stage: {stage}")


if __name__ == "__main__":
    print("CAIN-GAN Architecture Components")
    print("=" * 70)

    # Test generator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    conditional_channels = 5

    # Create model
    model = CAINGANModel(conditional_channels=conditional_channels)
    model = model.to(device)

    # Test footprint generation
    x = torch.randn(batch_size, conditional_channels, 256, 256).to(device)
    footprint = model.forward_footprint(x)
    print(f"✓ Footprint generator output: {footprint.shape}")

    # Test height generation
    heights = model.forward_height(x)
    print(f"✓ Height generator output: {heights.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total trainable parameters: {total_params:,}")
