import torch
import torch.nn as nn
from math import ceil

# Model architecture parameters
base_architecture = [                      # [expand_ratio, channels, repeats, stride, kernel_size]
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3],
]

# Phi values for compound scaling:
#       Depth     : (d) = alpha ^ phi
#       Width     : (w) = beta  ^ phi
#       Resolution: (r) = gamma ^ phi

compoundScaling_params = {                           # (phi_value, resolution, drop_rate)
    "b0": (0, 224, 0.2),  
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5),
}


# Convolution-Network layer
class CNNLayer(nn.Module):
    """
    A standard convolution layer with batch normalization and SiLU activation.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolution kernel.
        stride (int): Stride for the convolution.
        padding (int): Padding for the convolution.
        groups (int): Number of groups for grouped convolution (default: 1).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super(CNNLayer, self).__init__()
        self.cnn = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, groups=groups)
        self.batchNorm = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()  # Swish activation

    def forward(self, x):
        return self.silu(self.batchNorm(self.cnn(x)))


# Squeeze & Excitation approach to capture global context and channel-wise attention
class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block for channel-wise attention.
    Args:
        in_channels (int): Number of input channels.
        compressed_channels (int): Number of compressed channels (reduced dimensionality).
    """
    def __init__(self, in_channels, compressed_channels):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Conv2d(in_channels, compressed_channels, kernel_size=1),  # Reduce channels
            nn.SiLU(),  # Swish activation
            nn.Conv2d(compressed_channels, in_channels, kernel_size=1),  # Restore channels
            nn.Sigmoid(),  # Sigmoid activation for attention weights
        )

    def forward(self, x):
        return x * self.se(x)  # Apply channel-wise attention



# Mobile-Inverted Bottleneck Convolution (MBConv) Block
class MBConvBlock(nn.Module):
    """
    Mobile Inverted Bottleneck Convolution (MBConv) Block.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolution kernel.
        stride (int): Stride for the convolution.
        padding (int): Padding for the convolution.
        expand_ratio (int): Expansion ratio for the inverted bottleneck.
        reduction (int): Reduction ratio for Squeeze-and-Excitation (default: 4).
        survival_prob (float): Survival probability for stochastic depth (default: 0.8).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, expand_ratio, reduction=4, survival_prob=0.8):
        super(MBConvBlock, self).__init__()
        self.survival_prob = survival_prob
        self.use_residual = in_channels == out_channels and stride == 1
        intermediate_channels = in_channels * expand_ratio
        self.expand = in_channels != intermediate_channels
        compressed_channels = int(in_channels / reduction)

        
        self.expand_conv = (
            CNNLayer(in_channels, intermediate_channels, kernel_size=3, stride=1, padding=1)
            if self.expand else nn.Identity()
        )

        # Depthwise convolution + Squeeze-and-Excitation + Projection
        self.conv = nn.Sequential(
            CNNLayer(
                in_channels=intermediate_channels,
                out_channels=intermediate_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=intermediate_channels,  
            ),
            SqueezeExcitation(intermediate_channels, compressed_channels),
            nn.Conv2d(intermediate_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def stochastic_depth(self, x):
        """
        Applies stochastic depth during training for regularization.
        During inference, the function returns the input unchanged.
        """
        if not self.training:
            return x

        
        binary_tensor = (
            torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
        )
        return x * binary_tensor / self.survival_prob

    def forward(self, inputs):
        # Expansion phase
        x = self.expand_conv(inputs)

        # Apply MBConv block
        x = self.conv(x)

        # Add residual connection if applicable
        if self.use_residual:
            return self.stochastic_depth(x) + inputs
        return x



# EfficientNet Model
class EfficientNet(nn.Module):
    """
    EfficientNet model with compound scaling.
    Args:
        version (str): Model version (e.g., "b0", "b1", etc.).
        num_classes (int): Number of output classes.
    """
    def __init__(self, version, num_classes):
        super(EfficientNet, self).__init__()
        width_factor, depth_factor, dropout_rate = self.calculate_factors(version)
        last_channels = ceil(1280 * width_factor)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.features = self.create_featureMap(width_factor, depth_factor, last_channels)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels, num_classes),
        )

    def calculate_factors(self, version, alpha=1.2, beta=1.1):
        """
        Calculates scaling factors for width, depth, and resolution based on the model version.
        """
        phi, res, drop_rate = compoundScaling_params[version]
        depth_factor = alpha ** phi
        width_factor = beta ** phi
        return width_factor, depth_factor, drop_rate

    def create_featureMap(self, width_factor, depth_factor, last_channels):
        """
        Creates the feature map layers for EfficientNet.
        """
        channels = int(32 * width_factor)
        features = [CNNLayer(3, channels, 3, stride=2, padding=1)]
        in_channels = channels

        for expand_ratio, channels, repeats, stride, kernel_size in base_architecture:
            out_channels = 4 * ceil(int(channels * width_factor) / 4)
            layer_repeats = ceil(repeats * depth_factor)

            for layer in range(layer_repeats):
                features.append(
                    MBConvBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        expand_ratio=expand_ratio,
                        stride=stride if layer == 0 else 1,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,  # if k=1: padd=0, k=3: padd=1, k=5: padd=2
                    )
                )
                in_channels = out_channels

        features.append(CNNLayer(in_channels, last_channels, kernel_size=1, stride=1, padding=0))
        return nn.Sequential(*features)

    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(x.view(x.shape[0], -1))


# # Test model-architecture
# def test():
#     device = "mps" if torch.backends.mps.is_available() else "cpu"
#     print(device)
#     version = "b2"
#     phi, res, drop_rate = compoundScaling_params[version]
#     num_examples, num_classes = 4, 10
#     x = torch.randn((num_examples, 3, res, res)).to(device)
#     model = EfficientNet(version=version, num_classes=num_classes).to(device)
#     print(model(x).shape)


# if __name__ == "__main__":
#     test()