import torch
from torch import nn

from models.unet_parts import Down, DoubleConv, Up, OutConv
from models.unet_parts_depthwise_separable import DoubleConvDS, UpDS, DownDS
from models.layers import CBAM
from models.regression_lightning import Precip_regression_base
from models.convlstm import ConvLSTM


class UNet(Precip_regression_base):
    def __init__(self, hparams):
        super(UNet, self).__init__(hparams=hparams)
        self.n_channels = self.hparams.n_channels
        self.n_classes = self.hparams.n_classes
        self.bilinear = self.hparams.bilinear

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNetDS_Attention(Precip_regression_base):
    def __init__(self, hparams):
        super(UNetDS_Attention, self).__init__(hparams=hparams)
        self.n_channels = self.hparams.n_channels
        self.n_classes = self.hparams.n_classes
        self.bilinear = self.hparams.bilinear
        reduction_ratio = self.hparams.reduction_ratio
        kernels_per_layer = self.hparams.kernels_per_layer

        self.inc = DoubleConvDS(self.n_channels, 64, kernels_per_layer=kernels_per_layer)
        self.cbam1 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam2 = CBAM(128, reduction_ratio=reduction_ratio)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.cbam3 = CBAM(256, reduction_ratio=reduction_ratio)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
        self.cbam4 = CBAM(512, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        self.cbam5 = CBAM(1024 // factor, reduction_ratio=reduction_ratio)
        self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4)
        x5Att = self.cbam5(x5)
        x = self.up1(x5Att, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits


class IndiaConvLSTM_UNet(nn.Module):
    """
    Spatiotemporal ConvLSTM-UNet for India multi-hazard nowcasting (MOSDAC).

    This is a plain nn.Module (not a LightningModule) because it is used as
    a backbone inside MultiTaskNowcastLightning (models/regression_lightning.py),
    which is itself the LightningModule that drives training.

    Forward inputs:
      dynamic_seq : (B, T, dynamic_channels, H, W)  - satellite frame sequence
      static_dem  : (B, static_channels, H, W)      - static terrain / DEM layer

    Forward output:
      (B, n_classes, H, W) - sigmoid probabilities, one channel per hazard
      (default n_classes=3: thunderstorm, cloudburst, flash-flood)

    At every encoder scale, a ConvLSTM fuses the temporal sequence into a
    single "nowcasting context" feature map, which is then decoded through a
    standard U-Net-style decoder with skip connections.
    """

    def __init__(self, dynamic_channels=2, static_channels=1, n_classes=3,
                 base_channels=48, kernels_per_layer=1):
        super().__init__()
        self.dynamic_channels = dynamic_channels
        self.static_channels = static_channels
        self.n_classes = n_classes

        in_ch = dynamic_channels + static_channels
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8

        # Shared per-frame spatial encoder (applied to every timestep with shared weights)
        self.inc = DoubleConvDS(in_ch, c1, kernels_per_layer=kernels_per_layer)
        self.down1 = DownDS(c1, c2, kernels_per_layer=kernels_per_layer)
        self.down2 = DownDS(c2, c3, kernels_per_layer=kernels_per_layer)
        self.down3 = DownDS(c3, c4, kernels_per_layer=kernels_per_layer)

        # Temporal fusion (ConvLSTM) across the sequence at every encoder scale
        self.lstm1 = ConvLSTM(c1)
        self.lstm2 = ConvLSTM(c2)
        self.lstm3 = ConvLSTM(c3)
        self.lstm4 = ConvLSTM(c4)

        # Decoder
        self.up1 = UpDS(c4 + c3, c3, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(c3 + c2, c2, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(c2 + c1, c1, kernels_per_layer=kernels_per_layer)
        self.outc = OutConv(c1, n_classes)

    def forward(self, dynamic_seq, static_dem):
        b, t, c, h, w = dynamic_seq.shape

        static_exp = static_dem.unsqueeze(1).expand(b, t, self.static_channels, h, w)
        x = torch.cat([dynamic_seq, static_exp], dim=2)              # (B, T, C, H, W)
        x = x.reshape(b * t, c + self.static_channels, h, w)          # fold time into batch

        f1 = self.inc(x)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        f4 = self.down3(f3)

        def to_seq(feat):
            _, cc, hh, ww = feat.shape
            return feat.reshape(b, t, cc, hh, ww)

        ctx1 = self.lstm1(to_seq(f1))
        ctx2 = self.lstm2(to_seq(f2))
        ctx3 = self.lstm3(to_seq(f3))
        ctx4 = self.lstm4(to_seq(f4))

        x = self.up1(ctx4, ctx3)
        x = self.up2(x, ctx2)
        x = self.up3(x, ctx1)
        logits = self.outc(x)
        return torch.sigmoid(logits)
