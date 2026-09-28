import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * bce_loss
        return focal_loss.mean()


class MultiTaskNowcastLightning(pl.LightningModule):
    """
    Lightning wrapper around IndiaConvLSTM_UNet for the 3-headed multi-hazard
    task: thunderstorm (channel 0), cloudburst (channel 1), flash-flood (channel 2).
    """

    def __init__(self, model, learning_rate=1e-4, w_tstorm=1.0, w_cloudburst=2.0, w_flood=2.0):
        super(MultiTaskNowcastLightning, self).__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.lr = learning_rate
        self.w_tstorm = w_tstorm
        self.w_cloudburst = w_cloudburst
        self.w_flood = w_flood
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.focal_loss = FocalLoss(alpha=0.75, gamma=2.0)

    def forward(self, satellite_data, static_dem):
        return self.model(satellite_data, static_dem)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _compute_multi_task_loss(self, y_pred, y_true):
        pred_tstorm, true_tstorm = y_pred[:, 0, :, :], y_true[:, 0, :, :]
        loss_tstorm = self.bce_loss(pred_tstorm, true_tstorm)

        pred_cburst, true_cburst = y_pred[:, 1, :, :], y_true[:, 1, :, :]
        loss_cburst = self.focal_loss(pred_cburst, true_cburst)

        pred_flood, true_flood = y_pred[:, 2, :, :], y_true[:, 2, :, :]
        loss_flood = self.focal_loss(pred_flood, true_flood)

        total_loss = (self.w_tstorm * loss_tstorm) + (self.w_cloudburst * loss_cburst) + (self.w_flood * loss_flood)
        return total_loss, loss_tstorm, loss_cburst, loss_flood

    def training_step(self, batch, batch_idx):
        satellite_data, static_dem, targets = batch
        predictions = self.forward(satellite_data, static_dem)
        loss, l_tstorm, l_cburst, l_flood = self._compute_multi_task_loss(predictions, targets)

        batch_size = satellite_data.size(0)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("train_tstorm_loss", l_tstorm, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("train_cloudburst_loss", l_cburst, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("train_flood_loss", l_flood, on_step=False, on_epoch=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        satellite_data, static_dem, targets = batch
        predictions = self.forward(satellite_data, static_dem)
        loss, l_tstorm, l_cburst, l_flood = self._compute_multi_task_loss(predictions, targets)

        batch_size = satellite_data.size(0)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("val_tstorm_loss", l_tstorm, on_epoch=True, batch_size=batch_size)
        self.log("val_cloudburst_loss", l_cburst, on_epoch=True, batch_size=batch_size)
        self.log("val_flood_loss", l_flood, on_epoch=True, batch_size=batch_size)
        return loss


class Precip_regression_base(pl.LightningModule):
    """
    Generic base class for single-target precipitation-regression models
    (used by the plain UNet / UNetDS_Attention variants). Not used by the
    IndiaConvLSTM_UNet multi-hazard model, which is a plain nn.Module
    wrapped directly by MultiTaskNowcastLightning above.
    """

    def __init__(self, hparams):
        super().__init__()
        hp = hparams if isinstance(hparams, dict) else vars(hparams)
        self.save_hyperparameters(hp)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        lr = self.hparams.get('lr', 1e-3)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer