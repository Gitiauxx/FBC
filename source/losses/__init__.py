from source.losses.cross_entropy_loss import CELoss, MCELoss
from source.losses.l2_loss import L2Loss
from source.losses.mi_loss import MILoss
from source.losses.mmd_loss import MMDLoss
from source.losses.vae_loss import VAELoss
from source.losses.ce_loss import CECondLoss

__all__ = ['CELoss', 'L2Loss', 'MILoss', 'MMDLoss', 'VAELoss', 'CECondLoss', 'MCELoss']
