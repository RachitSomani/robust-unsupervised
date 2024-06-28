from .prelude import *
from lpips import LPIPS


class MultiscaleLPIPS:
    def __init__(
        self,
        min_loss_res: int = 16,
        level_weights: List[float] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        l1_weight: float = 0.1
    ):
        super().__init__()
        self.min_loss_res = min_loss_res
        self.weights = level_weights
        self.l1_weight = l1_weight
        self.lpips_network = LPIPS(net="vgg", verbose=False).cuda()

    def measure_lpips(self, x, y, mask):
        if mask is not None:
            # To avoid biasing the results towards black pixels, but random noise in the masked areas
            noise = (torch.randn_like(x) + 0.5) / 2.0
            x = x + noise * (1.0 - mask)
            y = y + noise * (1.0 - mask)

        return self.lpips_network(x, y, normalize=True).mean() 

    def __call__(self, f_hat, x_clean: Tensor, y: Tensor, mask: Optional[Tensor] = None, consistency_weight: float = 0.1):
        x = f_hat(x_clean)

        losses = []

        if mask is not None:
            mask = F.interpolate(mask, y.shape[-1], mode="area")

        x_perturbed = x_clean + torch.randn_like(x_clean) * 0.01  # Add small perturbations
        x_perturbed = f_hat(x_perturbed)
        consistency_loss = F.l1_loss(x, x_perturbed)

        x_perturbed = F.interpolate(x_perturbed, size=y.shape[2:], mode='bilinear', align_corners=False)
            
        
        for weight in self.weights:
            # At extremely low resolutions, LPIPS stops making sense, so omit those
            if y.shape[-1] <= self.min_loss_res:
                break
            
            if weight > 0:
                loss_x = self.measure_lpips(x, y, mask)
                loss_x_perturbed = self.measure_lpips(x_perturbed, y, mask)
                symmetric_loss = (loss_x + loss_x_perturbed) / 2.0
                losses.append(weight * symmetric_loss)

            if mask is not None:
                mask = F.avg_pool2d(mask, 2)

            x = F.avg_pool2d(x, 2)
            x_clean = F.avg_pool2d(x_clean, 2)
            y = F.avg_pool2d(y, 2)
        
        total = torch.stack(losses).sum(dim=0) if len(losses) > 0 else 0.0
        l1 = self.l1_weight * F.l1_loss(x, y)
        total_loss = total + l1 + consistency_weight * consistency_loss
        return total_loss
        
