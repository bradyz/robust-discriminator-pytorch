import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.inception import inception_v3


INCEPTION_POOL3_DIM = 2048
EPS = 1e-6

cache = {'initialized': False}


class Normalize(nn.Module):
    def forward(self, x):
        return (x - 0.5) * 2.0


class Inception3Pool3(nn.Module):
    """
    Modified version from -
    https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py.
    """
    def __init__(self):
        super().__init__()

        self.model = inception_v3(pretrained=True, transform_input=False)

    @torch.no_grad()
    def forward(self, img):
        # 299 x 299 x 3
        x = self.model.Conv2d_1a_3x3(img)
        x = self.model.Conv2d_2a_3x3(x)
        x = self.model.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.model.Conv2d_3b_1x1(x)
        x = self.model.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.model.Mixed_5b(x)
        x = self.model.Mixed_5c(x)
        x = self.model.Mixed_5d(x)
        x = self.model.Mixed_6a(x)
        x = self.model.Mixed_6b(x)
        x = self.model.Mixed_6c(x)
        x = self.model.Mixed_6d(x)
        x = self.model.Mixed_6e(x)
        x = self.model.Mixed_7a(x)
        x = self.model.Mixed_7b(x)
        x = self.model.Mixed_7c(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # 1 x 1 x 2048

        return x.squeeze()


def upsample_inception3_pool3(size=(299, 299)):
    return torch.nn.Sequential(
            Normalize(),
            nn.Upsample(size=size, mode='bilinear', align_corners=True),
            Inception3Pool3()).eval()


def get_mu_sigma(network, image_generator, n, device):
    features = list()

    while len(features) < n:
        x = next(image_generator).to(device)
        phi = network(x).detach().cpu().numpy()

        features.extend(phi)

    return np.mean(features, axis=0), np.cov(features, rowvar=False)


def frechet_distance(mu_1, sigma_1, mu_2, sigma_2):
    eps_matrix = EPS * np.eye(INCEPTION_POOL3_DIM)
    sqrtm = scipy.linalg.sqrtm(np.dot(sigma_1, sigma_2) + eps_matrix)

    lhs = np.linalg.norm(mu_1 - mu_2, 2)
    rhs = np.trace(sigma_1 + sigma_2 - 2.0 * sqrtm)

    return lhs + rhs


def compute_fid_score(fake_generator, real_generator, device, n=10000):
    """
    images must be a torch tensor of shape (N, C, H, W) in [0, 1].
    """
    if cache['initialized']:
        inception = cache['inception']
        mu_2, sigma_2 = cache['mu_2'], cache['sigma_2']
    else:
        inception = upsample_inception3_pool3().to(device)
        mu_2, sigma_2 = get_mu_sigma(inception, real_generator, n, device)

        cache['initialized'] = True
        cache['inception'] = inception
        cache['mu_2'], cache['sigma_2'] = mu_2, sigma_2

    mu_1, sigma_1 = get_mu_sigma(inception, fake_generator, n, device)

    return frechet_distance(mu_1, sigma_1, mu_2, sigma_2)
