import argparse
import pathlib

import tqdm
import torch
import torchvision
import numpy as np
import wandb

from .models import Generator, Discriminator
from .data import get_data
from .fid_scorer import compute_fid_score


def train(g, g_optim, d, d_optim, data, config, device):
    g.train()
    d.train()

    criterion = torch.nn.BCELoss()
    real_label = torch.full((config['data_args']['batch_size'],), 1, device=device)
    fake_label = torch.full((config['data_args']['batch_size'],), 0, device=device)

    for i, (x_real, _) in enumerate(tqdm.tqdm(data)):
        x_fake = g(torch.randn([x_real.shape[0], config['z_dim']]).to(device))
        d_fake, phi_fake = d(x_fake)

        # Train G.
        g_optim.zero_grad()
        d_optim.zero_grad()

        g_loss = criterion(d_fake, real_label)
        # g_loss = ((1.0 - d_fake) ** 2).mean()
        # g_loss = -(d_fake).mean()
        g_loss.backward(retain_graph=True)

        g_optim.step()

        # Train D.
        x_real = x_real.to(device)
        x_real.requires_grad_()
        x_real.retain_grad()
        x_fake.retain_grad()

        d_real, phi_real = d(x_real)

        g_optim.zero_grad()
        d_optim.zero_grad()

        d_loss = criterion(d_fake, fake_label) + criterion(d_real, real_label)
        # d_loss = (d_fake ** 2).mean() + ((1.0 - d_real) ** 2).mean()
        # d_loss = (d_fake).mean() + -(d_real).mean()
        d_loss.backward(retain_graph=True)

        metrics = {
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item(),
            }

        if not config['baseline']:
            x_fake_grad = x_fake.grad / (x_fake.grad.reshape(x_fake.shape[0], -1).norm(dim=1)[:, None, None, None] + 1e-8)
            x_fake_adv = x_fake.detach() + config['eps'] * x_fake_grad

            x_real_grad = x_real.grad / (x_real.grad.reshape(x_real.shape[0], -1).norm(dim=1)[:, None, None, None] + 1e-8)
            x_real_adv = x_real.detach() + config['eps'] * x_real_grad


            _, phi_fake_adv = d(x_fake_adv)
            _, phi_real_adv = d(x_real_adv)

            reg_loss = config['reg'] * ((phi_fake_adv - phi_fake).norm() + (phi_real_adv - phi_real).norm())
            reg_loss.backward()

            metrics['reg_loss'] = reg_loss.item()

        d_optim.step()

        if i % 500 == 0:
            image_fake = torchvision.utils.make_grid(x_fake.detach().cpu(), normalize=True, range=(-1, 1))
            image_fake = np.array(torchvision.transforms.functional.to_pil_image(image_fake))

            image_real = torchvision.utils.make_grid(x_real.detach().cpu(), normalize=True, range=(-1, 1))
            image_real = np.array(torchvision.transforms.functional.to_pil_image(image_real))

            metrics['fake'] = [wandb.Image(image_fake)]
            metrics['real'] = [wandb.Image(image_real)]

        wandb.log(metrics, step=wandb.summary['step'])
        wandb.summary['step'] += 1


def infinite_fakes(g, config, device, batch_size=8):
    while True:
        z = torch.randn([batch_size, config['z_dim']]).to(device)
        x_fake = g(z)

        yield x_fake.detach().cpu()


def infinite_reals(data, batch_size=8):
    while True:
        for x, _ in data:
            yield x


def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)


def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = get_data(**config['data_args'])

    g = Generator(config['z_dim'], config['ngf']).to(device)
    g.apply(weights_init)
    g_optim = torch.optim.Adam(g.parameters(), lr=config['lr'], betas=(0.5, 0.999))

    d = Discriminator(ndf=config['ndf']).to(device)
    d.apply(weights_init)
    d_optim = torch.optim.Adam(d.parameters(), lr=config['lr'], betas=(0.5, 0.999))

    fake_generator = infinite_fakes(g, config, device)
    real_generator = infinite_reals(data)

    wandb.init(project='robust_discriminator', config=config)
    wandb.summary['step'] = 0

    fixed_noise = torch.randn(config['data_args']['batch_size'], config['z_dim']).to(device)

    for epoch in range(config['epochs']):
        train(g, g_optim, d, d_optim, data, config, device)

        x_fixed = g(fixed_noise)
        image_fixed = torchvision.utils.make_grid(x_fixed.detach().cpu(), normalize=True, range=(-1, 1))
        image_fixed = np.array(torchvision.transforms.functional.to_pil_image(image_fixed))

        fid = compute_fid_score(fake_generator, real_generator, device)

        if fid < wandb.summary.get('fid', np.inf):
            wandb.summary['best_fid'] = fid
            wandb.summary['best_epoch'] = epoch

        wandb.log({'fid': fid, 'fixed': [wandb.Image(image_fixed)]}, step=wandb.summary['step'])

        if epoch % 10 == 0:
            torch.save(g.state_dict(), pathlib.Path(wandb.run.dir) / 'generator_%03d.t7')
            torch.save(d.state_dict(), pathlib.Path(wandb.run.dir) / 'discriminator_%03d.t7')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data args.
    parser.add_argument('--dataset_dir', type=pathlib.Path, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--baseline', action='store_true', default=False)

    # Args.
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--z_dim', type=int, default=512)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--ngf', type=int, default=64)

    parser.add_argument('--eps', type=float, default=1e-4)
    parser.add_argument('--reg', type=float, default=1e1)

    args = parser.parse_args()

    config = {
            'baseline': args.baseline,

            'lr': args.lr,
            'epochs': args.epochs,

            'z_dim': args.z_dim,
            'ngf': args.ngf,
            'ndf': args.ndf,

            'reg': args.reg,
            'eps': args.eps,

            'data_args': {
                'dataset_dir': args.dataset_dir,
                'batch_size': args.batch_size,
                'num_workers': args.num_workers,
                },
            }

    main(config)
