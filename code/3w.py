import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torchvision.utils as vutils
from torch import autograd

import torch.nn.init as init

# Создание директорий и загрузка данных
os.makedirs('/data/CelebA', exist_ok=True)
os.makedirs('./checkpoints', exist_ok=True)
os.makedirs('./generated_images', exist_ok=True)

interpolate_mode = 'nearest'

transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = datasets.CelebA(root='/data/CelebA', split='train', transform=transform, download=False)
BATCH_SIZE = 64
data_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# Определение 2x блока
class ResBlock2x(nn.Module):
    def __init__(self):
        super(ResBlock2x, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.ConvTranspose2d(8, 8, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1)
        
        # Инициализация весов
        self._initialize_weights()

    def _initialize_weights(self):
        for param in self.parameters():
            if hasattr(param, 'data'):
                init.normal_(param.data, mean=0, std=0.01)

    def forward(self, x):
        residual = F.interpolate(x, scale_factor=2, mode=interpolate_mode)  # Масштабирование для соответствия размеров
        out = F.selu(self.conv1(x))
        out = F.selu(self.upsample(out))
        out = self.conv2(out)
        out += residual
        return out

# Определение генератора
class Generator(nn.Module):
    def __init__(self, input_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.SELU(True),
            nn.Linear(512, 1024),
            nn.SELU(True),
            nn.Linear(1024, 2048),
            nn.SELU(True),
            nn.Linear(2048, 3 * 8 * 8),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), 3, 8, 8)
        return x

# Определение дискриминатора
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1),

            nn.Flatten(),
            nn.Linear(4*4, 1)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for param in self.parameters():
            if hasattr(param, 'data'):
                init.normal_(param.data, mean=0, std=0.01)


    def forward(self, x):
        return self.model(x)

# Функция для вычисления градиентного штрафа
def gradient_penalty(real, fake, critic):
    m = real.shape[0]
    epsilon = torch.rand(m, 1, 1, 1).cuda()
    
    interpolated_img = epsilon * real + (1 - epsilon) * fake
    interpolated_out = critic(interpolated_img)

    grads = autograd.grad(outputs=interpolated_out, inputs=interpolated_img,
                          grad_outputs=torch.ones(interpolated_out.shape).cuda(),
                          create_graph=True, retain_graph=True)[0]
    
    grads = grads.reshape([m, -1])
    grad_penalty = ((grads.norm(2, dim=1) - 1) ** 2).mean()

    return grad_penalty

# Класс для всей модели
class FullModel(nn.Module):
    def __init__(self, latent_dim, num_resblocks):
        super(FullModel, self).__init__()
        self.generator = Generator(input_dim=latent_dim)
        self.resblocks = nn.ModuleList([ResBlock2x() for _ in range(num_resblocks)])

    def forward(self, x):
        x = self.generator(x)
        for resblock in self.resblocks:
            x = resblock(x)
        return x

    def train_model(self, data_loader, num_epochs, num_resblocks, n_critic, gradient_penalty_weight):
        total_iters = 0

        for block_index in range(num_resblocks + 1):
            # Сброс параметров дискриминатора в начале каждого блока
            discriminator._initialize_weights()
            optimizer_d = Adam(discriminator.parameters(), lr=0.0001, betas=(0.0, 0.9), weight_decay=2e-5)

            g_loss = torch.tensor(0)
            # Оптимизатор для текущего блока
            if block_index == 0:
                optimizer_g = Adam(self.generator.parameters(), lr=0.0001, betas=(0.0, 0.9), weight_decay=2e-5)
                current_block = self.generator
            else:
                optimizer_g = Adam(self.resblocks[block_index - 1].parameters(), lr=0.0001, betas=(0.0, 0.9), weight_decay=2e-5)
                current_block = self.resblocks[block_index - 1]

            for epoch in range(num_epochs):
                for i, (imgs, _) in enumerate(data_loader):
                    total_iters += 1
                    real_imgs = imgs.cuda()

                    # Обучение дискриминатора (критика)
                    optimizer_d.zero_grad()

                    z = torch.randn(BATCH_SIZE, latent_dim).cuda()
                    with torch.no_grad():
                        if block_index == 0:
                            fake_imgs = self.generator(z)
                        else:
                            fake_imgs = self.generator(z)
                            for j in range(block_index - 1):
                                fake_imgs = self.resblocks[j](fake_imgs)
                    
                    fake_imgs = F.interpolate(fake_imgs, size=128, mode=interpolate_mode)

                    real_imgs = imgs.cuda().requires_grad_(True)  # Добавляем requires_grad=True
                    fake_imgs = fake_imgs.requires_grad_(True)   # Добавляем requires_grad=True

                    fake_out = discriminator(fake_imgs.detach())
                    real_out = discriminator(real_imgs.detach())
                    d_loss = (real_out.mean() - fake_out.mean()) + gradient_penalty(real_imgs, fake_imgs, discriminator) * gradient_penalty_weight

                    d_loss.backward()
                    optimizer_d.step()

                    # Обучение генератора или текущего блока
                    if total_iters % n_critic == 0 or i == 0:
                        optimizer_g.zero_grad()

                        z = torch.randn(BATCH_SIZE, latent_dim).cuda()
                        if block_index == 0:
                            gen_imgs = self.generator(z)
                        else:
                            gen_imgs = self.generator(z)
                            for j in range(block_index):
                                gen_imgs = self.resblocks[j](gen_imgs)
                        gen_imgs = F.interpolate(gen_imgs, size=128, mode=interpolate_mode)
                        g_loss = discriminator(gen_imgs).mean()

                        if g_loss.item() > 0:
                            g_loss.backward()
                            optimizer_g.step()

                    # Сохранение промежуточных результатов каждые 100 итераций
                    if i % 100 == 0:
                        print(f"Block [{block_index}/{num_resblocks}], Epoch [{epoch}/{num_epochs}], Step [{i}/{len(data_loader)}]  Loss D: {d_loss.item()}, Loss G: {g_loss.item()}")
                        vutils.save_image(gen_imgs.data[:25], f'./generated_images/fake_images-{block_index}-{epoch}-{i}.png', nrow=5, normalize=True)
                        torch.save(self.state_dict(), f'./checkpoints/model-{block_index}-{epoch}-{i}.pth')

# Параметры
latent_dim = 100  # Размерность входного вектора для генератора
num_resblocks = 3  # Количество блоков ResBlock2x
num_epochs = 1  # Количество эпох для каждого блока
n_critic = 3  # Количество итераций обучения дискриминатора на одну итерацию генератора
gradient_penalty_weight = 10

# Инициализация моделей
generator = Generator(input_dim=latent_dim).cuda()
discriminator = Discriminator().cuda()
full_model = FullModel(latent_dim, num_resblocks).cuda()

# Запуск обучения
full_model.train_model(data_loader, num_epochs, num_resblocks, n_critic, gradient_penalty_weight)
