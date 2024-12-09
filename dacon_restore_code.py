import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import zipfile

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Define Generator
class InpaintingGenerator(nn.Module):
    def __init__(self):
        super(InpaintingGenerator, self).__init__()
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.enc5 = self.conv_block(512, 1024)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = self.conv_block(1024 + 512, 512)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = self.conv_block(512 + 256, 256)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = self.conv_block(256 + 128, 128)

        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4 = self.conv_block(128 + 64, 64)

        self.final = nn.Conv2d(64, 3, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(nn.MaxPool2d(2)(e1))
        e3 = self.enc3(nn.MaxPool2d(2)(e2))
        e4 = self.enc4(nn.MaxPool2d(2)(e3))
        e5 = self.enc5(nn.MaxPool2d(2)(e4))

        d1 = self.dec1(torch.cat([self.up1(e5), e4], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d2), e2], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d3), e1], dim=1))

        return torch.sigmoid(self.final(d4))


# Define Discriminator
class LSGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(LSGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)


# Define Dataset
class ImageDataset(Dataset):
    def __init__(self, input_dir, gt_dir, input_images, gt_images, transform=None, limit=None):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.input_images = input_images[:limit] if limit else input_images  # limit 적용
        self.gt_images = gt_images[:limit] if limit else gt_images          # limit 적용
        self.transform = transform

    def __len__(self):
        return len(self.input_images)  # 길이는 limit에 따라 조정됨

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.input_images[idx])
        gt_path = os.path.join(self.gt_dir, self.gt_images[idx])
        input_image = cv2.imread(input_path)
        gt_image = cv2.imread(gt_path)
        if self.transform:
            input_image = self.transform(input_image)
            gt_image = self.transform(gt_image)
        return (
            torch.tensor(input_image).permute(2, 0, 1).float() / 255.0,
            torch.tensor(gt_image).permute(2, 0, 1).float() / 255.0
        )



# SSIM Loss
class SSIMLoss(nn.Module):
    def __init__(self, win_size=3):
        super(SSIMLoss, self).__init__()
        self.win_size = win_size

    def forward(self, img1, img2):
        img1_np = img1.detach().permute(0, 2, 3, 1).cpu().numpy()
        img2_np = img2.detach().permute(0, 2, 3, 1).cpu().numpy()

        ssim_scores = []
        for i in range(img1_np.shape[0]):
            ssim_score = ssim(
                img1_np[i],
                img2_np[i],
                multichannel=True,
                data_range=1.0,
                win_size=self.win_size
            )
            ssim_scores.append(ssim_score)

        ssim_mean = np.mean(ssim_scores)
        return 1 - torch.tensor(ssim_mean, device=img1.device)

def gradient_penalty(discriminator, real_data, fake_data, device):
    batch_size, channels, height, width = real_data.size()
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    interpolated = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True).to(device)

    interpolated_output = discriminator(interpolated)
    grad_outputs = torch.ones_like(interpolated_output).to(device)

    gradients = torch.autograd.grad(
        outputs=interpolated_output,
        inputs=interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty

# Load file lists
input_images = sorted(os.listdir("./train_input"))
gt_images = sorted(os.listdir("./train_gt"))

# Train/Validation Split
train_inputs, val_inputs, train_gts, val_gts = train_test_split(input_images, gt_images, test_size=0.2, random_state=42)

# Create Datasets and DataLoaders
train_dataset = ImageDataset("./train_input", "./train_gt", train_inputs, train_gts, limit = None)
val_dataset = ImageDataset("./train_input", "./train_gt", val_inputs, val_gts, limit = None)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

# Initialize Models, Loss, Optimizers
generator = InpaintingGenerator().to(device)
discriminator = LSGANDiscriminator().to(device)

adversarial_loss = nn.MSELoss()
ssim_loss = SSIMLoss()

optimizer_G = optim.AdamW(generator.parameters(), lr=0.0001, betas=(0.5, 0.9))
optimizer_D = optim.AdamW(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

lambda_gp = 10  # Gradient Penalty 가중치
# StepLR 설정
scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=10, gamma=0.5)
scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=15, gamma=0.5)

# Training Loop
epochs = 50
result_dir = "result2"
os.makedirs(result_dir, exist_ok=True)
checkpoint_path = "checkpoint.pth"
test_input_dir = "./test_input"
for epoch in range(epochs):
    generator.train()
    discriminator.train()
    running_loss_G = 0.0
    running_loss_D = 0.0

    # Training Phase
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
        for input_images, gt_images in train_loader:
            input_images, gt_images = input_images.to(device), gt_images.to(device)

            real_labels = torch.ones_like(discriminator(gt_images)).to(device)
            fake_labels = torch.zeros_like(discriminator(input_images)).to(device)

            # Generator Loss
            optimizer_G.zero_grad()
            fake_images = generator(input_images)
            pred_fake = discriminator(fake_images)

            g_loss_adv = adversarial_loss(pred_fake, real_labels)
            g_loss_pixel = nn.MSELoss()(fake_images, gt_images)
            g_loss_ssim = ssim_loss(fake_images, gt_images)
            g_loss = g_loss_adv + 50 * g_loss_pixel + 50 * g_loss_ssim
            g_loss.backward()
            optimizer_G.step()

            # Discriminator Loss
            optimizer_D.zero_grad()
            pred_real = discriminator(gt_images)
            loss_real = adversarial_loss(pred_real, real_labels)

            pred_fake = discriminator(fake_images.detach())
            loss_fake = adversarial_loss(pred_fake, fake_labels)
            gp = gradient_penalty(discriminator, gt_images, fake_images.detach(), device)
            d_loss = loss_real + loss_fake + lambda_gp * gp
            d_loss.backward()
            optimizer_D.step()

            running_loss_G += g_loss.item()
            running_loss_D += d_loss.item()

            pbar.set_postfix(generator_loss=g_loss.item(), discriminator_loss=d_loss.item())
            pbar.update(1)

    print(f"Epoch [{epoch+1}/{epochs}] - Generator Loss: {running_loss_G / len(train_loader):.4f}, Discriminator Loss: {running_loss_D / len(train_loader):.4f}")

    # Validation Phase
    generator.eval()
    val_loss_G = 0.0
    with torch.no_grad():
        for input_images, gt_images in val_loader:
            input_images, gt_images = input_images.to(device), gt_images.to(device)
            fake_images = generator(input_images)

            g_loss_pixel = nn.MSELoss()(fake_images, gt_images)
            g_loss_ssim = ssim_loss(fake_images, gt_images)
            g_loss = 50 * g_loss_pixel + 50 * g_loss_ssim
            val_loss_G += g_loss.item()

    val_loss_G /= len(val_loader)
    print(f"Epoch [{epoch+1}/{epochs}] - Validation Generator Loss: {val_loss_G:.4f}")
    
    # StepLR 업데이트
    scheduler_G.step()
    scheduler_D.step()

    print(f"Epoch {epoch+1}/{epochs} - Learning Rate G: {scheduler_G.get_last_lr()[0]:.6f}, "
          f"Learning Rate D: {scheduler_D.get_last_lr()[0]:.6f}")

    output_dir = f"output_images_epoch_{epoch+1}"
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for img_name in sorted(os.listdir(test_input_dir)):
            img_path = os.path.join(test_input_dir, img_name)
            img = cv2.imread(img_path)
            input_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            output = generator(input_tensor).squeeze().permute(1, 2, 0).cpu().numpy() * 255.0
            output = output.astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, img_name), output)

    zip_filename = os.path.join(result_dir, f"epoch_{epoch+1}.zip")
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for img_name in os.listdir(output_dir):
            zipf.write(os.path.join(output_dir, img_name), arcname=img_name)
    print(f"Epoch {epoch+1} results saved to {zip_filename}")

    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict()
    }, checkpoint_path)
