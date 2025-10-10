import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
from tqdm import tqdm
from torchvision import transforms
import torch.nn.functional as F
import sys, os
from dataset import SuperResolutionDataset
from Simple_CNN import SimpleCNN
from loss import image_compare_loss
from TV_activation import TVLeakyReLU
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  
sys.path.insert(0, str(ROOT))
import core.logger as Logger




# trian function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    first= True
    for inputs, targets,*_ in tqdm(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        if first:
                    # snapshot params to verify they change
                    old_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()

        if first:
            total_gnorm = 0.0
            n_params = 0
            for p in model.parameters():
                if p.grad is not None:
                    total_gnorm += p.grad.norm().item()
                    n_params += 1
            print(f"[DEBUG] grad-norm: {total_gnorm:.6f} over {n_params} params")
        optimizer.step()
        train_loss += loss.item()
        if first:
            delta = 0.0
            for k, v in model.state_dict().items():
                delta += (v.detach() - old_state[k]).abs().sum().item()
            print(f"[DEBUG] total-param-change: {delta:.6f}")
            first = False

    return train_loss / len(train_loader)


# evaluate function
def calculate_psnr_batch(pred_batch, target_batch, max_val=1.0):
    """Calculate the mean PSNR of all images within a batch"""
    batch_size = pred_batch.size(0)
    mse = torch.mean(torch.pow(pred_batch - target_batch, 2), dim=[1, 2, 3])
    psnr = torch.zeros(batch_size, dtype=torch.float32, device=pred_batch.device)
    psnr[mse > 0] = 10 * torch.log10((max_val ** 2) / mse[mse > 0])
    avg_psnr = torch.mean(psnr)
    return avg_psnr.item()


def evaluate(model, dataloader, device):
    model.eval()
    total_psnr = 0.0
    with torch.no_grad():
        for inputs, targets, *_ in tqdm(dataloader, desc='Test', leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            total_psnr += calculate_psnr_batch(outputs, targets)
    avg_psnr = total_psnr / len(dataloader)
    return avg_psnr

# Save the CNN prediction results
def save_res(model, dataloader, device,lr,hr):
    model.eval()
    with torch.no_grad():
        for  inputs, targets , path , _ in tqdm(dataloader, desc='Test', leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            image = ((outputs[0] + 1) / 2)
            out_path = path[0].replace(f'lr_{lr}',f'cnn_sr_{lr}_{hr}')
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            save_image(image, out_path)


def main():
    # ----------  JSON config ----------
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', dest='config', type=str, required=True)
    args = parser.parse_args()

    def require(cfg: dict, key: str):
        if key not in cfg:
            raise KeyError(f'Missing required key in pretrain config: "{key}"')
        return cfg[key]
    
    args_for_logger = argparse.Namespace(
    config=args.config,
    phase=None,
    gpu_ids=None,
    debug=False,
    enable_wandb=False,
    log_wandb_ckpt=False,
    log_eval=False,
    )

    opt = Logger.parse(args_for_logger)
    opt = Logger.dict_to_nonedict(opt)

    pre = opt.get('pretrain', {}) or {}
    if not isinstance(pre, dict) or not pre:
        raise KeyError('Missing "pretrain" block in config JSON.')  

      
    lr               = int(require(pre, 'lr'))
    hr               = int(require(pre, 'hr'))
    hr_dir           = require(pre, 'hr_dir')
    lr_dir           = require(pre, 'lr_dir')
    batch_size       = int(require(pre, 'batch_size'))
    batch_size_eval  = int(require(pre, 'batch_size_eval'))
    learning_rate    = float(require(pre, 'learning_rate'))
    epochs           = int(require(pre, 'epochs'))
    num_workers      = int(require(pre, 'num_workers'))
    ckpt_path = pre.get('ckpt_path')          # may be None
    if ckpt_path in (None, "", "null", "None"):
        ckpt_path = "./pretrain_CNN/cnn_weights.pth"                      # normalize all “empty” values
    if hr % lr != 0:
        raise ValueError(f'Invalid sizes: hr ({hr}) must be an integer multiple of lr ({lr}).')
    scale_factor = hr // lr

    if not os.path.isdir(hr_dir):
        raise FileNotFoundError(f'hr_dir not found: {hr_dir}')
    if not os.path.isdir(lr_dir):
        raise FileNotFoundError(f'lr_dir not found: {lr_dir}')

    print('[Pretrain Config Loaded]')
    print(f'  lr={lr}, hr={hr}, scale_factor={scale_factor}')
    print(f'  hr_dir={hr_dir}')
    print(f'  lr_dir={lr_dir}')
    print(f'  batch_size={batch_size}, batch_size_eval={batch_size_eval}, epochs={epochs}')
    print(f'  learning_rate={learning_rate}, num_workers={num_workers}')
    print(f'  ckpt_path={ckpt_path}')
    print('------------------------------------------------------')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   
    
    # ---------- transforms (needed for dataset) ----------
    transform = transforms.Compose([transforms.ToTensor()])


    # Create dataset
    train_dataset = SuperResolutionDataset(hr_dir, lr_dir, transform)
    test_dataset=SuperResolutionDataset(hr_dir, lr_dir, transform,train=False)
    eval_idx = np.arange(batch_size_eval)
    eval_dataset = Subset(test_dataset, eval_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size_eval, shuffle=False,num_workers = num_workers)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,num_workers = num_workers)

    # Create models, loss functions and optimizers
    model = SimpleCNN(scale_factor=scale_factor).to(device)
   
    print("Using device:", device)
    print("Model device:", next(model.parameters()).device)

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path))
        print(f"Loaded weights from {ckpt_path}")
    else:
        print("No pretrained model found — training from scratch")
   

    criterion = image_compare_loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training and evaluation models, and save the model weights along with prediction images
    for epoch in range(epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_psnr = evaluate(model, eval_loader, device)
        print('Epoch [{}/{}], Train Loss: {:.4f}, '
              'Test PSNR: {:.4f}'.format(epoch + 1, epochs, train_loss, test_psnr))

        torch.save(model.state_dict(),ckpt_path)

    save_res(model, test_loader, device,lr,hr)


if __name__ == '__main__':
    main()
