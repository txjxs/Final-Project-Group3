import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

from VAE_data import LABColorizeDataset, lab_to_rgb_tensor
from VAE_model import CVAE
from VAE_loss import LABCVAELoss


def train_epoch(model, dataloader, loss_fn, optimizer, device, beta_annealing=1.0):
    model.train()
    
    total_loss_sum = 0
    color_loss_sum = 0
    kl_loss_sum = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        L_noisy = batch['input'].to(device)  # (B, 1, H, W)
        ab_target = batch['target'].to(device)  # (B, 2, H, W)

        ab_pred, mu, logvar = model(L_noisy)

        total_loss, color_loss, kl_loss = loss_fn(ab_pred, ab_target, mu, logvar)

        if beta_annealing < 1.0:
            total_loss = color_loss + (beta_annealing * loss_fn.beta) * kl_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss_sum += total_loss.item()
        color_loss_sum += color_loss.item()
        kl_loss_sum += kl_loss.item()
        num_batches += 1

        pbar.set_postfix({
            'loss': f'{total_loss.item():.4f}',
            'color': f'{color_loss.item():.4f}',
            'kl': f'{kl_loss.item():.4f}'
        })
    
    return (total_loss_sum / num_batches,
            color_loss_sum / num_batches,
            kl_loss_sum / num_batches)


def validate(model, dataloader, loss_fn, device):
    """Validate the model"""
    model.eval()
    
    total_loss_sum = 0
    color_loss_sum = 0
    kl_loss_sum = 0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for batch in pbar:
            L_noisy = batch['input'].to(device)
            ab_target = batch['target'].to(device)
            
            # Forward pass
            ab_pred, mu, logvar = model(L_noisy)
            
            # Compute loss
            total_loss, color_loss, kl_loss = loss_fn(ab_pred, ab_target, mu, logvar)
            
            # Accumulate
            total_loss_sum += total_loss.item()
            color_loss_sum += color_loss.item()
            kl_loss_sum += kl_loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'color': f'{color_loss.item():.4f}',
                'kl': f'{kl_loss.item():.4f}'
            })
    
    return (total_loss_sum / num_batches,
            color_loss_sum / num_batches,
            kl_loss_sum / num_batches)


def plot_training_curves(train_history, val_history, save_path):
    """Plot and save training curves"""
    epochs = range(1, len(train_history['total']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('L*a*b* Training with Chromatic Weighting', fontsize=16)
    
    # Total loss
    axes[0, 0].plot(epochs, train_history['total'], 'b-', label='Train')
    axes[0, 0].plot(epochs, val_history['total'], 'r-', label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Color loss
    axes[0, 1].plot(epochs, train_history['color'], 'b-', label='Train')
    axes[0, 1].plot(epochs, val_history['color'], 'r-', label='Validation')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Color Loss (Chromatic-weighted MSE)')
    axes[0, 1].set_title('Color Loss on a*b* channels')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # KL loss
    axes[1, 0].plot(epochs, train_history['kl'], 'b-', label='Train')
    axes[1, 0].plot(epochs, val_history['kl'], 'r-', label='Validation')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('KL Divergence')
    axes[1, 0].set_title('KL Divergence')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Color loss (zoomed)
    axes[1, 1].plot(epochs, train_history['color'], 'b-', label='Train')
    axes[1, 1].plot(epochs, val_history['color'], 'r-', label='Validation')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Color Loss')
    axes[1, 1].set_title('Color Loss (Zoomed)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, max(train_history['color'][:5])])  # Zoom to first 5 epochs range
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_results(model, dataloader, device, save_path, num_samples=4):
    """Visualize L*a*b* colorization results"""
    model.eval()
    
    # Get a batch
    batch = next(iter(dataloader))
    L_noisy = batch['input'][:num_samples].to(device)
    ab_target = batch['target'][:num_samples].to(device)
    L_clean = batch['L_clean'][:num_samples].to(device)

    with torch.no_grad():
        ab_pred, _, _ = model(L_noisy)

    rgb_pred = lab_to_rgb_tensor(L_clean, ab_pred)
    rgb_target = lab_to_rgb_tensor(L_clean, ab_target)

    L_noisy = L_noisy.cpu()
    rgb_pred = rgb_pred.cpu()
    rgb_target = rgb_target.cpu()

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    for i in range(num_samples):
        # Input L*
        axes[i, 0].imshow(L_noisy[i, 0], cmap='gray')
        axes[i, 0].set_title('Input: Noisy L* (Lightness)')
        axes[i, 0].axis('off')

        output_img = rgb_pred[i].permute(1, 2, 0).clamp(0, 1).numpy()
        axes[i, 1].imshow(output_img)
        axes[i, 1].set_title('Output: Predicted Colors')
        axes[i, 1].axis('off')

        target_img = rgb_target[i].permute(1, 2, 0).clamp(0, 1).numpy()
        axes[i, 2].imshow(target_img)
        axes[i, 2].set_title('Target: Ground Truth')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():

    config = {
        # Data
        'train_dir': '../data/train2017',
        'val_dir': '../data/val2017',
        'batch_size': 32,
        'image_size': 128,
        'noise_std': 0.1,
        'num_workers': 4,

        'in_channels': 1,
        'out_channels': 2,
        'latent_dim': 128,

        'chromatic_weight': 5.0,
        'beta': 0.001,
        'use_kl_annealing': True,
        'warmup_epochs': 10,

        'num_epochs': 50,
        'learning_rate': 0.0001,

        'checkpoint_dir': 'checkpoints_lab',
        'output_dir': 'outputs_lab',
        'save_every': 5,
        'plot_every': 5,
    }

    Path(config['checkpoint_dir']).mkdir(exist_ok=True)
    Path(config['output_dir']).mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\n" + "=" * 70)
    print("Loading Data")
    print("=" * 70)
    
    train_dataset = LABColorizeDataset(
        config['train_dir'],
        image_size=config['image_size'],
        noise_std=config['noise_std']
    )
    
    val_dataset = LABColorizeDataset(
        config['val_dir'],
        image_size=config['image_size'],
        noise_std=config['noise_std']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    print("\n" + "=" * 70)
    print("Creating Model")
    print("=" * 70)
    
    model = CVAE(
        in_channels=config['in_channels'],
        out_channels=config['out_channels'],
        latent_dim=config['latent_dim']
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    print("\n" + "=" * 70)
    print("Loss Function: L*a*b* Chromatic-Weighted MSE + KL")
    print("=" * 70)
    
    loss_fn = LABCVAELoss(
        chromatic_weight=config['chromatic_weight'],
        beta=config['beta']
    ).to(device)
    
    print(f"Chromatic weight: {config['chromatic_weight']}")
    print(f"Beta (KL weight): {config['beta']}")
    print(f"KL annealing: {config['use_kl_annealing']}")
    print(f"Warmup epochs: {config['warmup_epochs']}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)
    
    train_history = {'total': [], 'color': [], 'kl': []}
    val_history = {'total': [], 'color': [], 'kl': []}
    
    best_val_loss = float('inf')
    
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['num_epochs']}")
        print("-" * 70)

        if config['use_kl_annealing'] and epoch <= config['warmup_epochs']:
            beta_annealing = epoch / config['warmup_epochs']
        else:
            beta_annealing = 1.0
        
        print(f"Current beta: {beta_annealing * config['beta']:.6f}")
        
        # Train
        train_total, train_color, train_kl = train_epoch(
            model, train_loader, loss_fn, optimizer, device, beta_annealing
        )

        val_total, val_color, val_kl = validate(
            model, val_loader, loss_fn, device
        )

        train_history['total'].append(train_total)
        train_history['color'].append(train_color)
        train_history['kl'].append(train_kl)
        
        val_history['total'].append(val_total)
        val_history['color'].append(val_color)
        val_history['kl'].append(val_kl)

        print(f"Epoch {epoch} Summary:")
        print(f"  Train - Total: {train_total:.6f}, Color: {train_color:.6f}, KL: {train_kl:.6f}")
        print(f"  Val   - Total: {val_total:.6f}, Color: {val_color:.6f}, KL: {val_kl:.6f}")

        if val_total < best_val_loss:
            best_val_loss = val_total
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_total,
                'config': config
            }
            torch.save(checkpoint, f"{config['checkpoint_dir']}/best_model_lab.pth")
            print(f"Best model saved (val_loss: {val_total:.6f})")

        if epoch % config['save_every'] == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_total,
                'config': config
            }
            torch.save(checkpoint, f"{config['checkpoint_dir']}/checkpoint_epoch_{epoch}.pth")
            print(f"Checkpoint saved: epoch {epoch}")
        
        # Plot curves
        if epoch % config['plot_every'] == 0 or epoch == 1:
            plot_path = f"{config['output_dir']}/curves_epoch_{epoch}.png"
            plot_training_curves(train_history, val_history, plot_path)
            print(f"Training curves saved: {plot_path}")
            
            # Visualize results
            vis_path = f"{config['output_dir']}/results_epoch_{epoch}.png"
            visualize_results(model, val_loader, device, vis_path)
            print(f"Results saved: {vis_path}")
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved in: {config['checkpoint_dir']}")
    print(f"Results saved in: {config['output_dir']}")

if __name__ == "__main__":
    main()
