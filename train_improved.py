import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
import os
import time
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# --- Use the SAME model architecture as your original training ---
class HybridModel(nn.Module):
    def __init__(self, num_classes=2, d_model=1280, nhead=8, num_encoder_layers=3, dim_feedforward=1024, dropout=0.1):
        super(HybridModel, self).__init__()
        
        # KEEP EfficientNet-B0 to match your original model
        self.cnn_backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1).features
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
            dropout=dropout, batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(d_model, num_classes))

    def forward(self, x):
        x = self.cnn_backbone(x)
        b, c, h, w = x.shape
        x = x.flatten(2).permute(2, 0, 1)
        x = self.transformer_encoder(x)
        x = x.permute(1, 2, 0)
        x = self.global_pool(x).squeeze(-1)
        return self.classifier(x)

# --- Focal Loss for Class Imbalance ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Handle both tensor and list formats for alpha
        if self.alpha is not None:
            if isinstance(self.alpha, (list, np.ndarray)):
                alpha = torch.tensor(self.alpha, device=inputs.device)
            else:
                alpha = self.alpha
        else:
            alpha = 1.0
            
        BCE_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class ImprovedTrainer:
    def __init__(self, data_dir="celeb-df"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_dir = data_dir
        
        # Class weights based on your dataset statistics
        self.class_weights = [3.198, 1.0]  # Real: 3.198x weight, Fake: 1.0x weight
        
        self.setup_data()
        self.setup_model()
    
    def setup_data(self):
        """Setup data with class balancing - FIXED batch size handling"""
        print("Setting up improved data pipeline...")
        
        # Enhanced transforms
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2)
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Load datasets
        train_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'train'), self.train_transform)
        val_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'val'), self.val_transform)
        
        # FIXED: Proper weight calculation
        class_counts = [10901, 34865]  # Your actual counts
        weights = [1.0 / count for count in class_counts]
        sample_weights = [weights[label] for label in train_dataset.targets]
        
        # Weighted Random Sampler
        self.sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        
        # FIXED: Consistent batch sizes
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=16,  # Use same batch size throughout
            sampler=self.sampler,
            num_workers=0,
            pin_memory=True,
            drop_last=True  # FIX: Drop incomplete batches to avoid size mismatches
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=16,  # Same as train batch size
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=True  # FIX: Drop incomplete batches
        )
        
        self.dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
        self.class_names = train_dataset.classes
        
        print(f"Classes: {self.class_names}")
        print(f"Training samples: {self.dataset_sizes['train']}")
        print(f"Validation samples: {self.dataset_sizes['val']}")
        print(f"Class weights: {self.class_weights}")
        print(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")
    
    def setup_model(self):
        """Setup model with improved configuration"""
        # Use the SAME architecture as your original model
        self.model = HybridModel(num_classes=len(self.class_names))
        
        # Try to load existing weights to continue training
        model_loaded = False
        if os.path.exists('best_hybrid_model.pth'):
            try:
                print("Loading existing model weights to continue training...")
                checkpoint = torch.load('best_hybrid_model.pth', map_location=self.device)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                    
                print("✓ Pre-trained weights loaded")
                model_loaded = True
            except Exception as e:
                print(f"✗ Could not load existing model: {e}")
                print("Starting with fresh weights...")
        
        self.model = self.model.to(self.device)
        
        # FIXED: Simplified Focal Loss without tensor conversion issues
        self.criterion = FocalLoss(alpha=self.class_weights, gamma=2.0)
        
        # Improved optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=0.00005,  # Lower learning rate for fine-tuning
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing with warm restarts
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=5,
            T_mult=2,
            eta_min=1e-7
        )
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Model {'loaded' if model_loaded else 'initialized'} successfully")
    
    def train_epoch(self, epoch):
        """FIXED: Training epoch with proper batch handling"""
        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        processed_samples = 0
        
        # FIXED: Simplified gradient accumulation
        accumulation_steps = 2
        self.optimizer.zero_grad()
        
        for i, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # FIXED: Ensure consistent batch size
            batch_size = inputs.size(0)
            if batch_size != 16:  # Skip if batch size doesn't match expected
                print(f"Skipping batch with size {batch_size}")
                continue
                
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss = loss / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * accumulation_steps * batch_size
            running_corrects += torch.sum(preds == labels.data)
            processed_samples += batch_size
            
            if i % 50 == 0:
                current_loss = loss.item() * accumulation_steps
                current_acc = torch.sum(preds == labels.data).item() / batch_size
                print(f'Epoch {epoch}, Batch {i}, Loss: {current_loss:.4f}, Acc: {current_acc:.4f}')
        
        # Handle remaining batches
        if len(self.train_loader) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        epoch_loss = running_loss / processed_samples if processed_samples > 0 else 0
        epoch_acc = running_corrects.double() / processed_samples if processed_samples > 0 else 0
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self):
        """FIXED: Validation epoch with proper batch handling"""
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        processed_samples = 0
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # FIXED: Ensure consistent batch size
                batch_size = inputs.size(0)
                if batch_size != 16:
                    continue
                    
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * batch_size
                running_corrects += torch.sum(preds == labels.data)
                processed_samples += batch_size
        
        epoch_loss = running_loss / processed_samples if processed_samples > 0 else 0
        epoch_acc = running_corrects.double() / processed_samples if processed_samples > 0 else 0
        
        return epoch_loss, epoch_acc
    
    def train(self, num_epochs=15):  # Reduced epochs for stability
        """Main training loop"""
        since = time.time()
        best_acc = 0.0
        patience = 5
        patience_counter = 0
        
        print("Starting improved training...")
        print("Focus: Improving real image classification (addressing class imbalance)")
        
        # Track progress
        train_losses = []
        val_accuracies = []
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            print('-' * 10)
            
            # Training phase
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validation phase
            val_loss, val_acc = self.validate_epoch()
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Store metrics
            train_losses.append(train_loss)
            val_accuracies.append(val_acc.cpu().numpy() if hasattr(val_acc, 'cpu') else val_acc)
            
            print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
            print(f'LR: {self.optimizer.param_groups[0]["lr"]:.8f}')
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), 'best_improved_model.pth')
                print(f'✓ New best model saved! Acc: {best_acc:.4f}')
                patience_counter = 0
            else:
                patience_counter += 1
                print(f'No improvement for {patience_counter} epochs')
            
            # Early stopping
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        time_elapsed = time.time() - since
        print(f'\nTraining complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
        print(f'Best val Acc: {best_acc:.4f}')
        
        # Load best model
        if os.path.exists('best_improved_model.pth'):
            self.model.load_state_dict(torch.load('best_improved_model.pth'))
        
        return self.model, train_losses, val_accuracies

def main():
    print("🚀 Improved Deepfake Detection Training")
    print("=" * 50)
    print("Focus: Addressing class imbalance while maintaining architecture compatibility")
    print("Using EfficientNet-B0 (same as original) with improved training techniques")
    
    try:
        trainer = ImprovedTrainer(data_dir="celeb-df")
        model, train_losses, val_accuracies = trainer.train(num_epochs=15)
        
        print("\n✅ Training completed!")
        print("Model saved as 'best_improved_model.pth'")
        
        # Plot training history
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(train_losses)
            plt.title('Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            
            plt.subplot(1, 2, 2)
            plt.plot(val_accuracies)
            plt.title('Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            
            plt.tight_layout()
            plt.savefig('improved_training_history.png')
            print("Training history plot saved as 'improved_training_history.png'")
            
        except ImportError:
            print("Matplotlib not available, skipping plot generation")
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()