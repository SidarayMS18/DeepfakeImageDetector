import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
import time
import numpy as np

# --- 1. Define the Hybrid CNN-Transformer Model ---

class HybridModel(nn.Module):
    def __init__(self, num_classes=2, d_model=1280, nhead=8, num_encoder_layers=3, dim_feedforward=1024, dropout=0.1):
        """
        Args:
            num_classes (int): Number of output classes (2 for real/fake).
            d_model (int): The embedding dimension required by the Transformer.
                           Must match the CNN backbone's output channels.
                           EfficientNet-B0 outputs 1280 channels.
            nhead (int): Number of attention heads in the Transformer.
            num_encoder_layers (int): Number of Transformer encoder layers.
            dim_feedforward (int): Hidden dimension of the Transformer's feedforward network.
            dropout (float): Dropout rate.
        """
        super(HybridModel, self).__init__()
        
        # --- CNN Backbone ---
        # Load a pretrained EfficientNet-B0
        self.cnn_backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1).features
        
        # --- Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # --- Classification Head ---
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        # 1. Pass through CNN backbone
        x = self.cnn_backbone(x)
        
        # 2. Prepare for Transformer
        b, c, h, w = x.shape
        x = x.flatten(2)
        x = x.permute(2, 0, 1)
        
        # 3. Pass through Transformer Encoder
        x = self.transformer_encoder(x)
        
        # 4. Global Pooling
        x = x.permute(1, 2, 0)
        x = self.global_pool(x)
        x = x.squeeze(-1)

        # 5. Pass through Classifier
        x = self.classifier(x)
        return x

# --- 2. Configuration and Setup ---

DATA_DIR = "celeb-df"

# Reduced parameters for faster testing
NUM_EPOCHS = 10
BATCH_SIZE = 16  # Reduced batch size
LEARNING_RATE = 0.0001
IMG_SIZE = 224

# --- 3. Setup CUDA Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if not torch.cuda.is_available():
    print("WARNING: CUDA not available. Training will be slow on CPU.")

# --- 4. Define Data Transforms and Load Datasets ---

# Simplified transforms for faster loading
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
data_transforms['test'] = data_transforms['val']

print("Loading datasets...")
print(f"Looking for dataset in: {os.path.abspath(DATA_DIR)}")

# Check if dataset exists
if not os.path.exists(DATA_DIR):
    print(f"ERROR: Dataset directory '{DATA_DIR}' not found!")
    print("Available files and folders in current directory:")
    for item in os.listdir('.'):
        if os.path.isdir(item):
            print(f"  [DIR] {item}")
        else:
            print(f"  [FILE] {item}")
    exit()

# Create datasets
try:
    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), data_transforms['train']),
        'val': datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), data_transforms['val']),
        'test': datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), data_transforms['test'])
    }
    
    # Create DataLoaders with num_workers=0 for Windows compatibility
    dataloaders = {
        x: DataLoader(
            image_datasets[x], 
            batch_size=BATCH_SIZE, 
            shuffle=(x == 'train'), 
            num_workers=0,  # Must be 0 on Windows
            pin_memory=True if torch.cuda.is_available() else False
        )
        for x in ['train', 'val', 'test']
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes

    print(f"Classes: {class_names}")
    print(f"Training data size: {dataset_sizes['train']}")
    print(f"Validation data size: {dataset_sizes['val']}")
    print(f"Test data size: {dataset_sizes['test']}")
    
    # Test data loading with a small batch
    print("\nTesting data loading...")
    test_batch = next(iter(dataloaders['train']))
    print(f"Test batch - Input shape: {test_batch[0].shape}, Label shape: {test_batch[1].shape}")
    print("Data loading test successful!")
    
except Exception as e:
    print(f"Error loading datasets: {e}")
    import traceback
    traceback.print_exc()
    exit()

# --- 5. Initialize Model, Loss, and Optimizer ---

print("\nInitializing model...")
model = HybridModel(num_classes=len(class_names))
model = model.to(device)

# Print model summary
print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# --- 6. Training and Validation Loop ---

def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()
    best_acc = 0.0
    train_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            batch_count = 0

            # Progress tracking
            total_batches = len(dataloaders[phase])
            start_time = time.time()

            # Iterate over data.
            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                # Move inputs and labels to the device
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                batch_count += 1

                # Print progress every 10 batches
                if batch_idx % 10 == 0:
                    batch_time = time.time() - start_time
                    print(f'  {phase}: Batch {batch_idx}/{total_batches} - Loss: {loss.item():.4f} - Time: {batch_time:.2f}s')
                    start_time = time.time()

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                train_losses.append(epoch_loss)
            else:
                val_accuracies.append(epoch_acc.cpu().numpy())

            # Save best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'best_hybrid_model.pth')
                print(f'New best model saved with accuracy: {best_acc:.4f}')

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    if os.path.exists('best_hybrid_model.pth'):
        model.load_state_dict(torch.load('best_hybrid_model.pth'))
    
    return model, train_losses, val_accuracies


# --- 7. Run Training ---
if __name__ == "__main__":
    print("Starting training...")
    print("If this gets stuck, press Ctrl+C to interrupt and check the dataset.")
    
    try:
        # Test forward pass with a single batch first
        print("\nTesting forward pass with one batch...")
        model.eval()
        test_inputs, test_labels = next(iter(dataloaders['train']))
        test_inputs = test_inputs.to(device)
        test_labels = test_labels.to(device)
        
        with torch.no_grad():
            test_output = model(test_inputs)
            print(f"Forward pass test successful! Output shape: {test_output.shape}")
        
        # Start training
        model_ft, train_losses, val_accuracies = train_model(
            model, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS
        )
        print("Training finished.")

        # Plot training history (optional)
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
            plt.savefig('training_history.png')
            print("Training history plot saved as 'training_history.png'")
            
        except ImportError:
            print("Matplotlib not available, skipping plot generation")

        # --- 8. Run Evaluation on Test Set ---
        print("\nEvaluating on test set...")
        model_ft.eval()
        running_corrects = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(dataloaders['test']):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model_ft(inputs)
                _, preds = torch.max(outputs, 1)
                
                running_corrects += torch.sum(preds == labels.data)
                
                if batch_idx % 10 == 0:
                    print(f'Test batch {batch_idx}/{len(dataloaders["test"])}')
        
        test_acc = running_corrects.double() / dataset_sizes['test']
        print(f'Test Set Accuracy: {test_acc:.4f}')

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Exiting.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()