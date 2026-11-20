import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
import os
import numpy as np
from analyze_model import HybridModel, ModelAnalyzer

class ModelImprover:
    def __init__(self, data_dir="celeb-df"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_dir = data_dir
        
    def get_data_stats(self):
        """Analyze dataset statistics"""
        train_dir = os.path.join(self.data_dir, 'train')
        real_count = len(os.listdir(os.path.join(train_dir, 'real')))
        fake_count = len(os.listdir(os.path.join(train_dir, 'fake')))
        
        print(f"📊 Dataset Statistics:")
        print(f"   Real images: {real_count}")
        print(f"   Fake images: {fake_count}")
        print(f"   Total images: {real_count + fake_count}")
        print(f"   Real/Fake ratio: {real_count/fake_count:.2f}")
        
        return real_count, fake_count
    
    def create_improved_transforms(self):
        """Create enhanced data transformations"""
        train_transform = transforms.Compose([
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
        
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        return train_transform, val_transform
    
    def setup_improved_training(self):
        """Setup improved training configuration"""
        
        # Analyze current model first
        print("🔍 Analyzing current model performance...")
        analyzer = ModelAnalyzer(self.data_dir)
        accuracy, _, _, _ = analyzer.test_model_accuracy()
        
        if accuracy < 80:
            print(f"⚠️  Current accuracy ({accuracy:.2f}%) is low. Consider these improvements:")
        else:
            print(f"✅ Current accuracy ({accuracy:.2f}%) is good!")
        
        # Get data statistics
        real_count, fake_count = self.get_data_stats()
        
        # Suggest improvements based on analysis
        self.suggest_improvements(real_count, fake_count, accuracy)
        
        return self.create_improved_training_plan(real_count, fake_count)
    
    def suggest_improvements(self, real_count, fake_count, accuracy):
        """Suggest specific improvements based on analysis"""
        print("\n💡 IMPROVEMENT SUGGESTIONS:")
        print("="*50)
        
        # Check class imbalance
        ratio = real_count / fake_count
        if ratio < 0.8 or ratio > 1.2:
            print(f"❌ Class imbalance detected (ratio: {ratio:.2f})")
            print("   → Use WeightedRandomSampler or oversample minority class")
            print("   → Use class weights in loss function")
        
        # Data quality suggestions
        if real_count + fake_count < 10000:
            print("❌ Small dataset detected")
            print("   → Collect more training data")
            print("   → Use more aggressive data augmentation")
            print("   → Consider transfer learning from larger datasets")
        
        # Architecture suggestions
        if accuracy < 85:
            print("⚡ Model architecture improvements:")
            print("   → Try larger backbone (EfficientNet-B3/B4)")
            print("   → Add spatial attention modules")
            print("   → Increase transformer layers")
            print("   → Use ensemble of multiple models")
        
        # Training suggestions
        print("🎯 Training improvements:")
        print("   → Use Focal Loss for hard examples")
        print("   → Implement learning rate warmup")
        print("   → Use cosine annealing scheduler")
        print("   → Add gradient clipping")
        print("   → Train for more epochs with early stopping")
        
        # Data augmentation suggestions
        print("🔄 Advanced data augmentation:")
        print("   → Add JPEG compression artifacts")
        print("   → Use mixup or cutmix augmentation")
        print("   → Add motion blur for video frames")
        print("   → Include various image quality levels")
    
    def create_improved_training_plan(self, real_count, fake_count):
        """Create an improved training plan"""
        
        plan = {
            'data_augmentation': {
                'transforms': self.create_improved_transforms(),
                'class_weights': [1.0, real_count/fake_count] if real_count > fake_count else [fake_count/real_count, 1.0]
            },
            'model_architecture': {
                'backbone': 'EfficientNet-B3',
                'transformer_layers': 4,
                'attention_heads': 8
            },
            'training_config': {
                'loss_function': 'FocalLoss',
                'optimizer': 'AdamW',
                'scheduler': 'CosineAnnealingWarmRestart',
                'batch_size': 16,
                'epochs': 50
            },
            'advanced_techniques': [
                'Gradient Accumulation',
                'Mixed Precision Training',
                'Stochastic Weight Averaging'
            ]
        }
        
        print("\n📋 RECOMMENDED TRAINING PLAN:")
        print("="*50)
        for category, details in plan.items():
            print(f"\n{category.replace('_', ' ').title()}:")
            if isinstance(details, dict):
                for key, value in details.items():
                    print(f"  • {key}: {value}")
            else:
                for item in details:
                    print(f"  • {item}")
        
        return plan

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

def main():
    print("🧠 Deepfake Detection Model Improver")
    print("="*50)
    
    improver = ModelImprover(data_dir="celeb-df")
    training_plan = improver.setup_improved_training()
    
    print("\n🎯 Next Steps:")
    print("1. Review the suggestions above")
    print("2. Implement the recommended improvements")
    print("3. Retrain your model with the new configuration")
    print("4. Run analysis again to measure improvement")
    
    # Save training plan to file
    with open('training_improvement_plan.txt', 'w') as f:
        f.write("Deepfake Detection Model Improvement Plan\n")
        f.write("="*50 + "\n\n")
        for category, details in training_plan.items():
            f.write(f"{category.upper()}:\n")
            if isinstance(details, dict):
                for key, value in details.items():
                    f.write(f"  {key}: {value}\n")
            else:
                for item in details:
                    f.write(f"  - {item}\n")
            f.write("\n")
    
    print("\n✅ Improvement plan saved as 'training_improvement_plan.txt'")

if __name__ == "__main__":
    main()