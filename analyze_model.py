import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# --- Define your model architecture (same as training) ---
class HybridModel(nn.Module):
    def __init__(self, num_classes=2, d_model=1280, nhead=8, num_encoder_layers=3, dim_feedforward=1024, dropout=0.1):
        super(HybridModel, self).__init__()
        
        self.cnn_backbone = models.efficientnet_b0(weights=None).features
        
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

class ModelAnalyzer:
    def __init__(self, data_dir="celeb-df"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_dir = data_dir
        self.model = self.load_model()
        self.class_names = ['fake', 'real']
        
        # Data transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Load test dataset
        self.test_dataset = datasets.ImageFolder(
            os.path.join(data_dir, 'test'), 
            self.transform
        )
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=32, 
            shuffle=False, 
            num_workers=0
        )
    
    def load_model(self):
        """Load the trained model"""
        model = HybridModel(num_classes=2)
        
        model_paths = ['best_hybrid_model.pth', './best_hybrid_model.pth']
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    checkpoint = torch.load(model_path, map_location=self.device)
                    if isinstance(checkpoint, dict):
                        model.load_state_dict(checkpoint)
                    else:
                        model.load_state_dict(checkpoint)
                    
                    model = model.to(self.device)
                    model.eval()
                    print(f"✓ Model loaded from {model_path}")
                    return model
                except Exception as e:
                    print(f"✗ Error loading {model_path}: {e}")
        
        raise Exception("No model file found!")
    
    def test_model_accuracy(self):
        """Test overall accuracy"""
        self.model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        accuracy = 100 * correct / total
        print(f'\n🎯 Overall Test Accuracy: {accuracy:.2f}%')
        print(f'📊 Correct: {correct}/{total}')
        
        return accuracy, all_predictions, all_labels, all_probabilities
    
    def analyze_misclassifications(self, predictions, labels, probabilities):
        """Analyze where the model is making mistakes"""
        misclassified = []
        
        for i, (pred, true, prob) in enumerate(zip(predictions, labels, probabilities)):
            if pred != true:
                misclassified.append({
                    'index': i,
                    'predicted': self.class_names[pred],
                    'true': self.class_names[true],
                    'confidence': max(prob),
                    'prob_real': prob[1],
                    'prob_fake': prob[0]
                })
        
        print(f"\n❌ Misclassified samples: {len(misclassified)}/{len(predictions)}")
        
        # Analyze misclassification patterns
        fake_as_real = [m for m in misclassified if m['true'] == 'fake' and m['predicted'] == 'real']
        real_as_fake = [m for m in misclassified if m['true'] == 'real' and m['predicted'] == 'fake']
        
        print(f"   Fake classified as Real: {len(fake_as_real)}")
        print(f"   Real classified as Fake: {len(real_as_fake)}")
        
        # Confidence analysis
        if misclassified:
            avg_confidence_misclassified = np.mean([m['confidence'] for m in misclassified])
            print(f"   Average confidence on wrong predictions: {avg_confidence_misclassified:.3f}")
        
        return misclassified
    
    def plot_confusion_matrix(self, predictions, labels):
        """Plot confusion matrix"""
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm
    
    def print_detailed_report(self, predictions, labels):
        """Print detailed classification report"""
        print("\n" + "="*60)
        print("📊 DETAILED CLASSIFICATION REPORT")
        print("="*60)
        
        report = classification_report(labels, predictions, 
                                     target_names=self.class_names, 
                                     digits=4)
        print(report)
        
        # Calculate per-class accuracy
        cm = confusion_matrix(labels, predictions)
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        
        print("\n📈 Per-class Accuracy:")
        for i, class_name in enumerate(self.class_names):
            print(f"   {class_name}: {class_accuracy[i]:.4f}")
    
    def analyze_confidence_distribution(self, probabilities, labels):
        """Analyze confidence distribution for correct vs incorrect predictions"""
        correct_confidences = []
        incorrect_confidences = []
        
        for i, (prob, true_label) in enumerate(zip(probabilities, labels)):
            pred_label = np.argmax(prob)
            confidence = np.max(prob)
            
            if pred_label == true_label:
                correct_confidences.append(confidence)
            else:
                incorrect_confidences.append(confidence)
        
        plt.figure(figsize=(10, 6))
        
        if correct_confidences:
            plt.hist(correct_confidences, bins=20, alpha=0.7, label='Correct Predictions', color='green')
        if incorrect_confidences:
            plt.hist(incorrect_confidences, bins=20, alpha=0.7, label='Incorrect Predictions', color='red')
        
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.title('Confidence Distribution: Correct vs Incorrect Predictions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('confidence_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        if correct_confidences:
            print(f"\n✅ Average confidence on correct predictions: {np.mean(correct_confidences):.4f}")
        if incorrect_confidences:
            print(f"❌ Average confidence on incorrect predictions: {np.mean(incorrect_confidences):.4f}")
    
    def find_most_confident_errors(self, misclassified, top_k=10):
        """Find the most confident wrong predictions"""
        if not misclassified:
            print("No misclassifications to analyze.")
            return
        
        # Sort by confidence (descending)
        most_confident_errors = sorted(misclassified, key=lambda x: x['confidence'], reverse=True)[:top_k]
        
        print(f"\n🔍 Top {top_k} Most Confident Errors:")
        print("-" * 80)
        print(f"{'Index':<8} {'True':<10} {'Predicted':<10} {'Confidence':<12} {'P(Real)':<10} {'P(Fake)':<10}")
        print("-" * 80)
        
        for error in most_confident_errors:
            print(f"{error['index']:<8} {error['true']:<10} {error['predicted']:<10} "
                  f"{error['confidence']:<12.4f} {error['prob_real']:<10.4f} {error['prob_fake']:<10.4f}")
    
    def run_complete_analysis(self):
        """Run complete analysis pipeline"""
        print("🚀 Starting Comprehensive Model Analysis...")
        print("="*60)
        
        # 1. Test accuracy
        accuracy, predictions, labels, probabilities = self.test_model_accuracy()
        
        # 2. Analyze misclassifications
        misclassified = self.analyze_misclassifications(predictions, labels, probabilities)
        
        # 3. Plot confusion matrix
        print("\n📈 Generating confusion matrix...")
        self.plot_confusion_matrix(predictions, labels)
        
        # 4. Print detailed report
        self.print_detailed_report(predictions, labels)
        
        # 5. Analyze confidence distribution
        print("\n📊 Analyzing confidence distribution...")
        self.analyze_confidence_distribution(probabilities, labels)
        
        # 6. Find most confident errors
        self.find_most_confident_errors(misclassified)
        
        print("\n" + "="*60)
        print("✅ Analysis Complete!")
        print("="*60)
        print("Generated files:")
        print("  - confusion_matrix.png")
        print("  - confidence_distribution.png")
        print(f"\n💡 Model Performance Summary:")
        print(f"   Overall Accuracy: {accuracy:.2f}%")
        print(f"   Misclassification Rate: {len(misclassified)/len(predictions)*100:.2f}%")

# --- Run the analysis ---
if __name__ == "__main__":
    try:
        analyzer = ModelAnalyzer(data_dir="celeb-df")
        analyzer.run_complete_analysis()
    except Exception as e:
        print(f"Error during analysis: {e}")
        print("Please make sure:")
        print("1. Your dataset folder 'celeb-df' exists")
        print("2. The folder structure is: celeb-df/test/real/ and celeb-df/test/fake/")
        print("3. 'best_hybrid_model.pth' exists in the current directory")