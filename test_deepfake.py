import torch
import torch.nn as nn
from torchvision import transforms, models
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn.functional as F

# --- Define the Hybrid CNN-Transformer Model (same as training) ---
class HybridModel(nn.Module):
    def __init__(self, num_classes=2, d_model=1280, nhead=8, num_encoder_layers=3, dim_feedforward=1024, dropout=0.1):
        super(HybridModel, self).__init__()
        
        # CNN Backbone
        self.cnn_backbone = models.efficientnet_b0(weights=None).features
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Classification Head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        # CNN backbone
        x = self.cnn_backbone(x)
        
        # Prepare for Transformer
        b, c, h, w = x.shape
        x = x.flatten(2)
        x = x.permute(2, 0, 1)
        
        # Transformer Encoder
        x = self.transformer_encoder(x)
        
        # Global Pooling
        x = x.permute(1, 2, 0)
        x = self.global_pool(x)
        x = x.squeeze(-1)

        # Classifier
        x = self.classifier(x)
        return x

class DeepfakeDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load the trained model
        self.model = self.load_model()
        self.class_names = ['fake', 'real']  # Adjust based on your training classes
        
        # Define image transformations (same as validation)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Create GUI
        self.create_gui()
    
    def load_model(self):
        """Load the trained model weights"""
        model = HybridModel(num_classes=2)
        
        # Look for the saved model file
        model_paths = [
            'best_hybrid_model.pth',
            './best_hybrid_model.pth',
            'model/best_hybrid_model.pth'
        ]
        
        model_loaded = False
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    # For CPU loading, specify map_location
                    checkpoint = torch.load(model_path, map_location=self.device)
                    
                    # Handle both state_dict and full model save
                    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                    
                    model = model.to(self.device)
                    model.eval()
                    print(f"Model loaded successfully from {model_path}")
                    model_loaded = True
                    break
                    
                except Exception as e:
                    print(f"Error loading model from {model_path}: {e}")
        
        if not model_loaded:
            print("WARNING: No model file found. Please ensure 'best_hybrid_model.pth' exists.")
            # Initialize with random weights for demo
            model = model.to(self.device)
            model.eval()
        
        return model
    
    def predict_image(self, image_path):
        """Predict if image is real or fake"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            original_image = image.copy()
            
            # Apply transformations
            input_tensor = self.transform(image)
            input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
            
            # Move to device
            input_batch = input_batch.to(self.device)
            
            # Predict
            with torch.no_grad():
                output = self.model(input_batch)
                probabilities = F.softmax(output, dim=1)
                confidence, prediction = torch.max(probabilities, 1)
                
            # Get results
            predicted_class = self.class_names[prediction.item()]
            confidence_score = confidence.item()
            
            # Get probability for each class
            probs = probabilities[0].cpu().numpy()
            fake_prob = probs[0]  # Assuming index 0 is fake
            real_prob = probs[1]  # Assuming index 1 is real
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence_score,
                'fake_probability': fake_prob,
                'real_probability': real_prob,
                'original_image': original_image,
                'processed_image': input_tensor
            }
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None
    
    def create_gui(self):
        """Create the graphical user interface"""
        self.root = tk.Tk()
        self.root.title("Deepfake Image Detector")
        self.root.geometry("800x700")
        self.root.configure(bg='#2c3e50')
        
        # Title
        title_label = tk.Label(
            self.root, 
            text="Deepfake Image Detection", 
            font=('Arial', 20, 'bold'),
            fg='white',
            bg='#2c3e50'
        )
        title_label.pack(pady=20)
        
        # Instructions
        instruction_label = tk.Label(
            self.root,
            text="Upload an image to check if it's real or AI-generated/fake",
            font=('Arial', 12),
            fg='#ecf0f1',
            bg='#2c3e50'
        )
        instruction_label.pack(pady=10)
        
        # Upload button
        upload_btn = tk.Button(
            self.root,
            text="Upload Image",
            command=self.upload_image,
            font=('Arial', 14, 'bold'),
            bg='#3498db',
            fg='white',
            padx=20,
            pady=10,
            relief='raised',
            bd=3
        )
        upload_btn.pack(pady=20)
        
        # Image display frame
        self.image_frame = tk.Frame(self.root, bg='#2c3e50')
        self.image_frame.pack(pady=10)
        
        self.image_label = tk.Label(self.image_frame, bg='#2c3e50')
        self.image_label.pack()
        
        # Results frame
        self.results_frame = tk.Frame(self.root, bg='#2c3e50')
        self.results_frame.pack(pady=20, fill='x')
        
        # Initialize result labels
        self.result_text = tk.StringVar()
        self.result_text.set("Please upload an image to analyze")
        
        self.result_label = tk.Label(
            self.results_frame,
            textvariable=self.result_text,
            font=('Arial', 16, 'bold'),
            fg='#ecf0f1',
            bg='#2c3e50',
            wraplength=700
        )
        self.result_label.pack()
        
        # Confidence label
        self.confidence_text = tk.StringVar()
        self.confidence_label = tk.Label(
            self.results_frame,
            textvariable=self.confidence_text,
            font=('Arial', 12),
            fg='#bdc3c7',
            bg='#2c3e50'
        )
        self.confidence_label.pack(pady=5)
        
        # Probability bars frame
        self.probability_frame = tk.Frame(self.root, bg='#2c3e50')
        self.probability_frame.pack(pady=10, fill='x')
        
        # Status bar
        self.status_text = tk.StringVar()
        self.status_text.set("Ready to analyze images")
        self.status_label = tk.Label(
            self.root,
            textvariable=self.status_text,
            font=('Arial', 10),
            fg='#7f8c8d',
            bg='#2c3e50',
            relief='sunken',
            bd=1
        )
        self.status_label.pack(side='bottom', fill='x', pady=5)
    
    def upload_image(self):
        """Handle image upload"""
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.analyze_image(file_path)
    
    def analyze_image(self, image_path):
        """Analyze the uploaded image"""
        self.status_text.set("Analyzing image...")
        self.root.update()
        
        # Clear previous results
        for widget in self.probability_frame.winfo_children():
            widget.destroy()
        
        # Make prediction
        result = self.predict_image(image_path)
        
        if result:
            self.display_results(image_path, result)
        else:
            messagebox.showerror("Error", "Failed to analyze the image. Please try another image.")
            self.status_text.set("Analysis failed")
    
    def display_results(self, image_path, result):
        """Display the analysis results"""
        # Display image
        image = result['original_image']
        image.thumbnail((400, 400))  # Resize for display
        photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo  # Keep a reference
        
        # Determine color based on prediction
        if result['predicted_class'] == 'real':
            color = '#27ae60'  # Green for real
            emoji = "✅"
        else:
            color = '#e74c3c'  # Red for fake
            emoji = "⚠️"
        
        # Update result text
        self.result_text.set(
            f"{emoji} Prediction: {result['predicted_class'].upper()} {emoji}\n"
            f"Confidence: {result['confidence']*100:.2f}%"
        )
        self.result_label.configure(fg=color)
        
        # Update confidence text
        self.confidence_text.set(
            f"Real: {result['real_probability']*100:.2f}% | "
            f"Fake: {result['fake_probability']*100:.2f}%"
        )
        
        # Create probability bars
        self.create_probability_bars(result['real_probability'], result['fake_probability'])
        
        # Update status
        self.status_text.set(f"Analysis complete - {os.path.basename(image_path)}")
        
        # Print detailed results to console
        print("\n" + "="*50)
        print("DEEPFAKE DETECTION RESULTS")
        print("="*50)
        print(f"Image: {os.path.basename(image_path)}")
        print(f"Prediction: {result['predicted_class'].upper()}")
        print(f"Confidence: {result['confidence']*100:.2f}%")
        print(f"Real Probability: {result['real_probability']*100:.2f}%")
        print(f"Fake Probability: {result['fake_probability']*100:.2f}%")
        print("="*50)
    
    def create_probability_bars(self, real_prob, fake_prob):
        """Create visual probability bars"""
        # Real probability bar
        real_frame = tk.Frame(self.probability_frame, bg='#2c3e50')
        real_frame.pack(fill='x', padx=50, pady=5)
        
        tk.Label(
            real_frame,
            text="REAL:",
            font=('Arial', 10, 'bold'),
            fg='#27ae60',
            bg='#2c3e50',
            width=8
        ).pack(side='left')
        
        real_canvas = tk.Canvas(real_frame, width=300, height=20, bg='#34495e', highlightthickness=0)
        real_canvas.pack(side='left', padx=5)
        real_canvas.create_rectangle(0, 0, real_prob * 300, 20, fill='#27ae60', outline='')
        real_canvas.create_text(150, 10, text=f"{real_prob*100:.1f}%", fill='white', font=('Arial', 9, 'bold'))
        
        # Fake probability bar
        fake_frame = tk.Frame(self.probability_frame, bg='#2c3e50')
        fake_frame.pack(fill='x', padx=50, pady=5)
        
        tk.Label(
            fake_frame,
            text="FAKE:",
            font=('Arial', 10, 'bold'),
            fg='#e74c3c',
            bg='#2c3e50',
            width=8
        ).pack(side='left')
        
        fake_canvas = tk.Canvas(fake_frame, width=300, height=20, bg='#34495e', highlightthickness=0)
        fake_canvas.pack(side='left', padx=5)
        fake_canvas.create_rectangle(0, 0, fake_prob * 300, 20, fill='#e74c3c', outline='')
        fake_canvas.create_text(150, 10, text=f"{fake_prob*100:.1f}%", fill='white', font=('Arial', 9, 'bold'))
    
    def run(self):
        """Start the application"""
        self.root.mainloop()

# --- Command Line Version ---
def command_line_test():
    """Command line interface for testing"""
    detector = DeepfakeDetector()
    
    print("\n" + "="*50)
    print("DEEPFAKE DETECTION - COMMAND LINE MODE")
    print("="*50)
    
    while True:
        print("\nOptions:")
        print("1. Test an image")
        print("2. Launch GUI")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            image_path = input("Enter the path to the image: ").strip()
            if os.path.exists(image_path):
                result = detector.predict_image(image_path)
                if result:
                    print(f"\nResults for {os.path.basename(image_path)}:")
                    print(f"Prediction: {result['predicted_class'].upper()}")
                    print(f"Confidence: {result['confidence']*100:.2f}%")
                    print(f"Real Probability: {result['real_probability']*100:.2f}%")
                    print(f"Fake Probability: {result['fake_probability']*100:.2f}%")
                else:
                    print("Error analyzing image.")
            else:
                print("File not found. Please check the path.")
        
        elif choice == '2':
            print("Launching GUI...")
            detector.run()
            break
        
        elif choice == '3':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    # Check if we're in a command line environment or GUI environment
    try:
        # Try to create a simple GUI to see if we're in a GUI environment
        test_root = tk.Tk()
        test_root.destroy()
        
        # If successful, ask user for preference
        print("Deepfake Image Detector")
        print("1. Command Line Interface")
        print("2. Graphical User Interface")
        
        choice = input("Choose interface (1 or 2): ").strip()
        
        if choice == '1':
            command_line_test()
        else:
            detector = DeepfakeDetector()
            detector.run()
            
    except:
        # If GUI fails, use command line
        print("GUI not available. Using command line mode.")
        command_line_test()