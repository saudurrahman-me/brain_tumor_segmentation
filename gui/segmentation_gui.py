
# File 7: Simple GUI application
#!/usr/bin/env python3
"""
Simple GUI for Brain Tumor Segmentation

A user-friendly interface for doctors to:
1. Load MRI scans
2. Run segmentation
3. View results
4. Generate reports

Usage:
    python gui/segmentation_gui.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet_model import UNet3D
from utils.mri_preprocessing import MRIPreprocessor

class SimpleSegmentationGUI:
    """Simple GUI for brain tumor segmentation"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Brain Tumor Segmentation System")
        self.root.geometry("1200x800")
        
        # Initialize components
        self.preprocessor = MRIPreprocessor(target_shape=(128, 128, 128))
        self.model = None
        self.current_mri = None
        self.current_segmentation = None
        
        self.create_ui()
    
    def create_ui(self):
        """Create the user interface"""
        # Left panel - Controls
        left_frame = ttk.Frame(self.root, padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # Title
        title = ttk.Label(left_frame, text="Brain Tumor Segmentation", 
                         font=('Helvetica', 16, 'bold'))
        title.pack(pady=10)
        
        # Step 1: Load Model
        model_frame = ttk.LabelFrame(left_frame, text="Step 1: Load Model", padding=10)
        model_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(model_frame, text="Select Model File", 
                  command=self.load_model).pack(fill=tk.X)
        
        self.model_label = ttk.Label(model_frame, text="No model loaded", foreground="red")
        self.model_label.pack(pady=5)
        
        # Step 2: Load MRI
        mri_frame = ttk.LabelFrame(left_frame, text="Step 2: Load MRI", padding=10)
        mri_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(mri_frame, text="Select Patient Folder", 
                  command=self.load_mri).pack(fill=tk.X)
        
        self.mri_label = ttk.Label(mri_frame, text="No MRI loaded")
        self.mri_label.pack(pady=5)
        
        # Step 3: Run Segmentation
        seg_frame = ttk.LabelFrame(left_frame, text="Step 3: Segmentation", padding=10)
        seg_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(seg_frame, text="Run Segmentation", 
                  command=self.run_segmentation).pack(fill=tk.X)
        
        # Results display
        results_frame = ttk.LabelFrame(left_frame, text="Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=10, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        self.results_text.insert(tk.END, "Results will appear here...")
        
        # Right panel - Visualization
        right_frame = ttk.Frame(self.root, padding="10")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Matplotlib figure for displaying images
        self.fig, self.axes = plt.subplots(1, 2, figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def set_status(self, message):
        """Update status bar"""
        self.status_bar.config(text=message)
        self.root.update()
    
    def load_model(self):
        """Load trained model"""
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("Model files", "*.h5"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.set_status("Loading model...")
                unet = UNet3D()
                unet.load_model(file_path)
                unet.compile_model()
                self.model = unet.model
                
                self.model_label.config(text=f"Loaded: {os.path.basename(file_path)}", 
                                       foreground="green")
                self.set_status("Model loaded successfully")
                messagebox.showinfo("Success", "Model loaded!")
            except Exception as e:
                self.set_status("Error loading model")
                messagebox.showerror("Error", str(e))
    
    def load_mri(self):
        """Load MRI scans"""
        folder_path = filedialog.askdirectory(title="Select Patient Folder")
        
        if folder_path:
            try:
                self.set_status("Loading MRI...")
                patient_data = self.preprocessor.preprocess_patient_data(folder_path)
                
                if patient_data:
                    self.current_mri = self.preprocessor.create_multimodal_input(patient_data)
                    patient_name = os.path.basename(folder_path)
                    
                    self.mri_label.config(text=f"Loaded: {patient_name}")
                    self.set_status(f"MRI loaded: {self.current_mri.shape}")
                    
                    # Display middle slice
                    self.display_mri_slice()
                    messagebox.showinfo("Success", "MRI loaded!")
                else:
                    messagebox.showerror("Error", "Could not load MRI files")
            except Exception as e:
                self.set_status("Error loading MRI")
                messagebox.showerror("Error", str(e))
    
    def display_mri_slice(self):
        """Display MRI slice"""
        if self.current_mri is None:
            return
        
        # Show middle slice of T1 modality
        slice_idx = self.current_mri.shape[2] // 2
        mri_slice = self.current_mri[:, :, slice_idx, 0]
        
        self.axes[0].clear()
        self.axes[0].imshow(mri_slice, cmap='gray')
        self.axes[0].set_title('MRI (T1)')
        self.axes[0].axis('off')
        
        self.axes[1].clear()
        self.axes[1].text(0.5, 0.5, 'Run segmentation\\nto see results', 
                         ha='center', va='center', fontsize=12)
        self.axes[1].axis('off')
        
        self.canvas.draw()
    
    def run_segmentation(self):
        """Run tumor segmentation"""
        if self.model is None:
            messagebox.showwarning("Warning", "Please load a model first")
            return
        
        if self.current_mri is None:
            messagebox.showwarning("Warning", "Please load MRI data first")
            return
        
        try:
            self.set_status("Running segmentation...")
            
            # Prepare input
            input_batch = np.expand_dims(self.current_mri, axis=0)
            
            # Predict
            prediction = self.model.predict(input_batch, verbose=0)
            self.current_segmentation = np.argmax(prediction[0], axis=-1)
            
            # Calculate volumes
            from predict import TumorAnalyzer
            analyzer = TumorAnalyzer()
            report = analyzer.analyze_segmentation(self.current_segmentation)
            
            # Display results
            self.display_results(report)
            self.set_status("Segmentation complete")
            
        except Exception as e:
            self.set_status("Error in segmentation")
            messagebox.showerror("Error", str(e))
    
    def display_results(self, report):
        """Display segmentation results"""
        # Update text
        results_text = f"""
SEGMENTATION RESULTS
{'='*40}

Tumor Volumes:
• Whole Tumor: {report['volumes']['Whole_Tumor_cc']:.2f} cc
• Tumor Core: {report['volumes']['Tumor_Core_cc']:.2f} cc
• Enhancing Tumor: {report['volumes']['Enhancing_Tumor_cc']:.2f} cc
• Necrotic Core: {report['volumes']['Necrotic_cc']:.2f} cc
• Edema: {report['volumes']['Edema_cc']:.2f} cc

Analysis complete.
"""
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, results_text)
        
        # Update visualization
        slice_idx = self.current_mri.shape[2] // 2
        mri_slice = self.current_mri[:, :, slice_idx, 0]
        seg_slice = self.current_segmentation[:, :, slice_idx]
        
        # Create colored segmentation
        seg_rgb = np.zeros((*seg_slice.shape, 3))
        seg_rgb[seg_slice == 1] = [1, 0, 0]  # Red
        seg_rgb[seg_slice == 2] = [0, 1, 0]  # Green
        seg_rgb[seg_slice == 3] = [1, 1, 0]  # Yellow
        
        self.axes[0].clear()
        self.axes[0].imshow(mri_slice, cmap='gray')
        self.axes[0].set_title('MRI (T1)')
        self.axes[0].axis('off')
        
        self.axes[1].clear()
        self.axes[1].imshow(seg_rgb)
        self.axes[1].set_title('Segmentation\\nR=Necrotic, G=Edema, Y=Enhancing')
        self.axes[1].axis('off')
        
        self.canvas.draw()

def main():
    """Start the GUI"""
    root = tk.Tk()
    app = SimpleSegmentationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()


with open('/mnt/kimi/output/brain_tumor_segmentation/gui/segmentation_gui.py', 'w') as f:
    f.write(gui_code)

print("✅ Created: gui/segmentation_gui.py")
print("\nSimple GUI for doctors to use the system.")
print("Run with: python gui/segmentation_gui.py")
