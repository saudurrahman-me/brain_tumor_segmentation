"""Tkinter interface for model inference on one patient MRI folder."""
import os
import json
import tkinter as tk
from tkinter import filedialog,messagebox,ttk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import tensorflow as tf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from predict import TumorAnalyzer
from utils.mri_preprocessing import MRIPreprocessor

class SegmentationGUI:
    def __init__(self,root):
        self.root=root; root.title("Research Brain MRI Segmentation"); root.geometry("1100x750")
        self.model=None; self.preprocessor=MRIPreprocessor(); self.mri=None; self.mask=None; self.patient_dir=None
        panel=ttk.Frame(root,padding=10); panel.pack(side=tk.LEFT,fill=tk.Y)
        ttk.Label(panel,text="3D MRI Segmentation",font=("Arial",15,"bold")).pack(pady=8)
        ttk.Button(panel,text="Load .keras model",command=self.load_model).pack(fill=tk.X,pady=4)
        ttk.Button(panel,text="Load patient folder",command=self.load_patient).pack(fill=tk.X,pady=4)
        ttk.Button(panel,text="Run prediction",command=self.run_prediction).pack(fill=tk.X,pady=4)
        ttk.Button(panel,text="Save results",command=self.save_results).pack(fill=tk.X,pady=4)
        self.status=tk.StringVar(value="Load a model and patient folder.")
        ttk.Label(panel,textvariable=self.status,wraplength=220).pack(pady=8)
        self.report=tk.Text(panel,width=32,height=22,wrap=tk.WORD); self.report.pack(fill=tk.BOTH,expand=True)
        self.figure,self.axes=plt.subplots(1,2,figsize=(8,5))
        self.canvas=FigureCanvasTkAgg(self.figure,master=root); self.canvas.get_tk_widget().pack(side=tk.RIGHT,fill=tk.BOTH,expand=True)
    def load_model(self):
        path=filedialog.askopenfilename(filetypes=[("Keras model","*.keras"),("All files","*.*")])
        if not path:return
        try:
            self.model=tf.keras.models.load_model(path,compile=False)
            self.preprocessor=MRIPreprocessor(tuple(self.model.input_shape[1:4]))
            self.status.set("Loaded model: "+os.path.basename(path))
        except Exception as exc: messagebox.showerror("Model loading failed",str(exc))
    def load_patient(self):
        path=filedialog.askdirectory()
        if not path:return
        try:
            data=self.preprocessor.preprocess_patient_data(path)
            self.mri=self.preprocessor.create_multimodal_input(data); self.patient_dir=path; self.mask=None
            self.status.set("Loaded patient: "+os.path.basename(path)); self.show_mri()
        except Exception as exc: messagebox.showerror("MRI loading failed",str(exc))
    def show_mri(self):
        z=self.mri.shape[2]//2
        self.axes[0].clear(); self.axes[0].imshow(self.mri[:,:,z,0],cmap="gray"); self.axes[0].set_title("T1 MRI"); self.axes[0].axis("off")
        self.axes[1].clear(); self.axes[1].set_title("Prediction not run"); self.axes[1].axis("off"); self.canvas.draw_idle()
    def run_prediction(self):
        if self.model is None or self.mri is None:
            messagebox.showwarning("Missing input","Load a model and patient MRI first."); return
        try:
            self.mask=np.argmax(self.model.predict(self.mri[np.newaxis,...],verbose=0)[0],axis=-1).astype(np.uint8)
            report=TumorAnalyzer(self.preprocessor.voxel_spacing_mm).analyze_segmentation(self.mask)
            self.report.delete("1.0",tk.END); self.report.insert(tk.END,"Research measurements only\n\n")
            for name,value in report["volumes"].items(): self.report.insert(tk.END,f"{name}: {value:.3f} cc\n")
            z=self.mask.shape[2]//2; plane=self.mask[:,:,z]; rgb=np.zeros((*plane.shape,3))
            rgb[plane==1]=(1,0,0); rgb[plane==2]=(0,1,0); rgb[plane==3]=(1,1,0)
            self.axes[0].clear(); self.axes[0].imshow(self.mri[:,:,z,0],cmap="gray"); self.axes[0].set_title("T1 MRI")
            self.axes[1].clear(); self.axes[1].imshow(self.mri[:,:,z,0],cmap="gray"); self.axes[1].imshow(rgb,alpha=.45); self.axes[1].set_title("Prediction overlay")
            for ax in self.axes: ax.axis("off")
            self.canvas.draw_idle(); self.status.set("Prediction complete.")
        except Exception as exc: messagebox.showerror("Prediction failed",str(exc))
    def save_results(self):
        if self.mask is None or self.preprocessor.last_affine is None:
            messagebox.showwarning("No results","Run a prediction first."); return
        directory=filedialog.askdirectory()
        if not directory:return
        try:
            patient=os.path.basename(os.path.normpath(self.patient_dir)); out=os.path.join(directory,patient); os.makedirs(out,exist_ok=True)
            nib.save(nib.Nifti1Image(self.mask,self.preprocessor.last_affine),os.path.join(out,"segmentation.nii.gz"))
            report=TumorAnalyzer(self.preprocessor.voxel_spacing_mm).analyze_segmentation(self.mask,patient)
            with open(os.path.join(out,"report.json"),"w",encoding="utf-8") as handle: json.dump(report,handle,indent=2)
            self.status.set("Saved results to "+out)
        except Exception as exc: messagebox.showerror("Save failed",str(exc))
def main():
    root=tk.Tk(); SegmentationGUI(root); root.mainloop()
if __name__=="__main__": main()
