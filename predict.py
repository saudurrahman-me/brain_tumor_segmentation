"""Run inference and save geometry-aware NIfTI masks and tumour-volume reports."""
import argparse,json,os
from datetime import datetime,timezone
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import tensorflow as tf
from utils.mri_preprocessing import MRIPreprocessor

class TumorAnalyzer:
    def __init__(self,voxel_spacing=(1.,1.,1.)):
        spacing=np.asarray(voxel_spacing,dtype=float)
        if spacing.shape!=(3,) or np.any(spacing<=0): raise ValueError("voxel_spacing must contain three positive mm values")
        self.voxel_spacing=tuple(float(v) for v in spacing); self.voxel_volume_cc=float(np.prod(spacing)/1000.)
    def analyze_segmentation(self,seg_mask,patient_id="Unknown"):
        mask=np.asarray(seg_mask)
        if mask.ndim!=3: raise ValueError(f"Expected a 3D mask, got {mask.shape}")
        if not set(np.unique(mask).tolist()).issubset({0,1,2,3}): raise ValueError("Labels must be 0, 1, 2 or 3")
        regions={"Whole_Tumor":mask>0,"Tumor_Core":np.isin(mask,(1,3)),
                 "Enhancing_Tumor":mask==3,"Necrotic_Core":mask==1,"Edema":mask==2}
        return {"patient_id":patient_id,"analysis_date_utc":datetime.now(timezone.utc).isoformat(),
                "voxel_spacing_mm":list(self.voxel_spacing),"voxel_volume_cc":self.voxel_volume_cc,
                "volumes":{k+"_cc":float(np.count_nonzero(v)*self.voxel_volume_cc) for k,v in regions.items()},
                "voxel_counts":{k:int(np.count_nonzero(v)) for k,v in regions.items()},
                "note":"Research output only; not validated for clinical diagnosis or treatment decisions."}
    @staticmethod
    def visualize_segmentation(mri,mask,patient_id,save_dir):
        z=mask.shape[2]//2; image=mri[:,:,z,0]; plane=mask[:,:,z]
        rgb=np.zeros((*plane.shape,3),dtype=np.float32)
        rgb[plane==1]=(1,0,0); rgb[plane==2]=(0,1,0); rgb[plane==3]=(1,1,0)
        fig,axes=plt.subplots(1,3,figsize=(13,4))
        axes[0].imshow(image,cmap="gray"); axes[0].set_title("T1 MRI")
        axes[1].imshow(rgb); axes[1].set_title("Predicted classes")
        axes[2].imshow(image,cmap="gray"); axes[2].imshow(rgb,alpha=.45); axes[2].set_title("Overlay")
        for ax in axes: ax.axis("off")
        fig.tight_layout(); path=os.path.join(save_dir,f"{patient_id}_segmentation.png")
        fig.savefig(path,dpi=150,bbox_inches="tight"); plt.close(fig); return path

def predict_single_patient(model,patient_dir,output_dir,preprocessor):
    patient_id=os.path.basename(os.path.normpath(patient_dir))
    data=preprocessor.preprocess_patient_data(patient_dir); mri=preprocessor.create_multimodal_input(data)
    probabilities=model.predict(mri[np.newaxis,...],verbose=0); mask=np.argmax(probabilities[0],axis=-1).astype(np.uint8)
    if preprocessor.last_affine is None: raise RuntimeError("Missing NIfTI affine metadata")
    report=TumorAnalyzer(preprocessor.voxel_spacing_mm).analyze_segmentation(mask,patient_id)
    out=os.path.join(output_dir,patient_id); os.makedirs(out,exist_ok=True)
    header=preprocessor.last_header.copy() if preprocessor.last_header is not None else None
    if header is not None: header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(mask,preprocessor.last_affine,header=header),os.path.join(out,"segmentation.nii.gz"))
    with open(os.path.join(out,"report.json"),"w",encoding="utf-8") as f: json.dump(report,f,indent=2)
    TumorAnalyzer.visualize_segmentation(mri,mask,patient_id,out)
    print(json.dumps(report["volumes"],indent=2)); return report

def main():
    p=argparse.ArgumentParser(description="Segment patient MRI volumes")
    p.add_argument("--model",required=True); p.add_argument("--input",required=True)
    p.add_argument("--output",default="predictions"); p.add_argument("--batch",action="store_true")
    a=p.parse_args()
    if not os.path.isfile(a.model): p.error(f"Model does not exist: {a.model}")
    model=tf.keras.models.load_model(a.model,compile=False)
    prep=MRIPreprocessor(tuple(model.input_shape[1:4])); os.makedirs(a.output,exist_ok=True)
    if a.batch:
        dirs=sorted(os.path.join(a.input,n) for n in os.listdir(a.input) if os.path.isdir(os.path.join(a.input,n)))
        if not dirs: raise ValueError(f"No patient folders found in {a.input}")
    else: dirs=[a.input]
    for folder in dirs: print(f"Processing {folder}"); predict_single_patient(model,folder,a.output,prep)
if __name__=="__main__": main()
