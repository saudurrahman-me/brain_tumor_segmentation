"""Train a 3D U-Net and evaluate once on a held-out test split."""
import argparse, json, os, random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from models.unet_model import UNet3D
from utils.data_generator import BraTSDataGenerator, create_data_split
from utils.evaluation import evaluate_model

def train_model(data_dir, output_dir="results", epochs=50, batch_size=1,
                learning_rate=1e-4, seed=42, target_shape=(128,128,128), base_filters=16):
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size and learning_rate must be positive")
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
    for gpu in tf.config.list_physical_devices("GPU"):
        try: tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc: print(f"GPU configuration skipped: {exc}")
    os.makedirs(output_dir, exist_ok=True)
    model_dir=os.path.join(output_dir,"models"); os.makedirs(model_dir,exist_ok=True)
    train_ids,val_ids,test_ids=create_data_split(data_dir,seed=seed)
    with open(os.path.join(output_dir,"data_split.json"),"w") as f:
        json.dump({"seed":seed,"train":train_ids,"validation":val_ids,"test":test_ids},f,indent=2)
    train_gen=BraTSDataGenerator(data_dir,train_ids,batch_size,target_shape,shuffle=True,augment=True,seed=seed)
    val_gen=BraTSDataGenerator(data_dir,val_ids,batch_size,target_shape,shuffle=False,augment=False,seed=seed)
    test_gen=BraTSDataGenerator(data_dir,test_ids,batch_size,target_shape,shuffle=False,augment=False,seed=seed)
    wrapper=UNet3D(input_shape=(*target_shape,4),base_filters=base_filters)
    model=wrapper.build_model(); wrapper.compile_model(learning_rate)
    print(f"Model parameters: {model.count_params():,}")
    best_path=os.path.join(model_dir,"best_model.keras")
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(best_path,monitor="val_loss",save_best_only=True,verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=10,restore_best_weights=True,verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=.5,patience=5,min_lr=1e-7,verbose=1),
        tf.keras.callbacks.CSVLogger(os.path.join(output_dir,"training_log.csv"))]
    history=model.fit(train_gen,validation_data=val_gen,epochs=epochs,callbacks=callbacks,verbose=1)
    model.save(os.path.join(model_dir,"final_model.keras"))
    best=tf.keras.models.load_model(best_path,compile=False)
    metrics=evaluate_model(best,test_gen,os.path.join(output_dir,"test_metrics.json"))
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].plot(history.history["loss"],label="Training"); axes[0].plot(history.history["val_loss"],label="Validation")
    axes[0].set(title="Loss",xlabel="Epoch",ylabel="Loss"); axes[0].legend()
    axes[1].plot(history.history["foreground_dice"],label="Training"); axes[1].plot(history.history["val_foreground_dice"],label="Validation")
    axes[1].set(title="Foreground Dice",xlabel="Epoch",ylabel="Dice"); axes[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(output_dir,"training_history.png"),dpi=150); plt.close(fig)
    print(f"Training complete. Best checkpoint: {best_path}")
    print(f"Held-out test results: {os.path.join(output_dir,'test_metrics.json')}")
    return best,history,metrics

def main():
    p=argparse.ArgumentParser(description="Train 3D U-Net brain MRI segmentation")
    p.add_argument("--data_dir",required=True); p.add_argument("--output_dir",default="results")
    p.add_argument("--epochs",type=int,default=50); p.add_argument("--batch_size",type=int,default=1)
    p.add_argument("--learning_rate",type=float,default=1e-4); p.add_argument("--seed",type=int,default=42)
    p.add_argument("--base_filters",type=int,default=16)
    a=p.parse_args()
    train_model(a.data_dir,a.output_dir,a.epochs,a.batch_size,a.learning_rate,a.seed,base_filters=a.base_filters)
if __name__=="__main__": main()
