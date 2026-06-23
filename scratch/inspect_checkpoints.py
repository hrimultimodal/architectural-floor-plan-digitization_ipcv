import torch
import os

models = {
    "Baseline YOLO11L Final": "/home/hri_multimodal/Desktop/IPCV/architectural-floor-plan-digitization_ipcv/runs/detect/models/yolo11l_final/weights/best.pt",
    "Tuning E01 (Resolution 1280)": "/home/hri_multimodal/Desktop/IPCV/19-ipcv-architectural-floor-plan-digitization/architectural-floor-plan-digitization_ipcv/runs/detect/models/tuning_runs/E01_res_1280/weights/best.pt",
    "Tuning E06 (Minority Combo)": "/home/hri_multimodal/Desktop/IPCV/19-ipcv-architectural-floor-plan-digitization/architectural-floor-plan-digitization_ipcv/runs/detect/models/tuning_runs/E06_minority_combo/weights/best.pt",
    "Tuning E07 (Full Combo)": "/home/hri_multimodal/Desktop/IPCV/19-ipcv-architectural-floor-plan-digitization/architectural-floor-plan-digitization_ipcv/runs/detect/models/tuning_runs/E07_full_combo/weights/best.pt"
}

for name, path in models.items():
    if not os.path.exists(path):
        print(f"{name} path not found: {path}")
        continue
    
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    print(f"\n=== Model: {name} ===")
    print(f"File path: {path}")
    
    # Check stored arguments in checkpoint
    train_args = ckpt.get("train_args", {})
    if not train_args:
        # Check in other keys
        train_args = ckpt.get("args", {})
        
    if isinstance(train_args, dict):
        print("Training parameters:")
        for key in ["imgsz", "batch", "epochs", "cls", "box", "copy_paste", "degrees"]:
            print(f"  {key}: {train_args.get(key, 'N/A')}")
    else:
        print(f"train_args type is: {type(train_args)}")
        # If it's an argparse.Namespace or similar, print its dict representation
        try:
            d = vars(train_args)
            for key in ["imgsz", "batch", "epochs", "cls", "box", "copy_paste", "degrees"]:
                print(f"  {key}: {d.get(key, 'N/A')}")
        except Exception as e:
            print(f"Could not convert to dict: {e}")
            print(train_args)
            
    # Check epoch and validation info in checkpoint
    epoch = ckpt.get("epoch", "N/A")
    best_fitness = ckpt.get("best_fitness", "N/A")
    print(f"Saved Epoch: {epoch}")
    print(f"Best Fitness: {best_fitness}")
