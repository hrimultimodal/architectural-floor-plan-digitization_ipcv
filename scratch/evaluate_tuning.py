import os
import sys
import yaml
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

# Define paths
data_yaml = "/home/hri_multimodal/Desktop/IPCV/architectural-floor-plan-digitization_ipcv/data/yolo_dataset_processed/dataset.yaml"

models_to_evaluate = {
    "Baseline YOLO11L": "/home/hri_multimodal/Desktop/IPCV/architectural-floor-plan-digitization_ipcv/runs/detect/models/yolo11l/weights/best.pt",
    "Baseline YOLO11L Final": "/home/hri_multimodal/Desktop/IPCV/architectural-floor-plan-digitization_ipcv/runs/detect/models/yolo11l_final/weights/best.pt",
    "Tuning E01 (Resolution 1280)": "/home/hri_multimodal/Desktop/IPCV/19-ipcv-architectural-floor-plan-digitization/architectural-floor-plan-digitization_ipcv/runs/detect/models/tuning_runs/E01_res_1280/weights/best.pt",
    "Tuning E06 (Minority Combo)": "/home/hri_multimodal/Desktop/IPCV/19-ipcv-architectural-floor-plan-digitization/architectural-floor-plan-digitization_ipcv/runs/detect/models/tuning_runs/E06_minority_combo/weights/best.pt",
    "Tuning E07 (Full Combo)": "/home/hri_multimodal/Desktop/IPCV/19-ipcv-architectural-floor-plan-digitization/architectural-floor-plan-digitization_ipcv/runs/detect/models/tuning_runs/E07_full_combo/weights/best.pt"
}

CLASS_NAMES = ["Door", "Window", "Wall", "Staircase", "Toilet", "Sink"]

results = []

for name, path in models_to_evaluate.items():
    if not os.path.exists(path):
        print(f"Weights not found for {name} at {path}")
        continue
    
    print(f"\nEvaluating {name}...")
    model = YOLO(path)
    
    # We run validation on the test split
    metrics = model.val(data=data_yaml, split="test", verbose=False)
    
    # Extract overall metrics
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    
    # Extract speed
    speed = metrics.speed
    preprocess_ms = speed.get("preprocess", 0)
    inference_ms = speed.get("inference", 0)
    postprocess_ms = speed.get("postprocess", 0)
    total_speed_ms = preprocess_ms + inference_ms + postprocess_ms
    
    # Extract class-wise metrics
    ap50_class = metrics.box.ap50
    ap_class = metrics.box.ap  # AP50-95 per class
    
    class_ap50 = {}
    class_ap50_95 = {}
    
    # Map class index to AP
    for idx, c_name in enumerate(CLASS_NAMES):
        if idx < len(ap50_class):
            class_ap50[c_name] = float(ap50_class[idx])
            class_ap50_95[c_name] = float(ap_class[idx])
        else:
            class_ap50[c_name] = 0.0
            class_ap50_95[c_name] = 0.0
            
    row = {
        "Model": name,
        "mAP50": map50,
        "mAP50-95": map50_95,
        "Precision": precision,
        "Recall": recall,
        "Speed (ms/img)": total_speed_ms,
        "Inference (ms)": inference_ms,
    }
    
    # Add class-wise metrics to row
    for c_name in CLASS_NAMES:
        row[f"{c_name}_AP50"] = class_ap50[c_name]
        row[f"{c_name}_AP50-95"] = class_ap50_95[c_name]
        
    results.append(row)

# Create DataFrame and print/save
df = pd.DataFrame(results)
print("\n=== EVALUATION RESULTS (TEST SPLIT) ===")
print(df.to_string(index=False))

# Save results
output_csv = "/home/hri_multimodal/Desktop/IPCV/architectural-floor-plan-digitization_ipcv/scratch/tuning_evaluation_test_split.csv"
df.to_csv(output_csv, index=False)
print(f"\nSaved results to {output_csv}")
