import torch

path1 = "/home/hri_multimodal/Desktop/IPCV/architectural-floor-plan-digitization_ipcv/runs/detect/models/yolo11l_final/weights/best.pt"
path2 = "/home/hri_multimodal/Desktop/IPCV/19-ipcv-architectural-floor-plan-digitization/architectural-floor-plan-digitization_ipcv/runs/detect/models/tuning_runs/E01_res_1280/weights/best.pt"

ckpt1 = torch.load(path1, map_location="cpu", weights_only=False)
ckpt2 = torch.load(path2, map_location="cpu", weights_only=False)

model1 = ckpt1["model"]
model2 = ckpt2["model"]

# Compare model state dicts
state_dict1 = model1.state_dict()
state_dict2 = model2.state_dict()

if state_dict1.keys() != state_dict2.keys():
    print("State dict keys are different!")
else:
    all_equal = True
    different_keys = []
    for k in state_dict1.keys():
        if not torch.equal(state_dict1[k], state_dict2[k]):
            all_equal = False
            different_keys.append(k)
            
    if all_equal:
        print("YES! The neural network weights are EXACTLY the same!")
    else:
        print("NO! The neural network weights are different.")
        print(f"Number of different keys: {len(different_keys)} out of {len(state_dict1)}")
        if different_keys:
            print("First different key sample:", different_keys[0])
