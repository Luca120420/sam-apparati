import os 
import cv2
import torch
import numpy as np
import supervision as sv

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

HOME = os.getcwd()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT = f"{HOME}/checkpoints/sam2_hiera_base_plus.pt"
CONFIG = "sam2_hiera_b+.yaml"
IMAGE_PATH = "/home/ubuntu/MLCode/Apparati/images/AE-A5.jpg"
SAVE_PATH = "/home/ubuntu/MLCode/Apparati/annotated-images/annotated_AE-A5.jpg"

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

sam2_model = build_sam2(CONFIG, CHECKPOINT, device=DEVICE, apply_postprocessing=False)

image_bgr = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

mask_generator_2 = SAM2AutomaticMaskGenerator(
    model=sam2_model,
    points_per_side=32,           
    points_per_batch=64,          
    pred_iou_thresh=0.5,          
    stability_score_thresh=0.85,  
    stability_score_offset=0.7,
    crop_n_layers=0,              
    box_nms_thresh=0.9
)

sam2_result_2 = mask_generator_2.generate(image_rgb)

mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
detections = sv.Detections.from_sam(sam_result=sam2_result_2)

annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

# Save the annotated image to the specified path
cv2.imwrite(SAVE_PATH, annotated_image)

print(f"Annotated image saved to {SAVE_PATH}")