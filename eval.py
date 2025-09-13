from searchdet_pipeline.core.detector import SearchDetDetector
import cv2
import os
import numpy as np

detector = SearchDetDetector()

reference_masks_dir = "input/reference_masks"  
positive_dir="examples/positive"
negative_dir="examples/negative"
ground_truth="ground_truth"
script_dir = os.path.dirname(os.path.abspath(__file__))

#QUESTION: что такое IoU? Правильно я понимаю,что это фактически равни двух бинаризованных массивов путем вычисления пересечения и наложения и позже их деления?
# metrics variables
all_ious = [] # to mean IoU
total_predictions = 0 # to mean IoU


pos_by_class, neg_imgs = detector.read_reference_images(
    positive_dir=positive_dir,
    negative_dir=negative_dir
)
detector.set_references(pos_by_class, neg_imgs)

for image_file in os.listdir(reference_masks_dir):
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(reference_masks_dir, image_file)
        image_np = detector.read_input_img(image_path)
        result = detector.find_present_elements(image_np)
        
        for i, elements in enumerate(result.get('found_elements')):
            original_image = image_np.copy()
            
            if len(original_image.shape) == 2:
                original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
            
            mask_data = elements['mask']['segmentation']
            confidence = elements['confidence']
            bbox = elements['bbox']
            
            colored_mask = np.zeros_like(original_image, dtype=np.uint8)
            mask_resized = cv2.resize(mask_data.astype(np.uint8), (original_image.shape[1], original_image.shape[0]))
            colored_mask[mask_resized > 0] = [255, 0, 0]
            
            overlay = cv2.addWeighted(original_image, 0.7, colored_mask, 0.3, 0)
            output_filename = f"output_{os.path.splitext(image_file)[0]}_{i}.png"
            cv2.imwrite(os.path.join(script_dir, output_filename), overlay)


            #TODO : add metrics

g


def calculate_iou(prediction_mask, groundtruth_mask):
    prediction_binary = (pred_mask > 0).astype(np.uint8) #QUESTION : Насколько правильно тут применять бинаризацию?ведь маски в gt могут приходить в формате 0/255 ,а из prediction_mask они могут приходит от 0/1 . Поэтому я пришел к выводу,что бинаризация нужна.
    groundtruth_binary = (gt_mask > 0).astype(np.uint8)
    intersection = np.logical_and(prediction_binary, groundtruth_binary).sum()
    union = np.logical_or(prediction_binary, groundtruth_binary).sum()
    return intersection / union if union > 0 else 0

def calculate_map_multiple_detections(detections, ground_truth_mask, iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]):
    #QUESTION:Правильно я понимаю,что массив iou_thresholds нужен для вчисления mAP@[0.5:0.95]?
    if not detections:
        return 0.0
    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
    aps = []
    for iou_thresh in iou_thresholds:
        precisions = []
        recalls = []
        tp = 0  
        fp = 0  
        gt_matched = False 
        for detection in detections:
            pred_mask = detection["mask"]
            iou = calculate_iou(pred_mask, ground_truth_mask)
            if iou >= iou_thresh and not gt_matched:
                tp += 1
                gt_matched = True
            else:
                fp += 1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / 1  
            precisions.append(precision)
            recalls.append(recall)
        ap = 0.0
        for i in range(len(recalls)):
            if i == 0:
                recall_delta = recalls[i]
            else:
                recall_delta = recalls[i] - recalls[i-1]
            ap += precisions[i] * recall_delta
        aps.append(ap)
    return np.mean(aps)