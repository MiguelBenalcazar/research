from datasets import load_dataset
import subprocess
import sys
import numpy as np
from PIL import Image
from pycocotools import mask 
import json
# Image Detection
from ultralytics import YOLO
import pickle

import numpy as np
from collections import Counter, defaultdict
from itertools import combinations

DATASET_NAME = "LVIS"#"Mapillary"#"COCOStuff"
NEW_WIDTH = 224
NEW_HEIGHT = 224

# https://huggingface.co/datasets/nvidia/describe-anything-dataset/viewer/COCOStuff/train?views%5B%5D=cocostuff

# Login using e.g. `huggingface-cli login` to access this dataset
# ds = load_dataset("nvidia/describe-anything-dataset", "LVIS")
# ds = load_dataset("nvidia/describe-anything-dataset", "Mapillary")
ds = load_dataset("nvidia/describe-anything-dataset", DATASET_NAME)
model = YOLO("./model/yolo11x.pt")  # pretrained YOLO11n model

def extract_data(data, idx):
    img_id, image_source=  extract_pickle_data(data)
    img = data['jpg']
    img.show()
    results = model(img, stream=True) 
    
    enumerate_results= enumerate(results)
    enumerate_results_item = next(enumerate_results)[1]

    data_extraction_image = extract_detection_information(enumerate_results_item)
    data_extraction_image['img_id'] = img_id
    data_extraction_image['image_source'] = image_source
    data_extraction_image['information_id'] = f"{DATASET_NAME}_{idx}"
    # data_extraction_image['jpg'] = img

    return data_extraction_image



def extract_pickle_data(data):
    pickle_raw_data = data['pickle']
    pickle_data = pickle.loads(pickle_raw_data)

    pickle_data_extraction= pickle_data[0] if len(pickle_data)>0 else pickle_data
    return pickle_data_extraction['img_id'], pickle_data_extraction['image']



def extract_detection_information(result):

    image_height, image_width = result.orig_shape 
    max_img_distance = np.sqrt(image_width**2 + image_height**2)
    boxes = result.boxes
    data_boxes = []
    class_counts = Counter()
    class_centers = defaultdict(list)

    for box in boxes:
        box_class_number = int(box.cls.item())
        box_class = result.names[box_class_number]
        class_counts[box_class] += 1
        confidence = round(box.conf.item(), 3)
        xyxy = box.xyxy[0].detach().cpu().numpy()

        # Compute center of box
        center_x = (xyxy[0] + xyxy[2]) / 2
        center_y = (xyxy[1] + xyxy[3]) / 2
        class_centers[box_class].append((center_x, center_y))

        data_boxes.append({
            "label": box_class,
            "confidence": confidence,
            "xyxy": xyxy.tolist(),
            "text": (
                f"A {box_class} is detected in the image with {confidence * 100:.2f}% confidence, "
                f"located from (x: {xyxy[0]:.2f}, y: {xyxy[1]:.2f}) to (x: {xyxy[2]:.2f}, y: {xyxy[3]:.2f})."
            )
        })

    # Generate summary with spatial relationship analysis
    summary_parts = []
    for cls, count in class_counts.items():
        if count > 1:
            centers = class_centers[cls]
            distances = []
            for (x1, y1), (x2, y2) in combinations(centers, 2):
                dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                distances.append(dist)

            avg_dist = np.mean(distances)
            dist_percent = (avg_dist / max_img_distance) * 100

            spatial_description = ""
            if dist_percent < 25:
                spatial_description = " They are close together."
            elif dist_percent < 60:
                spatial_description = " They are moderately spaced."
            else:
                spatial_description = " They are widely spread out."

            summary_parts.append(
                f"There are {count} {cls}s in the image." + spatial_description
            )
        else:
            summary_parts.append(f"There is {count} {cls} in the image.")

    summary_text = " ".join(summary_parts)
    return {
        "summary_text":summary_text,
        "detections": data_boxes
    }



def change_size(img, box):
    width, height = img.size
    scale_x = NEW_WIDTH / width
    scale_y = NEW_HEIGHT / height
    box[[0, 2]] *= scale_x
    box[[1, 3]] *= scale_y
    return box





def main():
    data = ds['train']
    responses_array = []
    SAVE_PATH = f"{DATASET_NAME}_train.jsonl"

    # with open(SAVE_PATH, "w", encoding="utf-8") as f:
    #     for idx, information in enumerate(data):
    #         detection = extract_data(information, idx)  # Avoid storing all in list
    #         json.dump(detection, f)
    #         f.write("\n")  # One line per example

    # print(f"Saved processed dataset to {SAVE_PATH} (streamed to avoid memory issues)")

    for idx, information in enumerate(data):
        selection = information
        detection = extract_data(selection, idx)
        print(detection)

        if idx ==2:
            break
    #     responses_array.append(detection)

    # data_save = {
    #     "dataset_name":DATASET_NAME,
    #     "train":responses_array

    # }

 





if __name__ == "__main__":
    main()