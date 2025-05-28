from datasets import load_dataset
import subprocess
import sys
import numpy as np
from PIL import Image
from pycocotools import mask 
import json
# Image Detection
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES
import supervision as sv


DATASET_NAME = "COCOStuff"
NEW_WIDTH = 224
NEW_HEIGHT = 224

# https://huggingface.co/datasets/nvidia/describe-anything-dataset/viewer/COCOStuff/train?views%5B%5D=cocostuff

# Login using e.g. `huggingface-cli login` to access this dataset
# ds = load_dataset("nvidia/describe-anything-dataset", "LVIS")
# ds = load_dataset("nvidia/describe-anything-dataset", "Mapillary")
ds = load_dataset("nvidia/describe-anything-dataset", DATASET_NAME)

def extract_data(data):
    # try:
    #     import pickle
    # except ImportError:
    #     subprocess.check_call([sys.executable, "-m", "pip", "install", "pickle"])
    #     import pickle

    model = RFDETRBase(device="cuda") 
    
    img = data['jpg']
    print(f"SIZE: {img.size}")
    dert_detection(model, img)

   


    # pickle_raw_data = data['pickle']
    # pickle_data = pickle.loads(pickle_raw_data)
    
    # if len(pickle_data)>1:
    #     for i in pickle_data:
    #         # print(i)
    #         convert_image2numpy(img, i)

    # else:
    #     # print(pickle_data)
    #     convert_image2numpy(img, pickle_data)

def change_size(img, box):
    width, height = img.size
    scale_x = NEW_WIDTH / width
    scale_y = NEW_HEIGHT / height
    box[[0, 2]] *= scale_x
    box[[1, 3]] *= scale_y
    return box


def extract_spatial_relationships(detections):
    """
    Extract basic spatial relationships between detected objects
    """
    print(detections)
    relationships = []
    objects = [(box, COCO_CLASSES[class_id]) for box, class_id in zip(detections.xyxy, detections.class_id)]
    
    # Compare each pair of objects
    for i, (box1, label1) in enumerate(objects):
        x1_1, y1_1, x2_1, y2_1 = box1
        for j, (box2, label2) in enumerate(objects[i+1:], i+1):
            x1_2, y1_2, x2_2, y2_2 = box2
            
            # Determine horizontal relationship
            if x2_1 < x1_2:
                relationships.append(f"{label1} is to the left of {label2}")
            elif x2_2 < x1_1:
                relationships.append(f"{label1} is to the right of {label2}")
                
            # Determine vertical relationship
            if y2_1 < y1_2:
                relationships.append(f"{label1} is above {label2}")
            elif y2_2 < y1_1:
                relationships.append(f"{label1} is below {label2}")
                
    return relationships



def dert_detection(model, image):
    detections = model.predict(image, threshold=0.6)
    print(extract_spatial_relationships(detections))

    # txt_base = "In the image, we can see the following objects: \n"
    # json_output = {"detected_objects": []}

    # for box, class_id in zip(detections.xyxy, detections.class_id):
    #     label = COCO_CLASSES[class_id]
    #     change_size(image, box)
    #     x1, y1, x2, y2 = map(lambda x: round(float(x), 2), box)
        
    #     json_output["detected_objects"].append({
    #         "label": label,
    #         "bounding_box": {
    #             "x1": x1,
    #             "y1": y1,
    #             "x2": x2,
    #             "y2": y2
    #         }
    #     })

    # print(json.dumps(json_output, indent=2))
    # print(json_output)
        
    #     # print(f"A {label} is detected at (x1={x1}, y1={y1}, x2={x2}, y2={y2}).")



    # # for i in detections:
    # #     txt_base+= f"A {COCO_CLASSES[i[3]]} is at ()"
    
    # print(txt_base)
    

    
    # Create label strings
    labels = [
        f"{COCO_CLASSES[class_id]} {confidence:.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
    ]

    # Annotate and display the image
    annotated_image = image.copy()#.resize((NEW_WIDTH, NEW_HEIGHT))
    annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
    annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

    sv.plot_image(annotated_image)


# def convert_image2numpy(img, data):
#     assert data is not None, "Data is empty"

#     # img  = data['jpg']
#     mask_rle = data['mask_rle']
#     assert isinstance(img, Image.Image), "Input must be a PIL Image"
#     decoded_mask = mask.decode(mask_rle)
#     # mask_img = Image.fromarray(decoded_mask * 255).convert("RGB")
#     # blended = Image.blend(img, mask_img, alpha=0.5)
#     # blended.show()
#     masked_img = apply_binary_mask(img, decoded_mask)
#     masked_img.show()
   
    


# def apply_binary_mask(img, mask):
#     """
#     Applies a binary mask to an image.
#     1 = keep pixel, 0 = transparent.
#     """
#     # Ensure image is RGBA
#     img = img.convert("RGBA")

#     # Convert mask to same size and 0-255 alpha channel
#     mask_resized = Image.fromarray((mask * 255).astype(np.uint8))#.resize(img.size)

#     # Extract RGB and apply new alpha from mask
#     r, g, b, _ = img.split()
#     masked_img = Image.merge("RGBA", (r, g, b, mask_resized))

#     return masked_img


def main():
    data = ds['train']
    selection = data[1535]
    # print(selection)


    extract_data(selection)

 





if __name__ == "__main__":
    main()