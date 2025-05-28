from datasets import load_dataset

import json
# Image Detection

import pickle

import numpy as np

import ollama
from io import BytesIO
import base64
from tqdm import tqdm
from datetime import datetime


DATASET_NAME = "COCOStuff"#"LVIS"#"Mapillary"#"COCOStuff"
NEW_WIDTH = 224
NEW_HEIGHT = 224

# https://huggingface.co/datasets/nvidia/describe-anything-dataset/viewer/COCOStuff/train?views%5B%5D=cocostuff

# Login using e.g. `huggingface-cli login` to access this dataset
# ds = load_dataset("nvidia/describe-anything-dataset", "LVIS")
# ds = load_dataset("nvidia/describe-anything-dataset", "Mapillary")
ds = load_dataset("nvidia/describe-anything-dataset", DATASET_NAME)


def extract_data(data, idx):
    img_id, image_source=  extract_pickle_data(data)
    img = data['jpg']
    prompt = "What do you see in the image:"
    response = vision_prompt(img, prompt, model="llama3.2-vision")
    return {
        "img_id":img_id,
        "summary_text":response,
        "image_source":image_source,
        'information_id': f"{DATASET_NAME}_{idx}"
    }
    
def extract_pickle_data(data):
    pickle_raw_data = data['pickle']
    pickle_data = pickle.loads(pickle_raw_data)

    pickle_data_extraction= pickle_data[0] if len(pickle_data)>0 else pickle_data
    return pickle_data_extraction['img_id'], pickle_data_extraction['image']

def vision_prompt(img, prompt, model="llama3.2-vision"):
    base64_img = pil_to_base64(img)
    response = ollama.chat(
        model=model,
        messages=[{
            'role': 'user',
            'content': prompt,
            'images': [base64_img]
        }],
    options={
        "temperature": 0.2,
        "top_p": 0.9,
        "repeat_penalty": 1.1
        }
    )

    return response["message"]["content"]

def pil_to_base64(pil_img, format='JPEG'):
    buf = BytesIO()
    pil_img.save(buf, format=format)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def main():
    data = ds['train']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    responses_array = []
    SAVE_PATH = f"./dataset/{timestamp}_{DATASET_NAME}_train_VLM.jsonl"

    # with open(SAVE_PATH, "w", encoding="utf-8") as f:
    #     for idx, information in enumerate(data):
    #         detection = extract_data(information, idx)  # Avoid storing all in list
    #         json.dump(detection, f)
    #         f.write("\n")  # One line per example


    ## ORIGINAL
    # with open(SAVE_PATH, "w", encoding="utf-8") as f:
    #     for idx, information in enumerate(tqdm(data, desc="Saving detections")):
    #         detection = extract_data(information, idx)
    #         json.dump(detection, f)
    #         f.write("\n")  # Save each example on a new line

    # print(f"Saved processed dataset to {SAVE_PATH} (streamed to avoid memory issues)")



    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        for idx, information in enumerate(tqdm(data, desc="Saving detections")):
            if 6710 <= idx <= 25713:
                detection = extract_data(information, idx)
                json.dump(detection, f)
                f.write("\n")  # Save each example on a new line
            
           
            

    print(f"Saved processed dataset to {SAVE_PATH} (streamed to avoid memory issues)")




    # for idx, information in enumerate(data):
    #     selection = information
    #     detection = extract_data(selection, idx)
    #     print(detection)

    #     if idx ==2:
    #         break
    # #     responses_array.append(detection)

    # # data_save = {
    # #     "dataset_name":DATASET_NAME,
    # #     "train":responses_array

    # # }



 

if __name__ == "__main__":
    main()