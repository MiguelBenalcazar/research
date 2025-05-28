import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Specify the GPU you want to use
import torch
from transformers import AutoImageProcessor, AutoModel
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
# import argparse
import pickle
from tqdm import tqdm
import json
from datetime import datetime


DATASET_NAME = "COCOStuff"#"LVIS"#"Mapillary"#"COCOStuff"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Model 
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small', use_fast = True)
model = AutoModel.from_pretrained('facebook/dinov2-small').to(device)

def get_gpu_information():
    if torch.cuda.is_available():
        gpu_idx = torch.cuda.current_device()
        print(f"Device: {torch.cuda.get_device_name(gpu_idx)}")
        print(f"Total Memory: {torch.cuda.get_device_properties(gpu_idx).total_memory / (1024 ** 3):.2f} GB")
        print(f"Allocated Memory: {torch.cuda.memory_allocated(gpu_idx) / (1024 ** 3):.2f} GB")
        print(f"Reserved Memory: {torch.cuda.memory_reserved(gpu_idx) / (1024 ** 3):.2f} GB")
    else:
        print("CUDA not available.")



class ClassDataset(Dataset):
    def __init__(self, dataset_name:str = "COCOStuff"):
        ds = load_dataset("nvidia/describe-anything-dataset", dataset_name)
        self.data = ds["train"]
        self.dataset_name = dataset_name
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        extracted_data = self.process_information(self.data[index])
        extracted_data["information_id"] = f"{self.dataset_name}_{index}"

        return extracted_data
    
    def process_information(self, data):
        img_id, image_source=  self.extract_pickle_data(data)
        img = data['jpg']
        inputs = processor(images=img, return_tensors="pt")

        return {
            "img_id":img_id,
            "image_source":image_source,
            "image": inputs
        }


    def extract_pickle_data(self, data):
        pickle_raw_data = data['pickle']
        pickle_data = pickle.loads(pickle_raw_data)

        pickle_data_extraction= pickle_data[0] if len(pickle_data)>0 else pickle_data
        return pickle_data_extraction['img_id'], pickle_data_extraction['image']


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  
    get_gpu_information()
    
    SAVE_PATH = f"./dataset/{timestamp}_{DATASET_NAME}_train_visionModel_embeddings.jsonl"
    

    datasetTrain = ClassDataset(dataset_name=DATASET_NAME)

    # with torch.no_grad():
    #     for batch in tqdm(datasetTrain, desc="Processing", total=len(datasetTrain)):
    #         inputs = batch['image'].to(device)
    #         outputs = model(**inputs)

    #         pooler_output = outputs.pooler_output.detach().to("cpu").squeeze(0).tolist()
    #         del batch['image']
    #         batch['pooler_output'] =  pooler_output
    #         print(batch)
    #         break




    with open(SAVE_PATH, "a", encoding="utf-8") as f:
        with torch.no_grad():
            for batch in tqdm(datasetTrain, desc="Processing", total=len(datasetTrain)):
                inputs = batch['image'].to(device)
                outputs = model(**inputs)

                pooler_output = outputs.pooler_output.detach().to("cpu").squeeze(0).tolist()
                del batch['image']
                batch['pooler_output'] =  pooler_output

                json.dump(batch, f)
                f.write("\n")  # Save each example on a new line
            
                # break
    


if __name__ == "__main__":
    main()