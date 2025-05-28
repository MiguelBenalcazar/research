import os
import json
from tqdm import tqdm


def files_path_check():
    files = ["COCOStuff_train", "COCOStuff_VisionModel_embeddings", "COCOStuff_VLM"]
    current_path = os.getcwd()
    parent_path = os.path.dirname(current_path)
    
    path_files =[os.path.join(parent_path, "data", "dataset_VLM", f"{file}.jsonl") for file in files]
    
    for path_file in path_files:
        assert os.path.exists(path_file) == True, f"File {path_file} does not exist. Please check the path."
    
    return path_files


def jsonl_generator(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def read_jsonl():
    path_files = files_path_check()


    current_path = os.getcwd()
    parent_path = os.path.dirname(current_path)

    SAVE_PATH = os.path.join(parent_path, "data", "dataset_VLM", "VLM_training_dataset.jsonl")

    with open(SAVE_PATH, "a", encoding="utf-8") as f:
        for item1, item2, item3 in tqdm(
            zip(jsonl_generator(path_files[0]), jsonl_generator(path_files[1]), jsonl_generator(path_files[2])),
            desc="Merging JSONL files"
        ):
            if item1["image_source"] == item2["image_source"] == item3["image_source"]:
                detections = [{k: v for k, v in i.items() if k != "text"} for i in item1["detections"]]

                data = {
                    "information_id": item1["information_id"],
                    "image_source": item1["image_source"],
                    "image_id": item1["img_id"],
                    "detections": detections,
                    "pooler_output": item2["pooler_output"],
                    "summary_text": item3["summary_text"]
                }

                json.dump(data, f)
                f.write("\n")

    # with open(SAVE_PATH, "a", encoding="utf-8") as f:
    #     for item1, item2, item3 in zip(jsonl_generator(path_files[0]), jsonl_generator(path_files[1]),jsonl_generator(path_files[2])):
    #         if item1["image_source"] == item2["image_source"] and item1["image_source"] == item3["image_source"] and item2["image_source"] == item3["image_source"]:
    #             detections = [{k: v for k, v in i.items() if k != "text"} for i in item1["detections"]]


    #             data = {
    #                 "information_id": item1["information_id"],
    #                 "image_source": item1["image_source"],
    #                 "image_id": item1["img_id"],
    #                 "detections": detections,
    #                 "pooler_output": item2["pooler_output"],
    #                 "summary_text": item3["summary_text"]
    #             }

    #             json.dump(data, f)
    #             f.write("\n")  # Save each example on a new line

    #             break
     
            
    # data1 = jsonl_generator(path_files[0])
    # print(len(data1))

    # for item1, item2, item3 in zip(jsonl_generator(path_files[0]), jsonl_generator(path_files[1]),jsonl_generator(path_files[2])):
    #     if item1["image_source"] == item2["image_source"] and item1["image_source"] == item3["image_source"] and item2["image_source"] == item3["image_source"]:
    #         detections = [{k: v for k, v in i.items() if k != "text"} for i in item1["detections"]]


    #         data = {
    #             "information_id": item1["information_id"],
    #             "image_source": item1["image_source"],
    #             "image_id": item1["img_id"],
    #             "detections": detections,
    #             "pooler_output": item2["pooler_output"],
    #             "summary_text": item3["summary_text"]
    #         }
           

    #     break

def main():
    read_jsonl()
    
if __name__ == "__main__":
    main()