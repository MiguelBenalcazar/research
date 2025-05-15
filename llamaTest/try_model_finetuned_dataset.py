"""
https://huggingface.co/docs/transformers/en/main_classes/text_generation
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from transformers import AutoTokenizer, LlamaForCausalLM
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from functools import partial
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm  # Optional: for progress bar
import json



def get_args():
    parser = argparse.ArgumentParser(description="Finetune LLaMA")

    parser.add_argument("--path", type=str, default="/home/home/Desktop/research/saved_models/LlamaFinetuneTestMalky/LlamaFinetune/final_model/", help="Path where model was saved.")
    parser.add_argument("--quantization", type=bool, default=True, help="Enabling quatization 8 bit")
    parser.add_argument("--device", type=int, default=1, help="Choose a device to run.")

    parser.add_argument("--max_new_tokens", type=int, default=512, help=" The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.")
    parser.add_argument("--temperature", type=float, default=0.1, help=" The value used to module the next token probabilities.")
    parser.add_argument("--top_p", type=float, default=0.9, help="If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.")
    
    parser.add_argument("--do_sample", type=bool, default=True, help="Whether or not to use sampling ; use greedy decoding otherwise.")

    parser.add_argument("--datasetName", type=str, default="FreedomIntelligence/medical-o1-reasoning-SFT", help="Set dataset name.")
    parser.add_argument("--max_length", type=int, default=512, help="Set dataset name.")

    parser.add_argument("--output_file", type=str, default="test", help="Set name of the file to save data.")
    return parser.parse_args()


class ClassDataset(Dataset):
    def __init__(self, datasetName:str, lang:str = 'en', tokenizer = None, max_length: int =512, split: str = 'train'):
        ds = load_dataset(datasetName, lang)
        all_data = ds['train']
        
        # Use Huggingface's built-in split
        dataset_split = all_data.train_test_split(test_size=0.2, seed=42)
        train_data = dataset_split['train']
        val_data = dataset_split['test']

        if split == 'train':
            self.data = train_data
        elif split == 'validation':
            self.data = val_data
        else:
            raise ValueError(f"Unknown split: {split}")


        # self.data = ds['train']
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.create_tokens_text(self.data[index]) 
    

    def create_tokens_text(self, data_text):
            conversation = (
                f"<|user|>\n{data_text['Question']}\n"
                f"<|assistant|>"
            )

            # Append EOS token to the conversation
            conversation += self.tokenizer.eos_token

            # Tokenize
            encodings = self.tokenizer(conversation, 
                            truncation=True,
                            max_length=self.max_length,
                            # padding="max_length",
                            return_tensors="pt"
                        )
      
            return {
                'conversation':conversation,
                "encodings":encodings
            }



def main():
    args = get_args()
    
    model_path = args.path
    save_file = f"./results/{args.output_file}.json"

    if model_path is None:
        HOME_PATH = os.path.dirname(os.getcwd())
        model_path = os.path.join(HOME_PATH, "saved_models", "LlamaFinetuneTestMalky","LlamaFinetune","final_model")
        assert os.path.exists(model_path), f"Error, no path {model_path}"

    print(f"Model in folowing path was loaded: {model_path}")

    if args.device is not None:
        device = f"cuda:{args.device}"
    else:
        device = args.device
 

    # # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path= model_path,
            # quantization_config={"load_in_8bit": args.quantization},  # Pass the quantization config here
            torch_dtype=torch.float16,
            device_map=device,  # You can uncomment this if you want automatic device allocation
            ignore_mismatched_sizes=True
        )  

    # # 3. Resize embeddings
    model.resize_token_embeddings(len(tokenizer))

    datasetVal = ClassDataset(
        datasetName = args.datasetName, 
        tokenizer=tokenizer, 
        split="validation", 
        max_length=args.max_length
    )

    data_total_validation = []

    for idx, data in enumerate(tqdm(datasetVal, desc="Validating")):
        
        inputs = data['encodings'].to(model.device)
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.do_sample,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        result = {
            'question': data['conversation'],
            'answer': generated_text
        }
        
        data_total_validation.append(result)

        

        # Save after each iteration
        
        with open(save_file, "w", encoding="utf-8") as f:
            json.dump(data_total_validation, f, ensure_ascii=False, indent=2)


    




if __name__ == "__main__":
    main()