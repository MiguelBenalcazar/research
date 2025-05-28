import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from CustomMultimodalProcessor import CustomMultimodalProcessor
from transformers import LlamaForCausalLM, AutoTokenizer
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
import requests

from datasets import load_dataset

from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import random
from torch.utils.data import Dataset, DataLoader
import pickle

HOME_PATH = os.path.dirname(os.getcwd())
# Load the processor from the saved directory
# processor = CustomMultimodalProcessor.from_pretrained("./processor/VLM_processor")
from transformers import AutoImageProcessor, AutoModel

path_model = "/home/home/Desktop/research/saved_models/VLM_test/VLM_test/final_model/"

processor = CustomMultimodalProcessor(tokenizer_path=path_model, image_model='facebook/dinov2-small')


class Image_Embedding_Proj(nn.Module):
    """
  
    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, dim: int, inter_dim: int, dropout_prob: float = 0.1):
        """
        Initializes the Expert layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = nn.Linear(in_features=dim, out_features=inter_dim)
        self.w2 = nn.Linear(in_features=inter_dim, out_features=inter_dim)
        self.w3 = nn.Linear(in_features=dim, out_features=inter_dim)

        self.norm1 = nn.LayerNorm(inter_dim)
        self.norm2 = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(dropout_prob)  # Dropout layer with specified probability



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Expert layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert computation.
            
        """

        x1 = self.norm1(self.dropout(self.w1(x)))
        x3 = self.norm1(self.dropout(self.w3(x)))

        return self.dropout(self.w2(F.silu(x1) * x3)) #Output=W_2(SiLU(W_1x) ⊙ (W_3x))


class train_VLM_model(nn.Module):
    def __init__(self, tokenizer, path_model):
        super().__init__()
      
    
        self.tokenizer =tokenizer
               
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
        

      
        # Load the model with the new quantization configuration
        self.llama_model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=path_model,
            quantization_config= quantization_config,    #{"load_in_8bit": True},  # Pass the quantization config here
            torch_dtype= torch.float16 if torch.cuda.is_available() else torch.float32,
            # device_map="auto"  # You can uncomment this if you want automatic device allocation
        )    

   
        # Image Adapter
    

        self.image_embedding_proj = Image_Embedding_Proj(
            dim=384,
            inter_dim=self.llama_model.config.hidden_size,
            dropout_prob=0.3, 


        )
    


    def forward(self, **batch):
        # Forward pass through the model
        input_ids = batch['input_ids']
        image_embedding = batch.pop("image_embeddings")
        if isinstance(image_embedding, list):
            image_embedding = torch.tensor(image_embedding).to(dtype=self.llama_model.dtype, device=self.llama_model.device)


        inputs_embeds = self._prepare_inputs_embeds(input_ids, image_embedding)


        image_mask = torch.ones((batch['attention_mask'].size(0), 1), dtype=batch['attention_mask'].dtype).to(self.llama_model.device)
        attention_mask = torch.cat([image_mask, batch['attention_mask']], dim=1)

    
        with torch.no_grad():
            outputs = self.llama_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=512,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                return_dict_in_generate=True
            )
            return outputs.sequences  # Return the sequences tensor
 

    def _prepare_inputs_embeds(self, input_ids, image_embeddings):
        # Get token embeddings
        inputs_embeds = self.llama_model.model.embed_tokens(input_ids)

        # Project image embeddings
        projected_image_embed = self.image_embedding_proj(image_embeddings)

        # Match dtype and device to model (usually float16 for quantized models)
        projected_image_embed = projected_image_embed.to(
            dtype=inputs_embeds.dtype,  # e.g., torch.float16
            device=inputs_embeds.device
        )

        # Ensure image embedding is 3D [B, 1, D]
        if projected_image_embed.ndim == 2:
            projected_image_embed = projected_image_embed.unsqueeze(1)  # [B, 1, D]
        elif projected_image_embed.ndim == 1:
            projected_image_embed = projected_image_embed.unsqueeze(0).unsqueeze(1)  # [1, 1, D]

        # Concatenate image + token embeddings
        inputs_embeds = torch.cat([projected_image_embed, inputs_embeds], dim=1)

        return inputs_embeds





processor_img = AutoImageProcessor.from_pretrained('facebook/dinov2-small', use_fast = True)

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
        # inputs = processor_img(images=img, return_tensors="pt")

        return {
            "img_id":img_id,
            "image_source":image_source,
            "image": img
        }


    def extract_pickle_data(self, data):
        pickle_raw_data = data['pickle']
        pickle_data = pickle.loads(pickle_raw_data)

        pickle_data_extraction= pickle_data[0] if len(pickle_data)>0 else pickle_data
        return pickle_data_extraction['img_id'], pickle_data_extraction['image']


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(path_model, local_files_only=True)

    model = train_VLM_model(tokenizer=tokenizer, path_model=path_model)
    model.image_embedding_proj.load_state_dict(torch.load(os.path.join(path_model, "image_embedding_proj.pt"), map_location="cpu"))
    
    model = model.to(device)

    text = "Detail everything you can observe in the image."
    # image_path = "/path/to/your/image.jpg"
    # image = Image.open(image_path).convert("RGB")
    
    # # url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    # url = "https://news.harvard.edu/wp-content/uploads/2023/11/dog_brains_2500.png?resize=1024,576"
    # image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    # image.show()

  
    

    validation_dataset = ClassDataset(dataset_name="COCOStuff")
    
    # image = None
    # for i in validation_dataset:
    #     image = i['image']
    #     break

    sample = random.choice(validation_dataset)
    image = sample['image']
    image.show()

    # Preprocess text and image
    inputs, image_embedding = processor(text, image=image)

    inputs_total = {
        "input_ids": inputs["input_ids"].to(device),
        "attention_mask": inputs["attention_mask"].to(device),
        "image_embeddings": image_embedding.to(device),
    }

    
    
    
    # Generate output
    try:
        outputs = model.forward(**inputs_total)
        print(f"Generated output shape: {outputs.shape}")
        print(f"Generated tokens: {outputs}")
        
        # Decode the entire output first for debugging
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Full decoded output: {full_response}")
        
        # Decode only the generated tokens
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:] if outputs.shape[1] > input_length else outputs[0]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"Generated response: {response}")
        
        if not response:
            print("Warning: Empty response. Possible reasons:")
            print(f"- Output length ({outputs.shape[1]}) <= input length ({input_length})")
            print(f"- Generated tokens: {generated_tokens}")
    except Exception as e:
        print(f"Error during generation: {e}")

    

if __name__ == "__main__":  
    main()


