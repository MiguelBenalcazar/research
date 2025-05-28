from transformers import AutoTokenizer, PreTrainedTokenizer, AutoImageProcessor, AutoModel
import torch
import os
import json


class CustomMultimodalProcessor:
    def __init__(self, tokenizer_path: str=None, image_model:str = 'facebook/dinov2-small'):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        self.image_model_name = image_model
        self.image_model = AutoModel.from_pretrained(image_model)
        self.image_processor = AutoImageProcessor.from_pretrained(image_model)

        # Set image model to eval mode for inference
        self.image_model.eval()

    def __call__(self, text, image=None, max_length=512, padding="max_length", truncation=True):

        def conversation_message(text, image):
            if image is None:
                return (
                    f"<|user|>\n{text}\n"
                    # f"<|image|><|end_image|>\n"
                    f"<|assistant|>\n"
                )
            else:

                return (
                    f"<|user|>\n{text}\n"
                    f"<|assistant|>\n"
                )

                # return (
                #     f"<|user|>\n{text}\n"
                #     f"<|image|><|end_image|>\n"
                #     f"<|assistant|>\n"
                # )




        # Process text  
        text = conversation_message(text, image)
        # text += self.tokenizer.eos_token
        print(f"Text after processing: {text}")
        
        # Tokenize text
        inputs = self.tokenizer(text, return_tensors="pt")
        # inputs = self.tokenizer(
        #     text,
        #     padding=padding,
        #     truncation=truncation,
        #     max_length=max_length,
        #     return_tensors="pt"
        # )

        # Get the device of the image model
        device = next(self.image_model.parameters()).device

        # Process image if provided
        image_embedding = None
        if image is not None:
            # Handle different image input types
            if isinstance(image, str):
                from PIL import Image
                image = Image.open(image).convert('RGB')
            
            # Process image
            image_inputs = self.image_processor(images=image, return_tensors="pt").to(device) 
            
            with torch.no_grad():
                image_outputs = self.image_model(**image_inputs)
                
                # Handle different model output formats
                if hasattr(image_outputs, 'pooler_output') and image_outputs.pooler_output is not None:
                    image_embedding = image_outputs.pooler_output.squeeze(0)
                elif hasattr(image_outputs, 'last_hidden_state'):
                    # For models without pooler_output, use mean pooling
                    image_embedding = image_outputs.last_hidden_state.mean(dim=1).squeeze(0)
                else:
                    # Fallback: use the first output if it's a tuple
                    if isinstance(image_outputs, tuple):
                        image_embedding = image_outputs[0].mean(dim=1).squeeze(0)
                    else:
                        raise ValueError(f"Unsupported image model output format: {type(image_outputs)}")
                    

        # print(f"Image model is on device: {device}")
        # print(f"Image tensor is on device: {image_inputs['pixel_values'].device}")
        
        return inputs, image_embedding

    
    def save_pretrained(self, save_directory):
        """Save the processor components to a directory"""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_directory)
        
        # Save image model and processor
        image_dir = os.path.join(save_directory, "image_model")
        os.makedirs(image_dir, exist_ok=True)
        self.image_model.save_pretrained(image_dir)
        self.image_processor.save_pretrained(image_dir)
        
        # Save processor config
        config = {
            "processor_type": "CustomMultimodalProcessor",
            "image_model_name": self.image_model_name,
            "image_embedding_dim": self._get_image_embedding_dim()
        }
        
        with open(os.path.join(save_directory, "processor_config.json"), "w") as f:
            json.dump(config, f, indent=2)


    def _get_image_embedding_dim(self):
        """Get the dimension of image embeddings"""
        # Create a dummy image to get embedding dimension
        dummy_image = torch.zeros(3, 224, 224)  # Standard image size
        dummy_inputs = {"pixel_values": dummy_image.unsqueeze(0)}
        
        with torch.no_grad():
            outputs = self.image_model(**dummy_inputs)
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                return outputs.pooler_output.shape[-1]
            elif hasattr(outputs, 'last_hidden_state'):
                return outputs.last_hidden_state.shape[-1]
            else:
                if isinstance(outputs, tuple):
                    return outputs[0].shape[-1]
                else:
                    raise ValueError("Cannot determine image embedding dimension")
                

    @classmethod
    def from_pretrained(cls, load_directory):
        """Load the processor from a directory"""
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(load_directory)
        
        # Load config to get image model name
        config_path = os.path.join(load_directory, "processor_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            image_model_name = config.get("image_model_name", "facebook/dinov2-small")
        else:
            # Fallback to default
            image_model_name = "facebook/dinov2-small"
            
        # Check if image model is saved locally
        image_dir = os.path.join(load_directory, "image_model")
        if os.path.exists(image_dir):
            # Load from local directory
            instance = cls.__new__(cls)
            instance.tokenizer = tokenizer
            instance.image_model_name = image_model_name
            instance.image_model = AutoModel.from_pretrained(image_dir)
            instance.image_processor = AutoImageProcessor.from_pretrained(image_dir)
            instance.image_model.eval()
            return instance
        else:
            # Load from hub
            return cls(tokenizer, image_model_name)
        
    
    def get_image_embedding_dim(self):
        """Public method to get image embedding dimension"""
        return self._get_image_embedding_dim()

    def to(self, device):
        """Move image model to specified device"""
        self.image_model = self.image_model.to(device)
        return self
    
