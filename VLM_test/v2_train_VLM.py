import argparse
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


import json
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

from transformers import AutoTokenizer, PreTrainedTokenizer, LlamaForCausalLM, LlamaTokenizer, default_data_collator, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import random
import logging
from typing import List, Optional, Tuple
from functools import partial
from torch.nn.utils.rnn import pad_sequence


from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PeftModel



from evaluate import load
from torchmetrics.text import BLEUScore, ROUGEScore


from pytorch_lightning.strategies import DeepSpeedStrategy

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

image_prompts = [
    "Describe the image.",
    "What do you see in the image?",
    "Provide a summary of the image.",
    "Give an overview of the visual scene.",
    "What is happening in the picture?",
    "Detail everything you can observe in the image.",
    "List the objects and their relationships.",
    "Explain the context or setting of the image.",
    "Describe the scene in detail.",
    "Break down the image contents element by element.",
    "Tell me what's in the picture.",
    "What do you notice in this photo?",
    "Can you talk about what’s shown here?",
    "What stands out to you in this image?",
    "Generate a caption for this image.",
    "Extract visual entities from the image.",
    "Explain the image content step by step.",
    "Summarize the visual content in natural language.",
    "Convert this image into a descriptive paragraph.",
    "Identify and describe the main elements in the image.",
    "Describe the image as if for someone who cannot see it.",
    "Provide a factual description of this image.",
    "What objects, actions, and locations are present in the image?"
]


def get_args():
    parser = argparse.ArgumentParser(description="Finetune LLaMA")

    parser.add_argument("--modelName", type=str, default='Llama3.2-1B-Instruct-hf', help="Set model name.")
    parser.add_argument("--projectName", type=str, default='VLM_test', help="Set project name.")
    parser.add_argument("--mode", type=str, default='train', help="Set mode: train, finetune, test.")
    parser.add_argument("--name_test", type=str, default='VLM_test', help="Set name of the test.")
    
    

    parser.add_argument("--path_dataset", type=str, default="/home/home/Desktop/research/data/dataset_VLM/VLM_training_dataset.jsonl", help="Path where the dataset is saved.")
    parser.add_argument("--seed", type=int, default=42, help="Set seed for reproducibility.")
    parser.add_argument("--batch_size", type=int, default=4, help="Set batch size.")
    parser.add_argument("--num_workers", type=int, default=4, help="Set number of workers.")


    parser.add_argument("--load_in_8bit", type=bool, default=True, help="Load model in 8-bit mode.")
    parser.add_argument("--use_lora", type=bool, default=True, help="Use LoRA for training.")

    # LoRA parameters
    parser.add_argument("--lora_r", type=int, default=4, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha scaling")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument("--lora_target_modules", type=str, nargs="+",  default= None, help="List of target modules for LoRA")

    parser.add_argument("--max_length", type=int, default=512, help="Set dataset name.")
    parser.add_argument("--max_epochs", type=int, default=20, help="Set number of epochs.")
    parser.add_argument("--warmup_steps", type=float, default=0.25, help="Set warmup steps.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Set learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Set weight decay.")
    parser.add_argument("--patience", type=int, default=2, help="Set patience for early stopping.")
    parser.add_argument("--monitor", type=str, default="loss", help="Set monitor for early stopping.")
    parser.add_argument("--max_steps", type=int, default=100, help="Set max steps.")


    parser.add_argument("--device", type=str, default='gpu', help="Set device to run the model.")
    parser.add_argument("--strategy", type=str, default='auto', help="Set strategy to run the model.")
    # parser.add_argument("--num_nodes", type=int, default=1, help="Set number of nodes.")
    parser.add_argument("--num_devices", type=int, default=1, help="Set number of devices.")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="Set gradient clip value.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32, help="Set gradient accumulation steps.")
    parser.add_argument("--precision", type=str, default="16-mixed", help="Set precision.")
    
    parser.add_argument("--image_embedding_dim", type=int, default=384, help="Set image embedding dimension.")


    
    # parser.add_argument("--output_file", type=str, default="test", help="Set name of the file to save data.")
    return parser.parse_args()


class ClassDataset(Dataset):
    def __init__(self, path_dataset:str, seed:int=42, tokenizer = None, max_length: int =512, split: str = 'train'):
        
        
        all_data = load_dataset("json", data_files=path_dataset, split="train")
         # parser.add_argument("--max_length", type=int, default=512, help="Set dataset name.")
   
        
        # Use Huggingface's built-in split
        dataset_split = all_data.train_test_split(test_size=0.2, seed=seed)


        if split == 'train':
            self.data = dataset_split['train']
        elif split == 'validation':
            self.data = dataset_split['test']
        else:
            raise ValueError(f"Unknown split: {split}")
        


        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        # image_source = self.data[idx]['image_source']
        # summary_text = self.data[idx]['summary_text']
        # pooler_output = self.data[idx]['pooler_output']
        # detections = self.data[idx]['detections']
        
        return self.create_tokens_text(self.data[idx])
    
    def create_tokens_text(self, data_text):
            
        # detection_text = ""
        # for det in data_text['detections']:
        #     label = det['label']
        #     conf = det['confidence']
        #     x1, y1, x2, y2 = det['xyxy']
        
        #     # Format with coordinates and confidence
        #     detection_text += f"- {label} ({conf:.2f} confidence) [coordinates: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})]\n"
    

        conversation = (
            f"<|user|>\n{random.choice(image_prompts)}\n"
            f"<|assistant|>\n"
            f"\nImage description:\n{data_text['summary_text']}\n"
        )

            # Append EOS token to the conversation
        conversation += self.tokenizer.eos_token

        encodings = self.tokenizer(conversation, 
                        truncation=True,
                        max_length=self.max_length,
                        padding="max_length",
                        return_tensors="pt"
                    )

        
        
        # Create labels (for causal language modeling)
        input_ids = encodings["input_ids"][0]
        attention_mask = encodings["attention_mask"][0]
        labels = input_ids.clone()

        
        # Mask labels from beginning up to (but not including) <|assistant|>
        user_token_id = self.tokenizer.convert_tokens_to_ids("<|user|>")
        assistant_token_id = self.tokenizer.convert_tokens_to_ids("<|assistant|>")

        user_start = None
        assistant_start = None
        for i, token_id in enumerate(input_ids):
            if token_id == user_token_id and user_start is None:
                user_start = i
            elif token_id == assistant_token_id and assistant_start is None:
                assistant_start = i
                break  # stop early

        if user_start is not None and assistant_start is not None:
            labels[user_start :assistant_start] = -100

        return {
            # "text":conversation,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            'image_embeddings': data_text['pooler_output']
            }

   

def cot_collate_fn(batch, tokenizer):
    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]
    image_embeddings = [ item["image_embeddings"] for item in batch]
   

    # Pad all to the longest sequence in the batch
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels,
        "image_embeddings": torch.tensor(image_embeddings)
    }


'''
----------------------------------------------------------------  DEFINE MODELS ----------------------------------------------------------------

'''

class Image_Embedding_Proj(nn.Module):
    """
  
    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, dim: int, inter_dim: int, dropout_prob: float = 0.1, mode:str="train"):
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

        if mode == "train":
            self._reset_parameters()

    def _reset_parameters(self):

        print("MIERDA DATA FINETUNED")
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.w1.weight)
        self.w1.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.w2.weight)
        self.w2.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.w3.weight)
        self.w3.bias.data.fill_(0)


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


class train_VLM_model(pl.LightningModule):
    def __init__(self, args,**kwargs):
        super().__init__()
        print("Initializing the model...")
    
        self.tokenizer = kwargs.get("tokenizer", None)

        self.save_hyperparameters(ignore=['dataloader_train', 'dataloader_val', "tokenizer"])


        # Load model and tokenizer
        if args.load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0
                )
                # quantization_config = {"load_in_8bit": True}
            except ImportError:
                logger.warning("Bit sAndBytesConfig not found, falling back to FP16")
                quantization_config = None
        else:
            quantization_config = None

      
        # Load the model with the new quantization configuration
        self.llama_model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args.model_path,
            quantization_config= quantization_config,    #{"load_in_8bit": True},  # Pass the quantization config here
            torch_dtype= torch.float16 if torch.cuda.is_available() else torch.float32,
            # device_map="auto",  # You can uncomment this if you want automatic device allocation
            ignore_mismatched_sizes=True if args.mode == "finetune" else False  # Ignore size mismatches for fine-tuning
        )    

        if self.hparams.args.num_added_tokens is not None and self.hparams.args.num_added_tokens > 0:
        # # Resize model embeddings!
            self.llama_model.resize_token_embeddings(self.hparams.args.tokenizer_size)   


        # Set up LoRA if enabled
        if args.use_lora:
            try:
                from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
                
                # Default to attention layers if not specified
                if args.lora_target_modules is None:
                    args.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
                
                # Configure LoRA
                peft_config = LoraConfig(
                    task_type = TaskType.CAUSAL_LM,
                    inference_mode = False,
                    r = args.lora_r,
                    lora_alpha = args.lora_alpha,
                    lora_dropout = args.lora_dropout,
                    target_modules = args.lora_target_modules,
                )
                
                logger.info(f"Applying LoRA with config: {peft_config}")
                self.llama_model = prepare_model_for_kbit_training(self.llama_model)
                self.llama_model = get_peft_model(self.llama_model, peft_config)
                
            except ImportError:
                logger.warning("PEFT library not found, LoRA not applied")


        # Freeze all LLaMA parameters (including LoRA adapters if any)
        self.llama_model.requires_grad_(False)

        # Optionally, confirm that only image_embedding_proj params are trainable


        self.llama_model.print_trainable_parameters()


        self.image_embedding_proj = Image_Embedding_Proj(
            dim=args.image_embedding_dim,
            inter_dim=self.llama_model.config.hidden_size,
            dropout_prob=0.3, 
            mode=self.hparams.args.mode

        )

    def forward(self, **batch):

        # Forward pass through the model
        input_ids = batch['input_ids']
        image_embedding = batch.pop("image_embeddings")
        if isinstance(image_embedding, list):
            image_embedding = torch.tensor(image_embedding).to(dtype=self.llama_model.dtype, device=self.device)


        inputs_embeds = self._prepare_inputs_embeds(input_ids, image_embedding)

        image_mask = torch.ones((batch['attention_mask'].size(0), 1), dtype=batch['attention_mask'].dtype).to(self.device)
        attention_mask = torch.cat([image_mask, batch['attention_mask']], dim=1)

        labels = batch['labels']
        ignore_token = -100  # ignore index for loss computation
        labels = torch.cat([ignore_token * torch.ones((labels.size(0), 1), dtype=labels.dtype).to(self.device), labels], dim=1)

        return self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
    
    def _prepare_inputs_embeds(self, input_ids, image_embeddings):
 
        # Helper method to prepare inputs_embeds with image projection
        inputs_embeds = self.llama_model.model.model.embed_tokens(input_ids)       
        
        projected_image_embed = self.image_embedding_proj(image_embeddings).unsqueeze(1)
        inputs_embeds = torch.cat([projected_image_embed, inputs_embeds], dim=1)
        return inputs_embeds


    def configure_optimizers(self):
        # no_decay = ["bias", "LayerNorm.weight"]
    
        # Get all parameters from the model including image_embedding_proj
        all_params = []
    
        # # Get model parameters with weight decay
        # model_params_decay = {
        #     "params": [p for n, p in self.llama_model.named_parameters() 
        #             if not any(nd in n for nd in no_decay)],
        #     "weight_decay": self.hparams.args.weight_decay,
        # }


        # all_params.append(model_params_decay)
    
        # # Get model parameters without weight decay
        # model_params_no_decay = {
        #     "params": [p for n, p in self.llama_model.named_parameters() 
        #             if any(nd in n for nd in no_decay)],
        #     "weight_decay": 0.0,
        # }
        # all_params.append(model_params_no_decay)
    
        # Add image embedding projection parameters
        image_proj_params = {
            "params": self.image_embedding_proj.parameters(),
            "weight_decay": self.hparams.args.weight_decay,
            # Optionally use a different learning rate for the adapter
            # "lr": self.hparams.args.learning_rate * 2.0  # Uncomment if you want a higher LR for adapter
        }
        all_params.append(image_proj_params)

        optimizer = torch.optim.AdamW(
            all_params, 
            lr=self.hparams.args.learning_rate
        )
    
    
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=self.hparams.args.warmup_steps,
            num_training_steps=self.hparams.args.max_steps
        )
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer, 
        #     num_warmup_steps=self.hparams.args.warmup_steps, 
        #     num_training_steps=self.hparams.args.max_steps
        # )
    
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
    
    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError


class trainVLMmodel(train_VLM_model):

    def _calculate_loss_(self, batch,  mode="train")-> Tuple[torch.Tensor, float]:
        
        outputs = self.forward(**batch)
       
        loss = outputs.loss
        self.log(f"{mode}_loss", loss, prog_bar=True, sync_dist=True)

        return {
            "loss": loss, 
        }
        
  
    
    def training_step(self, batch, batch_idx):
        metrics = self._calculate_loss_(batch=batch, mode="train")

        # Log learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", current_lr, on_step=True, on_epoch=False, sync_dist=True)

        return metrics["loss"]
    
    def validation_step(self, batch, batch_idx):
        metrics = self._calculate_loss_(batch=batch, mode="val")
        return metrics["loss"]
    

def train_finetuneModel(args, **kwargs):
  
    CHECKPOINT_PATH = kwargs.get("CHECKPOINT_PATH", None)
    dataloader_train = kwargs.get("dataloader_train", None)
    dataloader_val = kwargs.get("dataloader_val", None)
    tokenizer = kwargs.get("tokenizer", None)
  
    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = os.path.join(CHECKPOINT_PATH, args.name_test)
    os.makedirs(root_dir, exist_ok=True)


    # Configure callbacks
    checkpoint_callback = ModelCheckpoint(
        # dirpath="checkpoints",
        filename="llama-3.2-1b-lora-{epoch:02d}-{val_loss:.5f}",
        # filename="llama-3.2-1b-lora-{epoch:02d}-{val_loss:.2f}",
        save_top_k=2,
        monitor= f'val_{args.monitor}',
        mode="min",
    )

     # Configure logger
    # logger = TensorBoardLogger("logs", name="llama-3.2-1b-lora")

    early_stopping_callback = EarlyStopping(
        monitor=f'val_{args.monitor}',  # Use val_f1 or val_pr_auc depending on your goal
        patience = args.patience,  # Stop after 20 epochs without improvement
        mode = "min",  # Maximize the F1 score
        verbose = True  # Optional: Print early stopping information
    )

    trainer = pl.Trainer(
        deterministic=True,
        default_root_dir=root_dir,
        max_epochs=args.max_epochs,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches= args.gradient_accumulation_steps, 
        precision="16-mixed", 
        # logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        log_every_n_steps=10,

        accelerator=args.device,
        devices=args.num_devices,            # Automatically detect how many GPUs
        strategy=args.strategy,            # Let Lightning choose DDP or DataParallel
        profiler="simple", # ['simple', 'advanced', 'pytorch', 'xla']
        # num_nodes = 1,

        
        # fast_dev_run=True
    )

    pretrained_filename = os.path.join(
            CHECKPOINT_PATH,
            args.name_test,
            "lightning_logs",
            "version_1",
            "checkpoints",
            "llama-3.2-1b-lora-epoch=04-val_loss=0.22153.ckpt"  # Adjust this filename as needed
        )

    print(f"Pretrained filename: {pretrained_filename}")

    model_save_path = os.path.join(CHECKPOINT_PATH, args.name_test, "final_model")
    os.makedirs(model_save_path, exist_ok=True)

    if args.mode == "train":
    # Train
        model = trainVLMmodel(args, **kwargs)
        trainer.fit(model = model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val )
    
    elif args.mode =="finetune":
        
        if not os.path.exists(pretrained_filename):
            raise FileNotFoundError(f"The directory {pretrained_filename} does not exist. Please check the path.")
        
        model = trainVLMmodel.load_from_checkpoint(pretrained_filename, args=args, **kwargs)
        trainer.fit(model= model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)

    elif args.mode=="test":

        if not os.path.exists(pretrained_filename):
            raise FileNotFoundError(f"The directory {pretrained_filename} does not exist. Please check the path.")
        
        model = trainVLMmodel.load_from_checkpoint(pretrained_filename, args=args, **kwargs)
        print("Model Load")
        

    # Save the final model
    if isinstance(model.llama_model, PeftModel):
        print("Merging LoRA weights with base model...")
        model.llama_model = model.llama_model.merge_and_unload()
        print("Model Merged")

     # Optionally, save the LLaMA model and tokenizer in Hugging Face format
    model.llama_model.save_pretrained(model_save_path)
    model.llama_model.config.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"LLaMA model and tokenizer saved to {model_save_path}")


    
    
    # Save the image projection layer separately (if needed for non-Lightning loading)
    torch.save(model.image_embedding_proj.state_dict(), os.path.join(model_save_path, "image_embedding_proj.pt"))
    print(f"Image projection layer saved to {os.path.join(model_save_path, 'image_embedding_proj.pt')}")


    print(f"Fused model and tokenizer saved to: {model_save_path}")

    return model

def verify_and_count_devices():
   
    count_dev = 0
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        count_dev = torch.cuda.device_count()
        print(f"Number of GPUs: {count_dev}")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("Using CPU")
    return count_dev


def main():
    args = get_args()

    # args.num_devices = verify_and_count_devices()
    # if  args.num_devices > 1:
    #     args.strategy = "ddp"#DeepSpeedStrategy(stage=2, offload_optimizer=True)#"deepspeed"
    # else:       
    #     args.strategy = "auto"
    # print(f"Using strategy: {args.strategy} with {args.num_devices} devices.")



    torch.set_float32_matmul_precision('high')
    # Setting the seed
    pl.seed_everything(args.seed,  workers=True)
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    #  Path to the folder where the pretrained models are saved
    HOME_PATH = os.path.dirname(os.getcwd())
    CHECKPOINT_PATH = os.path.join(HOME_PATH, "saved_models", args.projectName)


    model_path = os.path.join(HOME_PATH,"Models", args.modelName)
    # Verify the path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The directory {model_path} does not exist. Please check the path.")

    args.model_path = model_path

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    # Set pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

  

    special_tokens_dict = {"additional_special_tokens": ["<|user|>", "<|assistant|>", "<|image|>", "<|projected_image_tokens|>", "<|end_image|>"]}
    args.num_added_tokens  = tokenizer.add_special_tokens(special_tokens_dict)


    print(f"Added {args.num_added_tokens } special tokens.")
    # tokenizer_size = len(tokenizer)
    args.tokenizer_size = len(tokenizer)

    train_dataset = ClassDataset(
        path_dataset=args.path_dataset, 
        tokenizer=tokenizer, 
        max_length=args.max_length, 
        split='train')
    
    validation_dataset = ClassDataset(
        path_dataset=args.path_dataset, 
        tokenizer=tokenizer, 
        max_length=args.max_length, 
        split='validation')
    
    
    dataloader_train = DataLoader(
            dataset=train_dataset,
            batch_size= args.batch_size,
            num_workers= args.num_workers,
            collate_fn=partial(cot_collate_fn, tokenizer=tokenizer),
            shuffle = True,
            drop_last=False,
            pin_memory= True
        )
    
    dataloader_val = DataLoader(
            dataset=validation_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=partial(cot_collate_fn, tokenizer=tokenizer),
            shuffle=False,
            drop_last=False,
            pin_memory=True
        )
    

    args.warmup_steps = int(args.warmup_steps * args.max_epochs)
    args.max_steps = len(dataloader_train) * args.max_epochs
    

    training_args = {
        "dataloader_train": dataloader_train,
        "dataloader_val": dataloader_val,
        "tokenizer": tokenizer,
        "CHECKPOINT_PATH":CHECKPOINT_PATH,
    }


    model = train_finetuneModel(args=args, **training_args)


        

if __name__ == "__main__":
    main()