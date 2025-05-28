import argparse
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

import torch
import pytorch_lightning as pl

from transformers import AutoTokenizer, PreTrainedTokenizer, LlamaForCausalLM, LlamaTokenizer, default_data_collator, get_linear_schedule_with_warmup
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import random
import logging
from typing import List, Optional, Tuple
from functools import partial
from torch.nn.utils.rnn import pad_sequence

from evaluate import load
from torchmetrics.text import BLEUScore, ROUGEScore

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
    "Can you talk about what's shown here?",
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
    parser = argparse.ArgumentParser(description="Finetune LLaMA VLM")

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
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=None, help="List of target modules for LoRA")

    parser.add_argument("--max_length", type=int, default=512, help="Set max sequence length.")
    parser.add_argument("--max_epochs", type=int, default=20, help="Set number of epochs.")
    parser.add_argument("--warmup_steps", type=float, default=0.25, help="Set warmup steps.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Set learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Set weight decay.")
    parser.add_argument("--patience", type=int, default=0.2, help="Set patience for early stopping.")
    parser.add_argument("--monitor", type=str, default="loss", help="Set monitor for early stopping.")
    parser.add_argument("--max_steps", type=int, default=100, help="Set max steps.")

    parser.add_argument("--device", type=str, default='gpu', help="Set device to run the model.")
    parser.add_argument("--strategy", type=str, default='auto', help="Set strategy to run the model.")
    parser.add_argument("--num_devices", type=int, default=1, help="Set number of devices.")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="Set gradient clip value.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32, help="Set gradient accumulation steps.")
    parser.add_argument("--precision", type=str, default="16-mixed", help="Set precision.")
    
    parser.add_argument("--image_embedding_dim", type=int, default=384, help="Set image embedding dimension.")
    
    return parser.parse_args()

class ClassDataset(Dataset):
    def __init__(self, path_dataset: str, seed: int = 42, tokenizer=None, max_length: int = 512, split: str = 'train'):
        all_data = load_dataset("json", data_files=path_dataset, split="train")
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
        return self.create_tokens_text(self.data[idx])
    
    def create_tokens_text(self, data_text):
        detection_text = ""
        for det in data_text['detections']:
            label = det['label']
            conf = det['confidence']
            x1, y1, x2, y2 = det['xyxy']
            detection_text += f"- {label} ({conf:.2f} confidence) [coordinates: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})]\n"

        conversation = (
            f"<|user|>\n{random.choice(image_prompts)}\n"
            f"<|image|><|end_image|>\n"
            f"<|assistant|>\nDetection Context:\n{detection_text}\n"
            f"\n{data_text['summary_text']}\n"
        )

        conversation += self.tokenizer.eos_token

        encodings = self.tokenizer(
            conversation, 
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encodings["input_ids"][0]
        attention_mask = encodings["attention_mask"][0]
        labels = input_ids.clone()

        # Mask labels for user prompts
        assistant_positions = []
        assistant_token_id = self.tokenizer.convert_tokens_to_ids("<|assistant|>")
        for i, token_id in enumerate(input_ids):
            if token_id == assistant_token_id:
                assistant_positions.append(i)

        if assistant_positions:
            is_assistant = False
            for i in range(len(labels)):
                if i in assistant_positions:
                    is_assistant = True
                if not is_assistant:
                    labels[i] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            'image_embeddings': data_text['pooler_output']
        }

def cot_collate_fn(batch, tokenizer):
    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]
    image_embeddings = [item["image_embeddings"] for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels,
        "image_embeddings": torch.tensor(image_embeddings)
    }

class VisionLanguageModel(pl.LightningModule):
    """
    Complete Vision-Language Model with proper component separation
    """
    def __init__(self, args, **kwargs):
        super().__init__()
        print("Initializing VLM model...")
        
        self.tokenizer = kwargs.get("tokenizer", None)
        self.save_hyperparameters(ignore=['tokenizer'])
        
        # Initialize LLaMA model
        self._init_llama_model(args)
        
        # Initialize Vision Adapter
        self._init_vision_adapter(args)
        
        # Setup LoRA if enabled
        if args.use_lora:
            self._setup_lora(args)
            
        self.llama_model.print_trainable_parameters()

    def _init_llama_model(self, args):
        """Initialize the LLaMA language model"""
        if args.load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0
                )
            except ImportError:
                logger.warning("BitsAndBytesConfig not found, falling back to FP16")
                quantization_config = None
        else:
            quantization_config = None

        self.llama_model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args.model_path,
            quantization_config=quantization_config,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        # Resize embeddings if new tokens were added
        if hasattr(args, 'num_added_tokens') and args.num_added_tokens > 0:
            self.llama_model.resize_token_embeddings(args.tokenizer_size)

    def _init_vision_adapter(self, args):
        """Initialize the vision-to-language adapter"""
        self.vision_config = {
            'input_dim': args.image_embedding_dim,
            'hidden_dim': self.llama_model.config.hidden_size,
            'dropout': 0.1
        }
        
        # Simple linear projection (can be made more sophisticated)
        self.vision_adapter = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self.vision_config['input_dim'],
                out_features=self.vision_config['hidden_dim']
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.vision_config['dropout']),
            torch.nn.Linear(
                in_features=self.vision_config['hidden_dim'],
                out_features=self.vision_config['hidden_dim']
            )
        )

    def _setup_lora(self, args):
        """Setup LoRA configuration"""
        try:
            from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
            
            if args.lora_target_modules is None:
                args.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
            
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=args.lora_target_modules,
            )
            
            logger.info(f"Applying LoRA with config: {peft_config}")
            self.llama_model = prepare_model_for_kbit_training(self.llama_model)
            self.llama_model = get_peft_model(self.llama_model, peft_config)
            
        except ImportError:
            logger.warning("PEFT library not found, LoRA not applied")

    def forward(self, **batch):
        """Forward pass through the complete VLM"""
        input_ids = batch['input_ids']
        image_embeddings = batch.pop("image_embeddings")
        
        if isinstance(image_embeddings, list):
            image_embeddings = torch.tensor(image_embeddings).to(dtype=self.llama_model.dtype, device=self.device)

        inputs_embeds = self._prepare_inputs_embeds(input_ids, image_embeddings)

        return self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )

    def _prepare_inputs_embeds(self, input_ids, image_embeddings):
        """Prepare input embeddings with vision features"""
        # Get text embeddings
        inputs_embeds = self.llama_model.model.embed_tokens(input_ids)
        
        # Project vision features to language space
        projected_vision_embeds = self.vision_adapter(image_embeddings).unsqueeze(1)
        
        # Replace image tokens with projected vision features
        image_token_id = self.tokenizer.convert_tokens_to_ids("<|image|>")
        image_token_mask = (input_ids == image_token_id)
        
        inputs_embeds = inputs_embeds.clone()
        for b in range(input_ids.size(0)):
            pos = torch.nonzero(image_token_mask[b], as_tuple=False)
            if pos.numel() > 0:
                idx = pos[0].item()
                inputs_embeds[b, idx] = projected_vision_embeds[b]
                
        return inputs_embeds

    def get_vision_adapter(self):
        """Get the vision adapter component"""
        return self.vision_adapter

    def get_llama_model(self):
        """Get the LLaMA model component"""
        return self.llama_model

    def save_components_separately(self, save_dir):
        """Save vision and language components separately"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save LLaMA model
        llama_save_path = os.path.join(save_dir, "llama_model")
        os.makedirs(llama_save_path, exist_ok=True)
        self.llama_model.save_pretrained(llama_save_path)
        logger.info(f"LLaMA model saved to {llama_save_path}")
        
        # Save vision adapter
        vision_save_path = os.path.join(save_dir, "vision_adapter")
        os.makedirs(vision_save_path, exist_ok=True)
        
        # Save vision adapter state dict
        torch.save(self.vision_adapter.state_dict(), os.path.join(vision_save_path, "vision_adapter.pt"))
        
        # Save vision adapter config
        with open(os.path.join(vision_save_path, "vision_config.json"), "w") as f:
            json.dump(self.vision_config, f, indent=2)
        
        logger.info(f"Vision adapter saved to {vision_save_path}")
        
        # Save tokenizer
        tokenizer_save_path = os.path.join(save_dir, "tokenizer")
        self.tokenizer.save_pretrained(tokenizer_save_path)
        logger.info(f"Tokenizer saved to {tokenizer_save_path}")
        
        return {
            "llama_path": llama_save_path,
            "vision_path": vision_save_path,
            "tokenizer_path": tokenizer_save_path
        }

    @classmethod
    def load_components_separately(cls, save_dir, args=None):
        """Load vision and language components separately"""
        # Load tokenizer
        tokenizer_path = os.path.join(save_dir, "tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Create model instance
        if args is None:
            # Create minimal args for loading
            class MinimalArgs:
                def __init__(self):
                    self.load_in_8bit = False
                    self.use_lora = False
                    self.image_embedding_dim = 384
            args = MinimalArgs()
            args.model_path = os.path.join(save_dir, "llama_model")
        
        model = cls(args, tokenizer=tokenizer)
        
        # Load vision adapter
        vision_path = os.path.join(save_dir, "vision_adapter")
        vision_state_dict = torch.load(os.path.join(vision_path, "vision_adapter.pt"))
        model.vision_adapter.load_state_dict(vision_state_dict)
        
        # Load vision config
        with open(os.path.join(vision_path, "vision_config.json"), "r") as f:
            model.vision_config = json.load(f)
        
        logger.info(f"Model components loaded from {save_dir}")
        return model, tokenizer

    def configure_optimizers(self):
        """Configure optimizers for both vision and language components"""
        no_decay = ["bias", "LayerNorm.weight"]

        # LLaMA model parameters
        llama_params_decay = {
            "params": [p for n, p in self.llama_model.named_parameters() 
                      if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": self.hparams.args.weight_decay,
        }

        llama_params_no_decay = {
            "params": [p for n, p in self.llama_model.named_parameters() 
                      if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        }

        # Vision adapter parameters (higher learning rate)
        vision_params = {
            "params": self.vision_adapter.parameters(),
            "weight_decay": self.hparams.args.weight_decay,
            "lr": self.hparams.args.learning_rate * 10.0  # Higher LR for vision adapter
        }

        all_params = [llama_params_decay, llama_params_no_decay, vision_params]

        optimizer = torch.optim.AdamW(all_params, lr=self.hparams.args.learning_rate)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=self.hparams.args.warmup_steps, 
            num_training_steps=self.hparams.args.max_steps
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

class TrainVLMModel(VisionLanguageModel):
    """Training wrapper for VLM model"""
    
    def _calculate_loss(self, batch, mode="train"):
        """Calculate loss for training/validation"""
        outputs = self.forward(**batch)
        loss = outputs.loss
        self.log(f"{mode}_loss", loss, prog_bar=True)
        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        metrics = self._calculate_loss(batch, mode="train")
        
        # Log learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", current_lr, on_step=True, on_epoch=False, sync_dist=True)
        
        return metrics["loss"]

    def validation_step(self, batch, batch_idx):
        metrics = self._calculate_loss(batch, mode="val")
        return metrics["loss"]

def train_vlm_model(args, **kwargs):
    """Main training function with proper model saving"""
    
    CHECKPOINT_PATH = kwargs.get("CHECKPOINT_PATH", None)
    dataloader_train = kwargs.get("dataloader_train", None)
    dataloader_val = kwargs.get("dataloader_val", None)
    tokenizer = kwargs.get("tokenizer", None)

    # Create root directory
    root_dir = os.path.join(CHECKPOINT_PATH, args.name_test)
    os.makedirs(root_dir, exist_ok=True)

    # Configure callbacks
    checkpoint_callback = ModelCheckpoint(
        filename="vlm-model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=2,
        monitor=f'val_{args.monitor}',
        mode="min",
    )

    early_stopping_callback = EarlyStopping(
        monitor=f'val_{args.monitor}',
        patience=args.patience,
        mode="min",
        verbose=True
    )

    # Configure trainer
    trainer = pl.Trainer(
        deterministic=True,
        default_root_dir=root_dir,
        max_epochs=args.max_epochs,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        precision=args.precision,
        callbacks=[checkpoint_callback, early_stopping_callback],
        log_every_n_steps=10,
        accelerator=args.device,
        devices=args.num_devices,
        strategy=args.strategy,
        profiler="simple",
    )

    # Model paths
    pretrained_filename = os.path.join(
        CHECKPOINT_PATH, args.name_test, "lightning_logs", "version_1", "checkpoints",
        "vlm-model-epoch=05-val_loss=1.16.ckpt"
    )

    # Training/Testing logic
    if args.mode == "train":
        model = TrainVLMModel(args, tokenizer=tokenizer)
        trainer.fit(model=model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)
        
    elif args.mode == "finetune":
        if not os.path.exists(pretrained_filename):
            raise FileNotFoundError(f"Checkpoint not found: {pretrained_filename}")
        model = TrainVLMModel.load_from_checkpoint(pretrained_filename, args=args, tokenizer=tokenizer)
        trainer.fit(model=model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)
        
    elif args.mode == "test":
        if not os.path.exists(pretrained_filename):
            raise FileNotFoundError(f"Checkpoint not found: {pretrained_filename}")
        model = TrainVLMModel.load_from_checkpoint(pretrained_filename, args=args, tokenizer=tokenizer)
        print("Model loaded for testing")

    # Save model components
    model_save_path = os.path.join(CHECKPOINT_PATH, args.name_test, "final_model")
    os.makedirs(model_save_path, exist_ok=True)

    # Save complete Lightning checkpoint
    trainer.save_checkpoint(os.path.join(model_save_path, "complete_model.ckpt"))
    print(f"Complete Lightning model saved to {os.path.join(model_save_path, 'complete_model.ckpt')}")

    # Save components separately
    component_paths = model.save_components_separately(model_save_path)
    print("Model components saved separately:")
    for component, path in component_paths.items():
        print(f"  {component}: {path}")

    return model

def main():
    args = get_args()

    # Set deterministic behavior
    torch.set_float32_matmul_precision('high')
    pl.seed_everything(args.seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Setup paths
    HOME_PATH = os.path.dirname(os.getcwd())
    CHECKPOINT_PATH = os.path.join(HOME_PATH, "saved_models", args.projectName)
    model_path = os.path.join(HOME_PATH, "Models", args.modelName)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    args.model_path = model_path

    # Load and configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add special tokens
    special_tokens_dict = {
        "additional_special_tokens": ["<|user|>", "<|assistant|>", "<|image|>", "<|end_image|>"]
    }
    args.num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    args.tokenizer_size = len(tokenizer)
    print(f"Added {args.num_added_tokens} special tokens.")

    # Create datasets
    train_dataset = ClassDataset(
        path_dataset=args.path_dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
        split='train'
    )
    
    validation_dataset = ClassDataset(
        path_dataset=args.path_dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
        split='validation'
    )

    # Create dataloaders
    dataloader_train = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=partial(cot_collate_fn, tokenizer=tokenizer),
        shuffle=True,
        drop_last=False,
        pin_memory=True
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

    # Calculate training steps
    args.warmup_steps = int(args.warmup_steps * args.max_epochs)
    args.max_steps = len(dataloader_train) * args.max_epochs

    # Training arguments
    training_args = {
        "dataloader_train": dataloader_train,
        "dataloader_val": dataloader_val,
        "tokenizer": tokenizer,
        "CHECKPOINT_PATH": CHECKPOINT_PATH,
    }

    # Train model
    model = train_vlm_model(args=args, **training_args)
    
    print("Training completed successfully!")
    print("Model components are saved separately for easy reuse.")

if __name__ == "__main__":
    main()