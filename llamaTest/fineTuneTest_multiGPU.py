import os


# Environment variable to help with memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

from pytorch_lightning.callbacks import ModelCheckpoint
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LlamaForCausalLM, get_linear_schedule_with_warmup

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from datasets import load_dataset
from dataclasses import dataclass, field

from typing import List, Optional, Tuple
from accelerate import Accelerator
from accelerate.utils import set_seed


# Make both GPUs visible
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Assuming 0 is 16GB and 1 is 8GB



@dataclass
class Args:
    BATCH_SIZE: int = 4  # Base batch size that will be adjusted per-GPU
    NUM_WORKERS: int = 4  # Reduced worker count for stability
    RANDOM_SEED: int = 42
    projectName: str = "LlamaFinetune"
    datasetName: str = "FreedomIntelligence/medical-o1-reasoning-SFT"
    modelName: str = 'Llama3.2-1B-Instruct-hf'

    # Model Parameters - optimized for memory efficiency
    lora_r: int = 8  # LoRA attention dimension
    lora_alpha: int = 16  # LoRA alpha parameter
    lora_dropout: float = 0.05
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 1000
    gradient_accumulation_steps: int = 16  # Accumulate gradients to compensate for smaller batch sizes
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"
    ])


class ClassDataset(Dataset):
    def __init__(self, datasetName:str, lang:str = 'en', tokenizer = None, max_length: int = 512, split: str = 'train'):
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

        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.create_tokens_text(self.data[index]) 
    
    def create_tokens_text(self, data_text):
        conversation = (
            f"<|user|>\n{data_text['Question']}\n"
            f"<|assistant|>\n{data_text['Complex_CoT']}\n\n"
            f"My response is:\n{data_text['Response']}"
        )

        # Tokenize
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

        # Mask labels for user prompts (optional)
        # This means we only calculate loss on assistant responses
        # Find positions of <|assistant|> tokens
        assistant_positions = []
        assistant_token_id = self.tokenizer.convert_tokens_to_ids("<|assistant|>")
        for i, token_id in enumerate(input_ids):
            if token_id == assistant_token_id:
                assistant_positions.append(i)

        # Set labels for non-assistant text to -100 (ignored in loss calculation)
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
            "labels": labels
            }


def train_with_accelerate(args):
    """
    Train the model using Hugging Face Accelerate for asymmetric multi-GPU training
    """

    

    # Set seed for reproducibility
    set_seed(args.RANDOM_SEED)
    
    # Path setup
    HOME_PATH = os.path.dirname(os.getcwd())
    CHECKPOINT_PATH = os.path.join(HOME_PATH, "saved_models", args.projectName)
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    
    model_path = os.path.join(HOME_PATH, "Models", args.modelName)
    # Verify the path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The directory {model_path} does not exist. Please check the path.")
    
    # Initialize accelerator
    # This will automatically handle asymmetric GPUs
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16",  # Use mixed precision for efficiency
        # even_batches=True, 
        log_with="tensorboard",
        project_dir=os.path.join(CHECKPOINT_PATH, "logs")
    )
    
    # Log GPU information
    accelerator.print(f"Using {accelerator.num_processes} GPU(s)")
    for i in range(accelerator.num_processes):
        device = torch.cuda.get_device_properties(i)
        accelerator.print(f"GPU {i}: {device.name} with {device.total_memory / 1e9:.2f} GB VRAM")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    
    # Set pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Add special tokens if not already added
    special_tokens_dict = {"additional_special_tokens": ["<|user|>", "<|assistant|>"]}
    num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    
    accelerator.print(f"Added {num_added_tokens} special tokens.")
    
    tokenizer_size = len(tokenizer)
    
    # Load datasets
    datasetTrain = ClassDataset(datasetName=args.datasetName, tokenizer=tokenizer, split="train")
    datasetVal = ClassDataset(datasetName=args.datasetName, tokenizer=tokenizer, split="validation")
    
    # Create dataloaders
    # Adjust per-device batch size based on available GPUs
    per_device_batch_size = max(1, args.BATCH_SIZE // accelerator.num_processes)
    
    train_dataloader = DataLoader(
        dataset=datasetTrain,
        batch_size=per_device_batch_size,
        num_workers=args.NUM_WORKERS,
        pin_memory=True,
        shuffle=True
    )
    
    eval_dataloader = DataLoader(
        dataset=datasetVal,
        batch_size=per_device_batch_size,
        num_workers=args.NUM_WORKERS,
        shuffle=False,
        pin_memory=True
    )
    
    # Load model 
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    )
    
    # Resize embeddings for new tokens
    if num_added_tokens > 0:
        model.resize_token_embeddings(tokenizer_size)
    
    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
    )
    
    # Get PEFT model
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Set up optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    
    # Set up learning rate scheduler
    # Calculate the total number of training steps
    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    max_train_steps = args.max_steps if args.max_steps > 0 else num_update_steps_per_epoch * args.max_epochs
    
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=max_train_steps,
    )
    
    # Prepare everything with accelerator
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    
    # Initialize trackers for reporting
    accelerator.init_trackers(args.projectName)
    
    # Training loop
    completed_steps = 0
    best_val_loss = float('inf')
    
    # Train the model
    for epoch in range(args.max_epochs):
        model.train()
        total_loss = 0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                batch = {k: v.to(accelerator.device) for k, v in batch.items()}

                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                
                # Update weights
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                total_loss += loss.detach().float()
            
            # Log training metrics
            if step % 10 == 0:
                accelerator.log({
                    "train_loss": loss.detach().float(),
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                }, step=completed_steps)
                
                accelerator.print(f"Epoch {epoch+1} | Step {step} | Loss: {loss.detach().float():.4f}")
            
            completed_steps += 1
            
            if completed_steps >= max_train_steps:
                break
        
        # Evaluation
        model.eval()

        with torch.no_grad():

            eval_loss = 0
            eval_steps = 0
        
            for batch in eval_dataloader:
                with torch.no_grad():
                    batch = {k: v.to(accelerator.device) for k, v in batch.items()}

                    outputs = model(**batch)
            
                eval_loss += outputs.loss.detach().float()
                eval_steps += 1
        
            eval_loss = eval_loss / eval_steps
        
            # Log validation metrics
            accelerator.log({
                "val_loss": eval_loss,
            }, step=completed_steps)
        
            accelerator.print(f"Epoch {epoch+1} | Validation Loss: {eval_loss:.4f}")
        
            # Save checkpoint if validation loss improved
            if eval_loss < best_val_loss:
                best_val_loss = eval_loss
            
                # Wait for all processes to reach this point
                accelerator.wait_for_everyone()
                set_seed(args.RANDOM_SEED)
            
                # Only save from the main process
                if accelerator.is_main_process:
                    checkpoint_dir = os.path.join(CHECKPOINT_PATH, f"checkpoint-epoch-{epoch+1}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                
                    # Get unwrapped model
                    unwrapped_model = accelerator.unwrap_model(model)

                    # Save model
                    unwrapped_model.save_pretrained(
                        checkpoint_dir,
                        # is_main_process=accelerator.is_main_process,
                        # save_function=accelerator.save,
                    )
                
                    # Save tokenizer
                    tokenizer.save_pretrained(checkpoint_dir)
                
                    accelerator.print(f"Saved checkpoint to {checkpoint_dir}")
    
    # Save final model
    accelerator.wait_for_everyone()
    set_seed(args.RANDOM_SEED)

    if accelerator.is_main_process:
        final_model_path = os.path.join(CHECKPOINT_PATH, "final_model")
        os.makedirs(final_model_path, exist_ok=True)

        # Get unwrapped model
        unwrapped_model = accelerator.unwrap_model(model)

        # Save model (ONLY in main process)
        unwrapped_model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)

        accelerator.print(f"Saved final model to {final_model_path}")
    
    # End training
    accelerator.end_training()


def main():
    args = Args()
    
    # Override max_epochs since we're using max_steps
    args.max_epochs = 10
    
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Train model with accelerate
    train_with_accelerate(args)


if __name__ == "__main__":
    main()