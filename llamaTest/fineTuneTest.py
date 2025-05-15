import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Specify the GPU you want to use
# os.environ['RANK'] = '0'  # Set the rank manually, useful in multi-GPU setups
# os.environ['LOCAL_RANK'] = '0'  # Set the local rank manually
# os.environ['WORLD_SIZE'] = '1'  # Set world size (for single-node multi-GPU setup)

import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, default_data_collator, get_linear_schedule_with_warmup
# from bitsandbytes import BitsAndBytesConfig

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PeftModel

from datasets import load_dataset
from dataclasses import dataclass, field

from typing import List, Optional, Tuple
from functools import partial
from torch.nn.utils.rnn import pad_sequence
import re
# from deepspeed.ops.adam import DeepSpeedCPUAdam

from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer



DEEPSPEED_TYPES = [
    "deepspeed_stage_1", 
    "deepspeed_stage_2", 
    "deepspeed_stage_2_offload",
    "deepspeed_stage_3",
    "deepspeed_stage_3_offload"]


@dataclass
class Args:
    BATCH_SIZE: int = 4
    NUM_WORKERS: int = 5
    RANDOM_SEED: int = 42
    projectName: str = "LlamaFinetuneTestMalky"
    datasetName: str = "FreedomIntelligence/medical-o1-reasoning-SFT"
    modelName: str = 'Llama3.2-1B-Instruct-hf'

    max_length:int=512

    # Model Parameters
    lora_r: int = 4
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 1000
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"
    ])

    device:str="gpu" if torch.cuda.is_available() else "cpu"
    num_devices:int = 1#torch.cuda.device_count()  if torch.cuda.is_available() else 0
    strategy:str= 'auto' # DEEPSPEED_TYPES[4] if torch.cuda.is_available() else "auto"


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

    
    # def create_tokens_text(self, data_text):
   
    #     conversation = (
    #         f"<|user|>\n{data_text['Question']}\n"
    #         f"<|assistant|>\nLet's think step by step.\n{data_text['Complex_CoT']}\n"
    #         f"The final answer is: {data_text['Response']}"
    #     )

    #     # Append EOS token to the conversation
    #     conversation += self.tokenizer.eos_token



    #     # Tokenize
    #     encodings = self.tokenizer(conversation, 
    #                     truncation=True,
    #                     max_length=self.max_length,
    #                     padding="max_length",
    #                     return_tensors="pt"
    #                 )
    
    #     # Create labels (for causal language modeling)
    #     input_ids = encodings["input_ids"][0]
    #     attention_mask = encodings["attention_mask"][0]
    #     labels = input_ids.clone()



    #     # Find positions of <|assistant|> and "The final answer is:"
    #     assistant_token_id = self.tokenizer.convert_tokens_to_ids("<|assistant|>")
    #     final_answer_text = "The final answer is:"

    #     # Tokenize the final answer text to find its token IDs
    #     final_answer_tokens = self.tokenizer(final_answer_text, add_special_tokens=False)["input_ids"]
    
    #     # Initialize variables
    #     assistant_start = None
    #     final_answer_start = None


    #     # Find assistant start
    #     for i, token_id in enumerate(input_ids):
    #         if token_id == assistant_token_id:
    #             assistant_start = i
    #             break

    #     # Find final answer start by looking for the sequence of tokens
    #     for i in range(len(input_ids) - len(final_answer_tokens) + 1):
    #         if input_ids[i:i + len(final_answer_tokens)].tolist() == final_answer_tokens:
    #             final_answer_start = i
    #             break


    #     # Selective loss masking for reasoning-only learning
    #     if assistant_start is not None and final_answer_start is not None:
    #         for i in range(len(labels)):
    #             # Keep labels only for the reasoning part (from <|assistant|> to before "The final answer is:")
    #             if i < assistant_start or i >= final_answer_start:
    #                 labels[i] = -100


    #     return {
    #         # "text":conversation,
    #         "input_ids": input_ids,
    #         "attention_mask": attention_mask,
    #         "labels": labels
    #         }
    
    def create_tokens_text(self, data_text):
        conversation = (
            f"<|user|>\n{data_text['Question']}\n"
            f"<|assistant|>\nLet's think step by step.\n{data_text['Complex_CoT']}\n"
            f"The final answer is: {data_text['Response']}"
        )

        # Append EOS token to the conversation
        conversation += self.tokenizer.eos_token

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
        assistant_token_id =self.tokenizer.convert_tokens_to_ids("<|assistant|>")
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
            # "text":conversation,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
            }




def cot_collate_fn(batch, tokenizer):
    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]

    # Pad all to the longest sequence in the batch
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels,
    }


class LoRAFineTuner(pl.LightningModule):
    def __init__(self, args,**kwargs):
        super().__init__()     
    
        self.save_hyperparameters(ignore=['dataloader_train_', 'dataloader_val_', "tokenizer"])

        learning_rate = kwargs.get("learning_rate", 5e-5)
        weight_decay = kwargs.get("weight_decay", 0.01)
        warmup_steps = kwargs.get("warmup_steps", 100)
        max_steps = kwargs.get("max_steps", 100)
        num_added_tokens = kwargs.get("num_added_tokens", None)
        tokenizer_size = kwargs.get("num_tokenizer_sizeadded_tokens", None)

        self.tokenizer = kwargs.get("tokenizer", None)
      

        # Load the model with the new quantization configuration
        self.model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args.model_path,
            quantization_config={"load_in_8bit": True},  # Pass the quantization config here
            torch_dtype=torch.float16,
            # device_map="auto"  # You can uncomment this if you want automatic device allocation
        )    

        if self.hparams.num_added_tokens and self.hparams.num_added_tokens > 0:
            new_vocab_size = len(self.tokenizer)
            self.model.resize_token_embeddings(new_vocab_size)


        # if self.hparams.num_added_tokens is not None:
        # # # Resize model embeddings!
        #     if self.hparams.num_added_tokens > 0:
        #         self.model.resize_token_embeddings(self.hparams.tokenizer_size)   

      
        # Prepare model for training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        peft_config = LoraConfig(
            task_type = TaskType.CAUSAL_LM,
            inference_mode = False,
            r = args.lora_r,
            lora_alpha = args.lora_alpha,
            lora_dropout = args.lora_dropout,
            target_modules = args.lora_target_modules,
        )
        
        # Get PEFT model
        self.model = get_peft_model(self.model, peft_config)

        
        
        self.model.print_trainable_parameters()
    
    

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate)
        
        # optimizer = DeepSpeedCPUAdam(optimizer_grouped_parameters, lr=self.hparams.learning_rate)


        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=self.hparams.warmup_steps, 
            num_training_steps=self.hparams.max_steps
        )
        
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
    
 
class fineTuneModel(LoRAFineTuner):

    # def extract_cot_and_answer(self, text: str):
    #     """
    #     Extracts Chain-of-Thought reasoning and final answer from generated or reference text.

    #     Args:
    #         text (str): Full model output including CoT reasoning and answer.

    #     Returns:
    #         tuple: (cot_reasoning: str, final_answer: str)
    #     """
    #     # Define common final answer patterns
    #     answer_patterns = [
    #         r"(?:Answer\s*[:\-]?\s*)(.*)",
    #         r"(?:So the answer is\s*[:\-]?\s*)(.*)",
    #         r"(?:The correct answer is\s*[:\-]?\s*)(.*)",
    #         r"(?:My answer is\s*[:\-]?\s*)(.*)",
    #         r"(?:Final answer\s*[:\-]?\s*)(.*)",
    #         r"(?:The final answer is\s*[:\-]?\s*)(.*)",
    #     ]

    #     for pattern in answer_patterns:
    #         match = re.search(pattern, text, flags=re.IGNORECASE)
    #         if match:
    #             answer = match.group(1).strip()
    #             cot = text[:match.start()].strip()
    #             return cot, answer

    #     # If no answer pattern matched, treat the whole text as CoT and answer as unknown
    #     return text.strip(), ""

    # def compute_bleu_or_rouge(self, gen: str, ref: str) -> float:
    #     """
    #     Compute BLEU or ROUGE score between generated and reference text.
    #     Returns ROUGE-L score by default (can switch to BLEU).
    #     """
    #     scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    #     scores = scorer.score(ref, gen)
    #     return scores['rougeL'].fmeasure  # Return ROUGE-L F1 score

    # # Alternative: BLEU score
    # # ref_tokens = [ref.split()]  # Reference as list of tokens
    # # gen_tokens = gen.split()    # Generated as tokens
    # # return sentence_bleu(ref_tokens, gen_tokens)

    # def exact_match(self, gen: str, ref: str) -> float:
    #     """
    #     Return 1.0 if generated and reference texts match exactly, else 0.0.
    #     """
    #     return 1.0 if gen.strip() == ref.strip() else 0.0


    
    # def _calculate_loss_(self, batch, mode="train") -> torch.Tensor:
    #     outputs = self.model(**batch)
    #     loss = outputs.loss

    #     # Generate model output
    #     with torch.no_grad():  # Don't accumulate gradients when generating
    #         generated_outputs = self.model.generate(
    #             input_ids=batch["input_ids"], 
    #             attention_mask=batch["attention_mask"],
    #             max_new_tokens=self.hparams.args.max_length,
    #             do_sample=True,
    #             pad_token_id=self.tokenizer.pad_token_id,
    #             eos_token_id=self.tokenizer.eos_token_id
    #         )
        
    #     # Decode predictions and references
    #     gen_texts = self.tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)
    #     # Preprocess labels: Replace invalid token IDs with pad_token_id
    #     labels = batch["labels"].clone()  # Avoid modifying the original tensor
    #     labels[labels == -100] = self.tokenizer.pad_token_id  # Replace invalid IDs
    #     ref_texts = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

    #     # Evaluate CoT reasoning and answer accuracy
    #     total_cot_score = 0
    #     total_ans_acc = 0
    #     for gen, ref in zip(gen_texts, ref_texts):
    #         cot_gen, ans_gen = self.extract_cot_and_answer(gen)
    #         cot_ref, ans_ref = self.extract_cot_and_answer(ref)

    #         cot_score = self.compute_bleu_or_rouge(cot_gen, cot_ref)
    #         acc = self.exact_match(ans_gen, ans_ref)

    #         total_cot_score += cot_score
    #         total_ans_acc += acc

    #     avg_cot_score = total_cot_score / len(gen_texts)
    #     avg_ans_acc = total_ans_acc / len(gen_texts)

    #     self.log(f"{mode}_loss", loss, prog_bar=True)
    #     self.log(f"{mode}_cot_score", avg_cot_score, prog_bar=True)
    #     self.log(f"{mode}_answer_acc", avg_ans_acc, prog_bar=True)

    #     return loss


    # def _calculate_loss_(self, batch,  mode="train")-> Tuple[torch.Tensor, float]:
    #     outputs = self.model(**batch)
    #     loss = outputs.loss


    #     # Generate model output
    #     generated_outputs = self.model.generate(
    #         input_ids=batch["input_ids"], 
    #         attention_mask=batch["attention_mask"],
    #         max_new_tokens=self.hparams.args.max_length,
    #         do_sample=True
    #     )
        
    #     # Decode predictions and references
    #     gen_texts = self.tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)
    #     # Preprocess labels: Replace invalid token IDs (e.g., -100) with pad_token_id or filter
    #     labels = batch["labels"].clone()  # Avoid modifying the original tensor
    #     labels[labels == -100] = self.tokenizer.pad_token_id  # Replace invalid IDs
    #     ref_texts = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

    #     # Evaluate CoT reasoning and answer accuracy
    #     total_cot_score = 0
    #     total_ans_acc = 0
    #     for gen, ref in zip(gen_texts, ref_texts):
    #         cot_gen, ans_gen = self.extract_cot_and_answer(gen)
    #         cot_ref, ans_ref = self.extract_cot_and_answer(ref)


    #         cot_score = self.compute_bleu_or_rouge(cot_gen, cot_ref)
    #         acc = self.exact_match(ans_gen, ans_ref)

    #         total_cot_score += cot_score
    #         total_ans_acc += acc

    #     avg_cot_score = total_cot_score / len(gen_texts)
    #     avg_ans_acc = total_ans_acc / len(gen_texts)


    #     self.log(f"{mode}_loss", loss, prog_bar=True)
    #     self.log(f"{mode}_cot_score", avg_cot_score, prog_bar=True)
    #     self.log(f"{mode}_answer_acc", avg_ans_acc, prog_bar=True)

    #     return loss



    def _calculate_loss_(self, batch,  mode="train")-> Tuple[torch.Tensor, float]:
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log(f"{mode}_loss", loss, prog_bar=True)
        return loss
    
    def forward(self, **inputs):
        return self.model(**inputs)
    
    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss_(batch=batch, mode="train")

        # Log learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", current_lr, on_step=True, on_epoch=False, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._calculate_loss_(batch=batch, mode="val")
        return loss
      

def train_finetuneModel(args, **kwargs):


    name_test = kwargs.get("name_test", "nameTestProject")
    monitor = kwargs.get("monitor", "val_loss")
    max_epochs = kwargs.get("max_epochs", 10)
    CHECKPOINT_PATH = kwargs.get("CHECKPOINT_PATH", 10)
    dataloader_train_ = kwargs.get("dataloader_train_", None)
    dataloader_val_ = kwargs.get("dataloader_val_", None)
    tokenizer = kwargs.get("tokenizer", None)
    mode = kwargs.get("mode", "train")
    patience = kwargs.get("patience", 2)

    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = os.path.join(CHECKPOINT_PATH, name_test)
    os.makedirs(root_dir, exist_ok=True)


    # Configure callbacks
    checkpoint_callback = ModelCheckpoint(
        # dirpath="checkpoints",
        filename="llama-3.2-1b-lora-{epoch:02d}-{val_loss:.4f}",
        # filename="llama-3.2-1b-lora-{epoch:02d}-{val_loss:.2f}",
        save_top_k=2,
        monitor= f'val_{monitor}',
        mode="min",
    )

     # Configure logger
    # logger = TensorBoardLogger("logs", name="llama-3.2-1b-lora")

    early_stopping_callback = EarlyStopping(
        monitor=f'val_{monitor}',  # Use val_f1 or val_pr_auc depending on your goal
        patience=patience,  # Stop after 20 epochs without improvement
        mode="min",  # Maximize the F1 score
        verbose=True  # Optional: Print early stopping information
    )

    trainer = pl.Trainer(
        deterministic=True,
        default_root_dir=root_dir,
        max_epochs=max_epochs,
        gradient_clip_val=5.0,
        accumulate_grad_batches=16, 
        precision="16-mixed", 
        # logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        log_every_n_steps=10,

        accelerator=args.device,
        devices=args.num_devices,            # Automatically detect how many GPUs
        strategy=args.strategy,            # Let Lightning choose DDP or DataParallel
        # profiler="simpler", #advanced , simpler
        # num_nodes = 1,

        
        # fast_dev_run=True
    )


    pretrained_filename = os.path.join(
            CHECKPOINT_PATH,
            name_test,
            "lightning_logs",
            "version_0",
            "checkpoints",
            "llama-3.2-1b-lora-epoch=09-val_loss=1.8723.ckpt"
        )

    if mode == "train":
    # Train
        model = fineTuneModel(args, **kwargs)
        trainer.fit(model = model, train_dataloaders=dataloader_train_, val_dataloaders=dataloader_val_ )
    
    elif mode =="finetune":
        
        if not os.path.exists(pretrained_filename):
            raise FileNotFoundError(f"The directory {pretrained_filename} does not exist. Please check the path.")
        
        model = fineTuneModel.load_from_checkpoint(pretrained_filename, args=args, **kwargs)
        trainer.fit(model= model, train_dataloaders=dataloader_train_, val_dataloaders=dataloader_val_)

    elif mode=="test":

        if not os.path.exists(pretrained_filename):
            raise FileNotFoundError(f"The directory {pretrained_filename} does not exist. Please check the path.")
        
        model = fineTuneModel.load_from_checkpoint(pretrained_filename, args=args, **kwargs)
        print("Model Load")
        

    
    # Save the final model
    if isinstance(model.model, PeftModel):
        print("Merging LoRA weights with base model...")
        model.model = model.model.merge_and_unload()
        print("Model Merged")

    # # FUSE LoRA weights into base model
    # # if isinstance(model.model, PeftModel):
    # model.model = model.model.merge_and_unload()
    # print("Model Fused")

    # Save model and tokenizer after training
    model_save_path = os.path.join(CHECKPOINT_PATH, name_test, "final_model")
    os.makedirs(model_save_path, exist_ok=True)
    
    # Save model
    model.model.save_pretrained(model_save_path)
    
    
    # Save tokenizer
    model.model.config.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
  
    print(f"Fused model and tokenizer saved to: {model_save_path}")


def main():
    


    args = Args()
    


    torch.set_float32_matmul_precision('high')
    # Setting the seed
    pl.seed_everything(args.RANDOM_SEED,  workers=True)
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # print("Device:", device)

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

    # Set padding to left for decoder-only models (important for generation)
    tokenizer.padding_side = "left"


    # Set pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add special tokens if not already added
    special_tokens_dict = {"additional_special_tokens": ["<|user|>", "<|assistant|>"]}
    num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)

    print(f"Added {num_added_tokens} special tokens.")  

    tokenizer_size = len(tokenizer)

    datasetTrain = ClassDataset(
        datasetName = args.datasetName, 
        tokenizer=tokenizer, 
        split="train", 
        max_length=args.max_length
        )
    datasetVal = ClassDataset(
        datasetName = args.datasetName, 
        tokenizer=tokenizer, 
        split="validation", 
        max_length=args.max_length
        )

    dataloader_train_ = DataLoader(
            dataset=datasetTrain,
            batch_size= args.BATCH_SIZE,
            num_workers= args.NUM_WORKERS,
            collate_fn=partial(cot_collate_fn, tokenizer=tokenizer),
            shuffle = True,
            pin_memory= True
        )
    
    dataloader_val_ = DataLoader(
            dataset=datasetVal,
            batch_size=args.BATCH_SIZE,
            num_workers=args.NUM_WORKERS,
            collate_fn=partial(cot_collate_fn, tokenizer=tokenizer),
            shuffle=False,
            pin_memory=True
        )
    
    max_epochs = 20
    max_steps = len(dataloader_train_) * max_epochs
    warmup_steps = 0.25 * max_steps

    training_params = {
        "name_test":"LlamaFinetune",
        "monitor":"loss",
        "max_epochs": 10,

        "learning_rate" :1.6e-4,
        "weight_decay": 0.01,
        "warmup_steps": warmup_steps,
        "max_steps": max_steps,

        "num_added_tokens": num_added_tokens,
        "tokenizer_size": tokenizer_size,
        "CHECKPOINT_PATH":CHECKPOINT_PATH,
        "dataloader_train_":dataloader_train_,
        "dataloader_val_":dataloader_val_,
        "tokenizer":tokenizer,
        "patience": 2,
        "mode":'finetune'
    }

    train_finetuneModel(args=args, **training_params)
    


if __name__ == "__main__":
    main()


# BLEU, METEOR< ROUGE< BERTSCORE