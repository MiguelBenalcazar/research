"""
https://huggingface.co/docs/transformers/en/main_classes/text_generation
https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT/viewer/en/train?views%5B%5D=en&row=4
https://www.latent.space/p/2025-papers
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from transformers import AutoTokenizer, LlamaForCausalLM
import torch
import argparse



def get_args():
    parser = argparse.ArgumentParser(description="Finetune LLaMA")

    parser.add_argument("--path", type=str, default="/home/home/Desktop/research/saved_models/LlamaFinetuneTestMalky/LlamaFinetune/final_model/", help="Path where model was saved.")
    parser.add_argument("--quantization", type=bool, default=True, help="Enabling quatization 8 bit")
    parser.add_argument("--device", type=int, default=1, help="Choose a device to run.")

    parser.add_argument("--max_new_tokens", type=int, default=512, help=" The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.")
    parser.add_argument("--temperature", type=float, default=0.1, help=" The value used to module the next token probabilities.")
    parser.add_argument("--top_p", type=float, default=0.9, help="If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.")
    
    parser.add_argument("--do_sample", type=bool, default=True, help="Whether or not to use sampling ; use greedy decoding otherwise.")

    return parser.parse_args()


def main():
    args = get_args()
    
    model_path = args.path

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


    user_prompt = (
        "A 45-year-old man presents with symptoms including a wide-based gait, "
        "a blank facial expression, hallucinations, memory issues, a resting "
        "tremor that resolves with movement, and bradykinesia. Based on these clinical findings, what is most likely to be observed in the histological specimen of his brain?"
    )
    

    full_prompt = f"<|user|>{user_prompt}<|assistant|>"

    # Tokenize
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    print(inputs)

    # Generate
    output = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        pad_token_id=tokenizer.eos_token_id,
    )
    # Decode
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    print(generated_text)




if __name__ == "__main__":
    main()