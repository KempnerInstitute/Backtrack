import argparse
import os
import random
import json
import torch
import numpy as np 

from accelerate import Accelerator
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset

from tokenization import data_preprocessing, load_tokenizer

import torch.nn as nn
import wandb

# Some global configs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from datasets.utils.logging import disable_progress_bar
# disable_progress_bar()


def main(args):
    # print args
    print(args)

    # read config from a json config file
    with open(args.config, "r") as f:
        config = json.load(f)

    # print config
    print(json.dumps(config, indent=4))

    # set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialize the accelerator
    accelerator = Accelerator()

    # set wandb basics 
    if args.wandb and accelerator.is_main_process:
        wandb_kwargs = { 
                "project": "reason",
                "entity": "harvardml",
            }
        
        os.environ["WANDB_ENTITY"] = wandb_kwargs["entity"]
        os.environ["WANDB_PROJECT"] = wandb_kwargs["project"]

        wandb.init(
            project=wandb_kwargs["project"],
            entity=wandb_kwargs["entity"],
            name=config["name"],
        )
        report = "wandb"
        disable_tqdm = True
    else:
        report = "none"
        disable_tqdm = False

    # load tokenizer
    tokenizer, mapping, reverse_mapping = load_tokenizer()
    # print tokenizer vocab size
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")


    # Define the GPT-2 configuration
    # model_config = GPT2Config(
    #     n_layer=6,          # Set the number of transformer layers to 8
    #     n_head=8,           # Number of attention heads
    #     n_embd=512,         # Hidden size (embedding dimension)
    #     vocab_size=tokenizer.vocab_size,     # Size of the vocabulary
    #     n_positions=4096,   # Maximum sequence length
    #     n_ctx=4096,          # Context window size
    #     use_cache=False,     # Whether to use the cache in the forward pass
    #     bos_token_id=tokenizer.bos_token_id,
    #     eos_token_id=tokenizer.eos_token_id,
    # )

    # # Initialize the GPT-2 model from the configuration
    # model = GPT2LMHeadModel(model_config).to(torch.bfloat16)

    # Define the Qwen2 configuration
    model_config = Qwen2Config(
        hidden_size=config["hidden_size"],         # Reduced hidden size (default Qwen-1.8B has 2048)
        num_hidden_layers=config["num_hidden_layers"],    # Smaller number of layers
        num_attention_heads=config["num_attention_heads"],  # Scaled-down attention heads
        intermediate_size=config["intermediate_size"],  # Feed-forward layer size
        max_position_embeddings=4096,  # RoPE supports longer contexts
        num_key_value_heads=config["num_key_value_heads"],   # Grouped Query Attention (improves efficiency)
        use_sliding_window=False,
        vocab_size=tokenizer.vocab_size,
        use_cache=False,  # Disable caching 
        rope_theta=1000000.0,     # Large theta for longer context handling
    )

    model = Qwen2ForCausalLM(model_config).to(torch.bfloat16)
    print(model)
    
    print(f"Model size: {sum(p.numel() for p in model.parameters()):,}")

    # load dataset
    if config["train_files"] == "all":
        train_files = os.listdir(config["data_dir"])
        train_files = [os.path.join(config["data_dir"], x)
                    for x in train_files if "sample" not in x]
    else:
        raise NotImplementedError
    # train_file = os.path.join(config["data_dir"], config["train_file"])
    val_file = os.path.join(config["data_dir"], config["val_file"])
    
    hf_datasets = load_dataset(
        "json",
        data_files={
            "train": train_files,
            "val": val_file,
        },
        cache_dir="/n/netscratch/dam_lab/Lab/sqin/cache/datasets/",
    )

    def tokenize(element):
        if config["train_type"] == 'sft':
            text = [
                tokenizer.bos_token_w_space 
                + element["board"][e] + '\n'
                + tokenizer.sol_start_token  + '\n'
                + element["full_trace_solution"][e] + '\n' 
                + tokenizer.sol_end_token 
                + tokenizer.eos_token_w_space
                for e in range(len(element["id"]))
            ]
            processed_text = data_preprocessing(text, mapping)
            
        elif config["train_type"] == 'oft':
            text = [
                tokenizer.bos_token_w_space 
                + element["board"][e] + '\n'
                + tokenizer.sol_start_token  + '\n'
                + element["shortcut_solution"][e] + '\n' 
                + tokenizer.sol_end_token 
                + tokenizer.eos_token_w_space
                for e in range(len(element["id"]))
            ]
            processed_text = data_preprocessing(text, mapping)
        
        elif config["train_type"] == 'sft-strat':
            text = [
                tokenizer.bos_token_w_space 
                + element["board"][e] + '\n'
                + tokenizer.sol_start_token  + '\n'
                + element["strategy_full_trace_solution"][e] + '\n' 
                + tokenizer.sol_end_token 
                + tokenizer.eos_token_w_space
                for e in range(len(element["id"]))
            ]
            processed_text = data_preprocessing(text, mapping)

        elif config["train_type"] == 'oft-strat':
            text = [
                tokenizer.bos_token_w_space 
                + element["board"][e] + '\n'
                + tokenizer.sol_start_token  + '\n'
                + element["strategy_shortcut_solution"][e] + '\n' 
                + tokenizer.sol_end_token 
                + tokenizer.eos_token_w_space
                for e in range(len(element["id"]))
            ]
            processed_text = data_preprocessing(text, mapping)


        # deprecated
        # elif config["train_type"] == 'dt': 
        #     text = [
        #         tokenizer.bos_token_w_space 
        #         + element["board"][e] + '\n'
        #         + tokenizer.sol_start_token  + '\n'
        #         + element["strategy_solution"][e] + '\n' 
        #         + tokenizer.sol_end_token 
        #         + tokenizer.eos_token_w_space
        #         for e in range(len(element["id"]))
        #     ]
        #     processed_text = data_preprocessing(text, mapping)

        else:
            raise ValueError(f"Invalid train type: {config['train_type']}")

        # tokenize the processed text
        outputs = tokenizer(processed_text,
                            truncation=True,
                            max_length=config["context_length"],
                            return_overflowing_tokens=True,
                            return_length=True,
                            stride=0,
                            padding="max_length",                        
                            )
        for i in range(len(outputs["input_ids"])):
            if len(outputs["input_ids"][i]) != config["context_length"]:
                print(i)
                print(outputs["input_ids"][i])
        
        return {"input_ids": outputs["input_ids"]}
    
    # take a subset of the val set
    hf_datasets["val"] = hf_datasets["val"].select(range(100))
    # standardize train data size
    hf_datasets["train"] = hf_datasets["train"].select(range(config["num_train"])) # currently fixed to 2.8M samples\

    tokenized_datasets = hf_datasets.map(
        tokenize, batched=True, remove_columns=hf_datasets["train"].column_names,
        cache_file_names={
            "train": os.path.join(config["data_cache_dir"], config["train_cache"]),
            "val":  os.path.join(config["data_cache_dir"], config["val_cache"]),
        },
    )
    # print some tokenized dataset example
    print("Tokenized dataset example")
    print(tokenized_datasets["train"][0])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    print("tokenized dataset", tokenized_datasets)

    training_args = TrainingArguments(
        # Output and logging
        output_dir=config["output_dir"],                # Output directory
        logging_steps=config["log_steps"],              # Log every N steps
        save_total_limit=config["save_total_limit"],    # Maximum number of checkpoints to keep
        save_steps=config["save_steps"],                # Save checkpoint every N steps

        # Training settings
        per_device_train_batch_size=config["per_device_batch_size"],  # Adjusted for GPUs
        gradient_accumulation_steps=config["gradient_accumulation_steps"],  # Accumulate gradients
        num_train_epochs=config["num_train_epochs"],    # Set to None for step-based training
        gradient_checkpointing=False,

        # Optimization
        learning_rate=config["lr"],                     # Learning rate
        weight_decay=config["weight_decay"],            # Weight decay
        warmup_steps=config["warmup_steps"],            # Warmup steps
        lr_scheduler_type=config["lr_scheduler_type"],  # Scheduler type

        # Evaluation
        evaluation_strategy="steps",                    # Evaluate during training
        eval_steps=config["eval_steps"],                # Evaluate every N steps
        per_device_eval_batch_size=config["per_device_batch_size"],  # Adjusted for GPUs
        
        # Wandb
        report_to=report,
        run_name=config["name"],
        
        # Other
        bf16=True,
        seed=config["seed"],                            # Random seed
        push_to_hub=False,
        torch_compile=True,
        metric_for_best_model="valid_loss",
        greater_is_better=False,
        disable_tqdm=disable_tqdm,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset={
            "valid": tokenized_datasets["val"],
        },
    )

    # train
    if args.resume:
        trainer.train(resume_from_checkpoint=args.ckpt)
    else:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sft.conf")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()
    main(args)
