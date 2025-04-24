import argparse
import json
import os
import random

import torch
from accelerate import Accelerator
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer
from transformers import TrainingArguments

import wandb

# global configs
import datasets
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def main(args):
    # read config from a json config file
    with open(args.config, "r") as f:
        config = json.load(f)

    # set seeds
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    # set up accelerator
    accelerator = Accelerator()

    if args.wandb and accelerator.is_main_process:
        wandb_kwargs = config.get(
            "wandb",
            {
                "project": "reason",
                "entity": "harvardml",
            },
        )
        wandb.init(
            project=wandb_kwargs["project"],
            entity=wandb_kwargs["entity"],
            name=config["name"],
            config=config,
        )


    # only GPTNeo model for now
    if config['architecture'] == "GPTNeo":
        from transformers import GPTNeoConfig, GPTNeoForCausalLM
        model_config = "../configs/gpt-neo-s.json"
        with open(model_config, "r") as f:
            model_config = json.load(f)
        if not args.reset:
            model_config = GPTNeoConfig(**model_config)
            model = GPTNeoForCausalLM(model_config).to(torch.bfloat16)
            # model = model.half()
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B",
                                                    device_map="auto",)
        else:
            model = GPTNeoForCausalLM.from_pretrained(
                args.ckpt,
                torch_dtype=torch.bfloat16,
                # attn_implementation="flash_attention_2",
            )

            tokenizer = AutoTokenizer.from_pretrained(args.ckpt)
        tokenizer.pad_token = tokenizer.eos_token
    
    elif config['architecture'] == "Qwen":
        from transformers import Qwen2Config, Qwen2ForCausalLM

        tokenizer = PreTrainedTokenizerFast(tokenizer_file="countdown_tokenizer.json")
        tokenizer.pad_token = "PAD"
        tokenizer.bos_token = " START "
        tokenizer.eos_token = " END "

        model_config = Qwen2Config(
            hidden_size=config["hidden_size"],         # Embedding dimension
            num_hidden_layers=config["num_hidden_layers"],     # Number of Transformer layers
            num_attention_heads=config["num_attention_heads"],   # Multi-head self-attention heads
            intermediate_size=config["intermediate_size"],   # Feedforward layer width
            max_position_embeddings=8192,  # RoPE: Long context handling
            num_key_value_heads=config["num_key_value_heads"],    # Grouped Query Attention for memory efficiency
            vocab_size=tokenizer.vocab_size,         # Adjust as per tokenizer vocabulary
            use_cache=False,  # Disable caching 
            rope_theta=1000000.0,     # Scaling factor for RoPE embeddings
        )
        model = Qwen2ForCausalLM(model_config)
        model.to("cuda" if torch.cuda.is_available() else "cpu")

    else:
        raise ValueError(f"Invalid architecture: {config['architecture']}")

    print(f"Number of parameters: {model.num_parameters():,}")
    print(model)

    # load dataset
    if "n100000" in config["train_file"]:
        assert config["train_file"] == "train0_b8_t100_n100000_dfs.json"
        assert "b4_3_dfs_deep" in config["data_dir"]
        train_file = [
            os.path.join(config["data_dir"], f"train{i}_b8_t100_n100000_dfs.json")
            for i in range(5)
        ]
    else:
        train_file = os.path.join(config["data_dir"], config["train_file"])
    
    val_file = os.path.join(config["data_dir"], config["val_file"])
    val_target_file = os.path.join(config["data_dir"], config["val_target_file"])
    hf_datasets = load_dataset(
        "json",
        data_files={
            "train": train_file,
            "val": val_file,
            "val_target": val_target_file,
        },
    )
    hf_datasets["train"] = hf_datasets["train"].select(range(int(config["num_train"])))

    context_length = config["context_length"]
    tokenizer.model_max_length = context_length

    def tokenize(element):
        if config["train_type"] == "dt":
            text = [
                tokenizer.bos_token
                + f"{element['rating'][e]:0.2f}->"
                + element["search_path"][e].strip()
                + tokenizer.eos_token
                for e in range(len(element["search_path"]))
            ]
        elif config["train_type"] == "sft":
            text = [
                tokenizer.bos_token
                + element["search_path"][e].strip()
                + tokenizer.eos_token
                for e in range(len(element["search_path"]))
            ]
        elif config["train_type"] == "oft":
            text = [
                tokenizer.bos_token
                + element["optimal_path"][e].strip()
                + tokenizer.eos_token
                for e in range(len(element["optimal_path"]))
            ]
        else:
            raise ValueError(f"Invalid train type: {config['train_type']}")
        outputs = tokenizer(
            text,
            truncation=True,
            max_length=context_length,
            return_overflowing_tokens=True,
            return_length=True,
            stride=0,
            padding="max_length",
        )
        return {"input_ids": outputs["input_ids"]}

    # tokenize dataset for causal LM
    if config['architecture'] == "GPTNeo":
        cache_suffix = "_gptneo"
    elif config['architecture'] == "Qwen":
        cache_suffix = "_qwen"
    else:
        raise ValueError(f"Invalid architecture: {config['architecture']}")
    
    tokenize_data = tokenize(hf_datasets["train"][0:10])
    print(tokenize_data)
    quit()
    tokenized_datasets = hf_datasets.map(
        tokenize, batched=True, remove_columns=hf_datasets["train"].column_names,
        cache_file_names={
            "train": f"/n/netscratch/dam_lab/Lab/sqin/cache/datasets/{config['train_file']}_{cache_suffix}_{config['train_type']}.cache",
            "val": f"/n/netscratch/dam_lab/Lab/sqin/cache/datasets/{config['val_file']}_{cache_suffix}_{config['train_type']}.cache",
            "val_target": f"/n/netscratch/dam_lab/Lab/sqin/cache/datasets/{config['val_target_file']}_{cache_suffix}_{config['train_type']}.cache",
        },
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    print("tokenized dataset", tokenized_datasets)

    # prepare training
    training_args = TrainingArguments(
        output_dir=os.path.join(config["output_dir"], config["name"]),
        per_device_train_batch_size=config["batch_size"],
        evaluation_strategy="steps",
        eval_steps=config["eval_steps"],
        logging_steps=config["log_steps"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        gradient_checkpointing=True,
        num_train_epochs=config["num_train_epochs"],
        weight_decay=config["weight_decay"],
        warmup_steps=config["warmup_steps"],
        lr_scheduler_type=config["lr_scheduler_type"],
        learning_rate=config["lr"],
        save_strategy="steps",
        save_total_limit=config["save_total_limit"],
        save_steps=config["save_steps"],
        seed=config["seed"],
        bf16=True,
        push_to_hub=False,
        report_to="wandb",
        run_name=config["name"],
        ddp_find_unused_parameters=False,
        load_best_model_at_end=True,
        torch_compile=True,
        metric_for_best_model="valid_loss",
        greater_is_better=False,
        disable_tqdm=True,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset={
            "valid": tokenized_datasets["val"],
            "valid_target": tokenized_datasets["val_target"],
        },
    )

    # train
    if args.resume:
        trainer.train(resume_from_checkpoint=args.ckpt)
    else:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../../configs/conf.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()
    main(args)
