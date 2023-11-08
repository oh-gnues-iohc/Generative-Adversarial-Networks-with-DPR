from typing import List, Optional, Literal
from transformers import AutoImageProcessor, ResNetForImageClassification, AutoTokenizer
import torch
from models.generator import Generator, GeneratorConfig
from models.dpr import ImageTextRetrieval, ImageTextRetrievalConfig
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from transformers import TrainingArguments, Trainer, HfArgumentParser
import logging
import os
from dataclasses import dataclass, field
import transformers
from utils.draw import draw_examples

@dataclass
class ModelArguments:
    pretrained_model_name_or_path: str = field(
        default="ohgnues/GAN"
    )
    use_auth_token: str = field(
        default=None, metadata={"help": "Authentication token required for private model usage"}
    )
    
@dataclass
class DPRArguments:
    dpr_model_name_or_path: str = field(
        default="ohgnues/ImageTextRetrieval"
    )
    is_freeze: bool = field(
        default=True, metadata={"help": "Specify whether to freeze the DPR model and not train the discriminator"}
    )

@dataclass
class DataArguments:
    path: str = field(
        default="poloclub/diffusiondb", metadata={"help": "Path or name of the dataset"}
    )
    name: str = field(
        default=None, metadata={"help": "Subset name"}
    )
    cache_dir: str = field(
        default=None, metadata={"help": "Location to store cache files"}
    )
    train_split: str = field(
        default="train", metadata={"help": "Name of the training data"}
    )
    eval_split: str = field(
        default=None, metadata={"help": "Name of the evaluation data"}
    )
    shuffle: bool = field(
        default=True, metadata={"help": "Whether to shuffle the data"}
    )
    text_column_name: str = field(
        default="prompt", metadata={"help": "Column name for text data"}
    )
    image_column_name: str = field(
        default="image", metadata={"help": "Column name for image data"}
    )
    max_length: int = field(
        default=100, metadata={"help": "Maximum token length"}
    )
    
@dataclass
class TrainArguments:
    output_dir: str = field(
        default="runs/", metadata={"help": "Output directory for training results"}
    )
    do_eval: bool = field(
        default=False, metadata={"help": "Flag indicating whether to perform evaluation during training"}
    )
    train_batch_size: int = field(
        default=64, metadata={"help": "Batch size for training"}
    )
    eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size for evaluation"}
    )
    num_train_epochs: float = field(
        default=5.0, metadata={"help": "Number of training epochs"}
    )
    learning_rate: float = field(
        default=5e-5, metadata={"help": "Learning rate for training"}
    )
    optimizer: Literal["Adam", "AdamW", "SGD"] = field(
        default="Adam", metadata={"help": "Optimizer choice (one of Adam, AdamW, or SGD)"}
    )
    device: Literal["cuda", "cpu"] = field(
        default="cuda", metadata={"help": "Device for training (cuda for GPU or cpu for CPU)"}
    )
    scheduler_type: Literal["linear", "cosine"] = field(
        default="linear", metadata={"help": "Type of learning rate scheduler (linear or cosine)"}
    )
    warmup_steps: int = field(
        default=0, metadata={"help": "Linear warmup over warmup_steps"}
    )
    warmup_ratio: float = field(
        default=0.0, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps"}
    )
    example_prompts: List[str] = field(
        default_factory=lambda: [
            "Photograph of a perfect face of a girl at sunset in 4K resolution.",
            "Hello Kitty eating a bird.",
            "UFO hovering over an African Jesus in a colorful style resembling Nigerian truck art.",
        ],
        metadata={"help": "List of example prompts for generating images or descriptions"},
    )
        
def train(args, generator, discriminator, tokenizer, processor):
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    
    generator.to(args.device)
    discriminator.to(args.device)
    
    examples = args.example_prompts
    tokenized_text = tokenizer(
            examples,
            truncation=True,
            padding="max_length",
            max_length=100,
            return_tensors="pt"
        ).to(args.device)
    
    embs = discriminator.encode("text", **tokenized_text)
    
    gens = generator(embs)
    
    for i, gen in enumerate(gens):
        
        draw_examples(gen, examples[i], args.output_dir, f"{i}")
        
    
    
            

if __name__ == "__main__":
    
    parser = HfArgumentParser((ModelArguments, DPRArguments, DataArguments, TrainArguments))
    model_args, dpr_args, data_args, train_args = parser.parse_args_into_dataclasses()
    
    generator = Generator.from_pretrained(**vars(model_args))
    
    discriminator = ImageTextRetrieval.from_pretrained(dpr_args.dpr_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(dpr_args.dpr_model_name_or_path)
    processor = AutoImageProcessor.from_pretrained(dpr_args.dpr_model_name_or_path)
    
    # if os.path.isdir(data_args.path):
    #     dataset = load_from_disk(data_args.path)
    # else:
    #     dataset = load_dataset(data_args.path, data_args.name, cache_dir=data_args.cache_dir)
        
    # if data_args.shuffle:
    #     dataset = dataset.shuffle()
        
    # def example_function(examples):

    #     tokenized_text = tokenizer(
    #         examples[data_args.text_column_name],
    #         truncation=True,
    #         padding="max_length",
    #         max_length=data_args.max_length,
    #         return_tensors="pt"
    #     )

    #     processed_image = processor(examples[data_args.image_column_name], return_tensors="pt")
    #     tokenized_text.update(processed_image)

    #     return tokenized_text

    # dataset = dataset.map(example_function, batched=True, remove_columns=dataset[data_args.train_split].column_names)
    
    
    train(train_args, generator, discriminator, tokenizer, processor)