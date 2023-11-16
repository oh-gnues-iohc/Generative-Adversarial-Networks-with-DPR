from typing import List, Optional, Literal
from transformers import AutoImageProcessor, ResNetForImageClassification, AutoTokenizer, BertModel
import torch
from models.generator import Generator, GeneratorConfig
from models.dpr import ImageTextRetrieval, ImageTextRetrievalConfig
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from transformers import HfArgumentParser
import logging
import os
from dataclasses import dataclass, field
import transformers
from utils.draw import draw_examples
from utils.train_utils import get_warmup_steps
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable

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
    is_freeze: bool = field(
        default=True, metadata={"help": "Specify whether to freeze the DPR model and not train the discriminator"}
    )
    output_dir: str = field(
        default="runs/", metadata={"help": "Output directory for training results"}
    )
    do_eval: bool = field(
        default=False, metadata={"help": "Flag indicating whether to perform evaluation during training"}
    )
    train_batch_size: int = field(
        default=32, metadata={"help": "Batch size for training"}
    )
    eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size for evaluation"}
    )
    num_train_epochs: float = field(
        default=10.0, metadata={"help": "Number of training epochs"}
    )
    learning_rate: float = field(
        default=5e-5, metadata={"help": "Learning rate for training"}
    )
    optimizer: Literal["Adam", "AdamW", "SGD"] = field(
        default="AdamW", metadata={"help": "Optimizer choice (one of Adam, AdamW, or SGD)"}
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
    save_term: int = field(
        default=50, metadata={"help": "Specifies the interval (in some unit of time) at which data is saved or backed up. The value '10' represents the default save interval."}
    )
    example_prompts: List[str] = field(
        default_factory=lambda: [
            "Photograph of a perfect face of a girl at sunset in 4K resolution.",
            "Hello Kitty eating a bird.",
            "UFO hovering over an African Jesus in a colorful style resembling Nigerian truck art.",
        ],
        metadata={"help": "List of example prompts for generating images or descriptions"},
    )
    
def loss(v1: torch.Tensor, v2: torch.Tensor, positive_idx_per_question: list=None):
    score = torch.matmul(v1, torch.transpose(v2, 0, 1))
    if len(v1.size()) > 1:
        q_num = v1.size(0)
        score = score.view(q_num, -1)
        
    softmax_scores = F.log_softmax(score, -1)
    if not positive_idx_per_question :
        positive_idx_per_question = list(range(v1.size(0)))

    loss = F.nll_loss(
        softmax_scores,
        torch.tensor(positive_idx_per_question).to(softmax_scores.device),
        reduction="mean",
    )
    
    return loss


def train(args, generator, discriminator, tokenizer, processor, dataset):
    
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
        
    optim_cls = torch.optim.AdamW if args.optimizer == "AdamW" else torch.optim.Adam
    
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, collate_fn=transformers.default_data_collator)
    
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
    
    # TODO if args.is_freeze:
    # if args.is_freeze:
    #     for param in discriminator.parameters():
    #         param.requires_grad = False
    
    optimizer = optim_cls(generator.parameters(), betas=(0.5, 0.999), lr=args.learning_rate)
    optimizer_D = optim_cls(discriminator.parameters(), lr=args.learning_rate)
    num_training_steps = args.num_train_epochs * len(dataloader)
    scheduler = transformers.get_scheduler(args.scheduler_type, optimizer=optimizer, 
                                            num_warmup_steps=get_warmup_steps(args.warmup_steps, args.warmup_ratio, num_training_steps), 
                                            num_training_steps=num_training_steps)
    
    scheduler_D = transformers.get_scheduler(args.scheduler_type, optimizer=optimizer_D, 
                                            num_warmup_steps=get_warmup_steps(args.warmup_steps, args.warmup_ratio, num_training_steps), 
                                            num_training_steps=num_training_steps)
    g_loss = nn.MSELoss()
    for epoch in range(int(args.num_train_epochs)):
        with torch.no_grad():
            embs = discriminator.encode("text", **tokenized_text)
            outputs = generator(embs)
            for i, output in enumerate(outputs):
                draw_examples(output, examples[i], args.output_dir, f"epoch_{epoch}_{i}")
            if epoch % args.save_term == 0:
                generator.save_pretrained(os.path.join(args.output_dir, f"epoch_{epoch}"))
                
        total_loss_g = 0.0
        total_loss_d = 0.0
        loss_g, loss_d = 0., 0. 
        
        torch.cuda.empty_cache()
            
        for param in discriminator.parameters():
            param.requires_grad = True
        for param in generator.parameters():
            param.requires_grad = False

        for batch in tqdm(dataloader):
            optimizer_D.zero_grad()
            
            text_embs = discriminator.encode("text", input_ids=batch["input_ids"].to(args.device), 
                                    token_type_ids=batch["token_type_ids"].to(args.device), 
                                    attention_mask=batch["attention_mask"].to(args.device))
            pixel_values = generator(text_embs)
            image_embs = discriminator.encode("image", pixel_values=pixel_values)
            origin_image_embs = discriminator.encode("image", pixel_values=batch["pixel_values"].to(args.device))
            v2 = torch.cat((origin_image_embs.unsqueeze(1), image_embs.unsqueeze(1)), dim=1)
            v2 = v2.view(-1, 512)
            label = [i for i in range(v2.size(0)) if i % 2 == 0]
            loss_d = loss(text_embs, v2, label)
            total_loss_d += loss_d.item()
            loss_d.backward()
            optimizer_D.step()
            scheduler_D.step()
        
        for param in discriminator.parameters():
            param.requires_grad = False
        for param in generator.parameters():
            param.requires_grad = True
                
        for batch in tqdm(dataloader):
            optimizer.zero_grad()
                
            text_embs = discriminator.encode("text", input_ids=batch["input_ids"].to(args.device), 
                                             token_type_ids=batch["token_type_ids"].to(args.device), 
                                             attention_mask=batch["attention_mask"].to(args.device))
            pixel_values = generator(text_embs)
            image_embs = discriminator.encode("image", pixel_values=pixel_values)
            origin_image_embs = discriminator.encode("image", pixel_values=batch["pixel_values"].to(args.device))
            
            target = torch.tensor([0.] * image_embs.size(0)).to(image_embs.device)
            score = torch.matmul(image_embs, torch.transpose(origin_image_embs, 0, 1))
            log_score = torch.diagonal(-1 * F.log_softmax(score, dim=-1))
            loss_g = g_loss(log_score, target)
            
            total_loss_g += loss_g.item()
            loss_g.backward()
            optimizer.step()
            scheduler.step()
            
        print(f"{total_loss_g / len(dataloader)}, {total_loss_d / len(dataloader)}")
            

if __name__ == "__main__":
    
    parser = HfArgumentParser((ModelArguments, DPRArguments, DataArguments, TrainArguments))
    model_args, dpr_args, data_args, train_args = parser.parse_args_into_dataclasses()
    
    # generator = Generator.from_pretrained(**vars(model_args))
    generator = Generator(GeneratorConfig(num_layer=5, activation="gelu", img_size=224))
    
    # discriminator = ImageTextRetrieval.from_pretrained(dpr_args.dpr_model_name_or_path)
    discriminator = ImageTextRetrieval(ImageTextRetrievalConfig())
    discriminator.text_encoder = BertModel.from_pretrained("bert-base-uncased")
    
    tokenizer = AutoTokenizer.from_pretrained(dpr_args.dpr_model_name_or_path)
    processor = AutoImageProcessor.from_pretrained(dpr_args.dpr_model_name_or_path)
    
    if os.path.isdir(data_args.path):
        dataset = load_from_disk(data_args.path)
    else:
        dataset = load_dataset(data_args.path, data_args.name, cache_dir=data_args.cache_dir)
        
    if data_args.shuffle:
        dataset = dataset.shuffle()
        
    def example_function(examples):

        tokenized_text = tokenizer(
            examples[data_args.text_column_name],
            truncation=True,
            padding="max_length",
            max_length=data_args.max_length,
            return_tensors="pt"
        )

        processed_image = processor(examples[data_args.image_column_name], return_tensors="pt")
        tokenized_text.update(processed_image)

        return tokenized_text

    dataset = dataset.map(example_function, batched=True, remove_columns=dataset[data_args.train_split].column_names)
    
    
    train(train_args, generator, discriminator, tokenizer, processor, dataset[data_args.train_split])