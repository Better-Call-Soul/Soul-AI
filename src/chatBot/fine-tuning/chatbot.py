# Import Required Libraries
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

from utils import preprocess_data
from eval import Evaluation
from constants import *

class Chatbot_Finetune:
  def __init__(self, save_model=False):
    '''
    Load the dataset, initialize the model, and define the evaluation class
    '''
    self.save_model = save_model
    self.model = None
    self.tokenizer = None
    self.trainer = None
    self.dataset = None
    
    # dataset loading 
    self.load_dataset()

    # initialize model
    self.initialize_model()
    
    # evaluation initialization
    self.eval = Evaluation(dataset=self.dataset, sample_size=sample_size)

  def load_dataset(self):
    '''
    Load the dataset
    '''
    self.dataset = preprocess_data(
      file_name=data_path,
      col_name='text',
      max_sequence_length=11000,
      num_samples=None
    )
    print(f'data set loaded successfully, dataset size = {len(self.dataset)}')
    
  def initialize_model(self):
    '''
    Initialize the model, tokenizer, and training arguments
    '''
    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    # Load base model
    self.model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map
    )
    self.model.config.use_cache = False
    self.model.config.pretraining_tp = 1

    # Load LLaMA tokenizer
    self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.tokenizer.padding_side = "right"

    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to="tensorboard"
    )

    # Set supervised fine-tuning parameters
    self.trainer = SFTTrainer(
        model=self.model,
        train_dataset=self.dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=self.tokenizer,
        args=training_arguments,
        packing=packing,
    )
    
    
  def train(self):
    # Evaluate the model before training
    self.eval.evaluate_model(model=self.model, tokenizer=self.tokenizer)

    # Train model
    self.trainer.train()
    
    # Evaluate the model after training
    self.eval.evaluate_model(model=self.model, tokenizer=self.tokenizer)

    if self.save_model:
      self.upload_model()
    
  def upload_model(self):
    # Save trained model
    self.trainer.model.save_pretrained(model_path + new_model)
    self.trainer.tokenizer.save_pretrained(model_path + new_model)