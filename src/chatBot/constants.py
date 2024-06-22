# Specify Our Training Parameters
model_name = "NousResearch/Llama-2-7b-chat-hf"
new_model = "Llama-2-7b-chat-finetune"
model_path = '/content/drive/MyDrive/Graduation-Project/models/chatbot/'
data_path = '/content/drive/MyDrive/Graduation-Project/Dataset/Dataset/counsel_chat/formatted_data.csv'

# QLoRA parameters
lora_r = 64
lora_alpha = 16
lora_dropout = 0.1

# bitsandbytes parameters
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

# TrainingArguments parameters
output_dir = "./results"
num_train_epochs = 2
fp16 = False
bf16 = False
per_device_train_batch_size = 4
# per_device_eval_batch_size = 4
gradient_accumulation_steps = 1
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "cosine" # constant
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 0
logging_steps = 25

# SFT parameters
max_seq_length = None
packing = False
device_map = {"": 0}

# Set sample size for evaluation
sample_size = 4