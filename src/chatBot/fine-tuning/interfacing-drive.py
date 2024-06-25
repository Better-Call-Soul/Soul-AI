import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    logging,
)
from peft import PeftModel

model_name = "NousResearch/Llama-2-7b-chat-hf"
new_model = "Llama-2-7b-chat-finetune"
model_path = '/content/drive/MyDrive/Graduation-Project/models/chatbot/'
device_map = {"": 0}

class chatbot:
  def __init__(self):
    self.history = ''
    self.instructions = 'you are a therapist, ask a question and be concise'
    self.pipe = self.load_pipeline()
    
  def load_pipeline(self):
    # Reload model in FP16 and merge it with LoRA weights
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    model = PeftModel.from_pretrained(base_model, model_path + new_model)
    model = model.merge_and_unload()

    # Reload tokenizer to save it
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Ignore warnings
    logging.set_verbosity(logging.CRITICAL)

    # Run text generation pipeline with our next model
    return pipeline(task="text-generation", 
                    model=model, 
                    tokenizer=tokenizer, 
                    max_new_tokens=256, 
                    # max_length=200
                    )
    
  def get_prompt(self, input):
    return f"""<s>[INST]
        instructions: {self.instructions}
        conversation history:
        {self.history}
        input:
        {input} [/INST]"""

  def predict(self, input):
    # print(self.get_prompt(input))
    result = self.pipe(self.get_prompt(input))
    response = result[0]['generated_text'].split("[/INST]")[-1].strip()
    self.history += f'''
        User: {input}
        Ai: {response}
        '''

    return response
  

chat = chatbot()

while True:
    user_input = input("Prompt (press 'q' to quit): ")
    if user_input.lower() == 'q':
        break
    else:
        response = chat.predict(user_input)
        print("AI:", response)