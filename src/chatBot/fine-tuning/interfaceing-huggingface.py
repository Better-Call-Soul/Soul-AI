import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    logging,
)

model_path = 'nouralmulhem/Llama-2-7b-chat-finetune'
device_map = {"": 0}
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

class chatbot:
  def __init__(self):
    self.instructions = 'you are a therapist, ask a question and be concise'
    self.pipe = self.load_pipeline()
    
  def load_pipeline(self):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    
    # Ignore warnings
    logging.set_verbosity(logging.CRITICAL)

    # Run text generation pipeline with our next model
    return pipeline(task="text-generation", 
                    model=model, 
                    tokenizer=tokenizer, 
                    max_new_tokens=256, 
                    # max_length=200
                    )

  # def get_prompt(self, input, history):
  #   return f"""<s>[INST]
  #       instructions: {self.instructions}
  #       conversation history:
  #       {history}
  #       input:
  #       {input} [/INST]"""    

  def get_prompt(self, input, history):
    return f"""<s>[INST] <<SYS>>
        instructions: {self.instructions}
        <</SYS>>
        Current conversation:
        {history}
        Human: {input}
        AI: [/INST]"""
        #         input:
        # {input} [/INST]

  def predict(self, input, history):
    # print(self.get_prompt(input, history))
    result = self.pipe(self.get_prompt(input, history))
    response = result[0]['generated_text'].split("[/INST]")[-1].strip()
    return response
  
chat = chatbot()

history = ""
while True:
    user_input = input("Prompt (press 'q' to quit): ")
    if user_input.lower() == 'q':
        break
    else:
        response = chat.predict(user_input, history)
        history += f"Human: {user_input}\nAI: {response}\n"
        print("AI:", response)