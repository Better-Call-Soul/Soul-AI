import random
from tqdm import tqdm
import evaluate
from transformers import pipeline

class Evaluation:
  def __init__(self, dataset, sample_size=4):
    self.dataset = dataset
    self.sample_size = sample_size
    
    processed_dataset = self.dataset.map(self.extract_qa_pairs)
    X = processed_dataset["input"]
    y = processed_dataset["response"]

    # Get random indices
    sampled_indices = self.get_random_indices(X)

    # Get the corresponding random_X and random_y
    self.random_X = [X[i] for i in sampled_indices]
    self.random_y = [y[i] for i in sampled_indices]

    # print("Random Samples of Inputs (random_X):", random_X[:5])
    # print("Random Samples of Responses (random_y):", random_y[:5])

  
  def extract_qa_pairs(self, example):
    input_text = example["text"].split("[INST]")[1].split("[/INST]")[0].strip()
    response_text = example["text"].split("[/INST]")[1].replace("</s>", "").strip()
    return {"input": input_text, "response": response_text}
  
  def get_random_indices(self, X):
    # Filter X to only include items with length < 200 tokens
    filtered_indices = [i for i, x in enumerate(X) if len(x) < 200]

    # Take a random sample from the filtered indices
    return random.sample(filtered_indices, self.sample_size)

  def get_predictions(self, X, pipe):
      predictions=[]

      for row in tqdm(X):
          result = pipe(f"""<s>[INST]{row}[/INST]""")

          result = result[0]['generated_text'].split("[/INST]")[-1].strip()

          predictions.append(result)

          ########################################
          ########## Model Generation ############
          ########################################

          # input_text = f"<s>[INST]{row}[/INST]"

          # Tokenize the input text
          # inputs = tokenizer(input_text, return_tensors="pt")

          # Generate a response
          # output = model.generate(
          #     inputs["input_ids"].to(model.device),
          #     max_length=512, #50
          #     num_return_sequences=1,
          #     no_repeat_ngram_size=2,
          #     top_k=50,
          #     top_p=0.95,
          #     temperature=0.7,
          # )

          # Decode the generated text
          # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

          # Remove [INST] and [/INST] tokens
          # generated_text = generated_text.replace("[INST]", "").replace("[/INST]", "").strip()

          # predictions.append(generated_text)

          # print(f"""
          # ----------------------------------------------------------
          # {generated_text}
          # ----------------------------------------------------------
          # """)

      return predictions
  
  def evaluation_metrics(self, predictions, random_y):
    sacrebleu = evaluate.load("sacrebleu")
    sacrebleu_results=sacrebleu.compute(predictions=predictions, references=random_y)

    bleu = evaluate.load('bleu')
    scores = bleu.compute(predictions=predictions, references=random_y)

    print(f'bleu score for {self.sample_size} sample = {sacrebleu_results["score"]} , score = {scores["bleu"]}')


  def evaluate_model(self, model, tokenizer):
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
    predictions = self.get_predictions(self.random_X, pipe)
    self.evaluation_metrics(predictions, self.random_y)

# eval = Evaluation(dataset=dataset, sample_size=4)
# eval.evaluate(model=model, tokenizer=tokenizer)