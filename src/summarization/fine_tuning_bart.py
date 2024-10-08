
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import pandas as pd
import numpy as np
import nltk
from datasets import load_metric,load_dataset
import torch
from preprocess.preprocess import Preprocessor

## Class for fine-tuning the model
class FineTuningSummarization:
    def __init__(self,dataset="samsum"):
        self.dataset= load_dataset(dataset)
        # check if dataset is valid
        if self.dataset is None:
            raise ValueError("Dataset not found")
        self.metric = load_metric("rouge")
        self.preprocess=Preprocessor()
        
    # Clean the text
    def clean_text(self,text):
        clean_text=self.preprocess.clean(text,
            ["remove_nonascii_diacritic",
            "remove_emails","clean_html",
            "remove_url","replace_repeated_chars","expand_sentence"]
            ,"")[0]
        return clean_text
    
    # Process the data for training
    def process(self,data):
        inputs = [str(doc) for doc in data["dialogue"]]
        clean_inputs=[self.clean_text(doc) for doc in inputs]
        model_inputs = self.tokenizer(clean_inputs, max_length=self.max_input_length, truncation=True)
        # if isinstance(data["summary"], pd.Series):
        #     summaries = data["summary"].tolist()  # convert Series to list
        # else:
        #     summaries = [str(summary) for summary in data["summary"]]  # convert each summary to string

        # Setup the tokenizer for summary process
        with self.tokenizer.as_target_tokenizer():
            clean_summary=[self.clean_text(doc) for doc in data["summary"]]
            labels = self.tokenizer(data[clean_summary], max_length=self.max_target_length, truncation=True)
            

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
   
    
    ## Compute Rouge score during validation
    def compute_metrics(self,eval_pred):

        predictions, labels = eval_pred
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}
    
    # Train the model
    def train(self,model="facebook/bart-large-xsum",max_input_length=512,max_target_length=128,batch_size=4):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForSeq2SeqLM.from_pretrained(model)
        self.max_input_length=max_input_length
        self.max_target_length=max_target_length
        self.batch_size=batch_size
        tokenized_dataset = self.dataset.map(self.process, batched=True)   
             
        # Define training args
        args = Seq2SeqTrainingArguments(
            "dialogue-summarization", # the name of output dir
            evaluation_strategy = "epoch", # how often the model will be evaluated during training
            learning_rate=2e-5, # controls the step size during training.
            per_device_train_batch_size=self.batch_size, # determines the batch size used for training 
            per_device_eval_batch_size=self.batch_size, # determines the batch size used for evaluation 
            gradient_accumulation_steps=2, # It accumulates gradients over multiple steps before updating the model weights
            weight_decay=0.01, # penalizes large weights to prevent overfitting. 
            save_total_limit=2, # limits the total number of checkpoints to save during training
            num_train_epochs=3, # This parameter specifies the number of epochs 
            predict_with_generate=True, # When set to True, it indicates that the model should use generation 
            fp16=True, # using 16-bit floating-point precision instead of the standard 32-bit.
        )
        
        print(self.tokenizer)
        collator = DataCollatorForSeq2Seq(self.tokenizer, model=model)
        trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=collator,
        tokenizer=self.tokenizer,
        compute_metrics=self.compute_metrics
        )
        trainer.train()
        trainer.evaluate()
        return model,self.tokenizer
    
    ## test the model
    def test(self,text,model,tokenizer):
        text=self.clean_text(text)
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            output = model.generate(input_ids)
        summary = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        return summary
    

if(__name__ == "__main__"):
    summ=FineTuningSummarization()
    model,tokenizer=summ.train()
    print(summ.test("The cat isn't in the box. The cat likes the box. The box is in the house. The house is in the city. The city is in the country. The country is in the world.",model,tokenizer))
     