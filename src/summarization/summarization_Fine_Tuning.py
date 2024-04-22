
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import pandas as pd
import numpy as np
import nltk
from datasets import load_metric
import torch

## Class for fine-tuning the model
class SummarizationFineTuning:
    def __init__(self,dataset_path):
        self.dataset_path=dataset_path
        self.metric = load_metric("rouge")
    
    # Process the data for training
    def process(self,data,tokenizer,max_input_length,max_target_length):
        inputs = [str(doc) for doc in data["dialogue"]]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
        if isinstance(data["summary"], pd.Series):
            summaries = data["summary"].tolist()  # convert Series to list
        else:
            summaries = [str(summary) for summary in data["summary"]]  # convert each summary to string

        # Setup the tokenizer for summary process
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(summaries, max_length=max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    ## Load data for training
    def load_data_training(self,tokenizer,max_input_length,max_target_length):
        train = pd.read_csv(self.dataset_path+"train.csv")
        train=self.process(train,tokenizer,max_input_length,max_target_length)
        val=pd.read_csv(self.dataset_path+"validation.csv")
        val=self.process(val,tokenizer,max_input_length,max_target_length)
        return train,val
    
    ## Compute Rouge score during validation
    def compute_metrics(self,eval_pred,tokenizer):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}
    
    # Train the model
    def train(self,model="facebook/bart-large-xsum",max_input_length=512,max_target_length=128,batch_size=4):
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForSeq2SeqLM.from_pretrained(model)
        train,val=self.load_data_training(tokenizer,max_input_length,max_target_length)
        # Define training args
        args = Seq2SeqTrainingArguments(
            "dialogue-summarization", #
            evaluation_strategy = "epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=2,
            weight_decay=0.01,
            save_total_limit=2,
            num_train_epochs=3,
            predict_with_generate=True,
            fp16=True,
        )
        collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=train,
        eval_dataset=val,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=self.compute_metrics
        )
        trainer.train()
        trainer.evaluate()
        return model,tokenizer
    
    ## test the model
    def test(self,text,model,tokenizer):
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            output = model.generate(input_ids)
        summary = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        return summary
    
 