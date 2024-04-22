
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import pandas as pd
import numpy as np
import nltk
from datasets import load_metric,load_dataset
import torch

## Class for fine-tuning the model
class SummarizationFineTuning:
    def __init__(self,dataset_path):
        self.dataset_path=dataset_path
        self.metric = load_metric("rouge")
    
    # Process the data for training
    def process(self,data):
        inputs = [str(doc) for doc in data["dialogue"]]
        model_inputs = self.tokenizer(inputs, max_length=self.max_input_length, truncation=True)
        # if isinstance(data["summary"], pd.Series):
        #     summaries = data["summary"].tolist()  # convert Series to list
        # else:
        #     summaries = [str(summary) for summary in data["summary"]]  # convert each summary to string

        # Setup the tokenizer for summary process
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(data["summary"], max_length=self.max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    ## Load data for training
    def load_data_training(self):
        train = load_dataset("csv", data_files="train.csv")["train"]
        val = load_dataset("csv", data_files="validation.csv")["train"]
        # train = pd.read_csv(self.dataset_path+"train.csv")
        train=self.process(train)
        # val=pd.read_csv(self.dataset_path+"validation.csv")
        val=self.process(val)
        return train,val
    
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
        train,val=self.load_data_training()
        print(train.keys())
        # Define training args
        args = Seq2SeqTrainingArguments(
            "dialogue-summarization", #
            evaluation_strategy = "epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=2,
            weight_decay=0.01,
            save_total_limit=2,
            num_train_epochs=3,
            predict_with_generate=True,
            fp16=True,
        )
        print(self.tokenizer)
        collator = DataCollatorForSeq2Seq(self.tokenizer, model=model)
        trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=train,
        eval_dataset=val,
        data_collator=collator,
        tokenizer=self.tokenizer,
        compute_metrics=self.compute_metrics
        )
        trainer.train()
        trainer.evaluate()
        return model,self.tokenizer
    
    ## test the model
    def test(self,text,model,tokenizer):
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            output = model.generate(input_ids)
        summary = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        return summary
    
 