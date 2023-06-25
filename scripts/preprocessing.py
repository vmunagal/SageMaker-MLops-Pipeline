
import argparse
import logging
import os 
import sys 
import subprocess
import pandas as pd 
import numpy as np




def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])



if __name__=="__main__":
    
    parse=argparse.ArgumentParser()
    
    parse.add_argument("--model_id", type=str)
    parse.add_argument("--dataset_name", type=str)
    parse.add_argument("--pytorch_version", type=str)
    
    install("fsspec==2023.1.0")
    install("s3fs==0.4.2")
    install("scikit-learn==0.22.1")
    install("datasets[s3]")
    install("datasets==2.12.0")
    install("transformers==4.30.1")
    install("torch==1.13.1")

    args, _ = parse.parse_known_args()
    
    from sklearn.model_selection import train_test_split
    from datasets import load_from_disk , Dataset , load_metric
    from transformers import AdamW,T5ForConditionalGeneration, T5TokenizerFast as T5Tokenizer , AutoTokenizer
    
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
  
    # lets make pandas to read data from s3 
    df=pd.read_csv(args.dataset_name)
    
    df_train,df_test=train_test_split(df,test_size=0.2,random_state=42)
    
    new_df_train=pd.concat([df_train,df_test],ignore_index=True) 
    
    new_df_train.to_csv("train.csv",index=False)
    df_test.to_csv("test.csv",index=False)
    
    train_data=pd.read_csv("train.csv")
    test_data=pd.read_csv("test.csv")
    
    
    train_dataset = Dataset.from_pandas(train_data)
    test_dataset = Dataset.from_pandas(test_data)
    
    logger.info(f"The lenght of the train dataset : {len(train_dataset)}")
    
    logger.info(f"The lenght of test dataset : {len(test_dataset)}")
    
    tokenizer=AutoTokenizer.from_pretrained(args.model_id)
    
    # encoder and decoder
    encoder_length =  128
    decoder_length =  512
    
    def preprocess_data(data):
        
           
        
        inputs = ["question: " + item for item in data["Question"]]
        
        outputs = ["answer: " + str(ans) for ans in data["Answer"]]
        

        model_inputs = tokenizer(inputs,max_length=encoder_length,padding='max_length', truncation=True,add_special_tokens=True)
        
    
        labels= tokenizer(outputs, max_length=decoder_length,padding='max_length', truncation=True,add_special_tokens=True)
        
        
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
        
        model_inputs["labels"]=labels["input_ids"]
        
        return model_inputs
    
    train_dataset=train_dataset.map(preprocess_data,batched=True,remove_columns=["Question","Answer"])
    
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    test_dataset=test_dataset.map(preprocess_data,batched=True,remove_columns=["Question","Answer"])
    
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    train_dataset.save_to_disk("/opt/ml/processing/train")
    
    test_dataset.save_to_disk("/opt/ml/processing/test")
    
