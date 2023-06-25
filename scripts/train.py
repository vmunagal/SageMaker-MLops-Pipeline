import torch 
import gc 
import pandas as pd 
import numpy as np
from transformers import AdamW, AutoModelForSeq2SeqLM, Seq2SeqTrainer , Seq2SeqTrainingArguments , AutoTokenizer , DataCollatorForSeq2Seq
import os 
import warnings
import sys 
import argparse
import logging
import json
from datasets import load_from_disk , Dataset , load_metric
from transformers.trainer_utils import get_last_checkpoint

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings('ignore')

if __name__=='__main__':
    
    parse = argparse.ArgumentParser()
    
    ## hyperparameters 
    parse.add_argument("--epochs",default=1)
    parse.add_argument("--train_batch_size",default=4)
    parse.add_argument("--test_batch_size",default=2)
    parse.add_argument("--train_grad_accumulation",default=1)
    parse.add_argument("--test_grad_accumulation",default=1)
    parse.add_argument("--learning_rate",default=3e-3)
    parse.add_argument("--warmup_steps",default=1000)
    parse.add_argument("--model_id",default="t5-base")
    ## sagemaker values 
    parse.add_argument("--output_data_dir",default=os.environ["SM_OUTPUT_DATA_DIR"])
    parse.add_argument("--model_dir",default=os.environ["SM_MODEL_DIR"])
    parse.add_argument("--n_gpus",default=os.environ["SM_NUM_GPUS"])
    parse.add_argument("--training_dir",default=os.environ["SM_CHANNEL_TRAIN"])
    parse.add_argument("--test_dir",default=os.environ["SM_CHANNEL_TEST"])
    
    args,_ =parse.parse_known_args()
    
        # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    

    
    train_dataset=load_from_disk(args.training_dir)
    test_dataset=load_from_disk(args.test_dir)
    
    tokenizer=AutoTokenizer.from_pretrained(args.model_id)
    
    
    metric = load_metric('rouge')
    

    def compute_metrics(eval_preds):
        
        preds, labels = eval_preds
        
        if isinstance(preds, tuple):
            preds = preds[0]
    
    # Replace -100 in the labels as we can't decode them.
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True ,clean_up_tokenization_spaces=True)  
    
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        
        return {'rouge1':result['rouge1'].mid.fmeasure,
                'rouge2':result['rouge2'].mid.fmeasure,
                'rougeL':result['rougeL'].mid.fmeasure,
                'rougeLsum':result['rougeLsum'].mid.fmeasure }
        

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id)
    
    label_pad_token_id = -100

    data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
     )
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"{args.output_data_dir}",
        overwrite_output_dir=True if get_last_checkpoint(args.output_data_dir) is not None else False,
        num_train_epochs=int(args.epochs),
        do_eval=True,
        generation_max_length=512, # decoder lenght 
        logging_strategy="steps",
        logging_steps=500,
        gradient_accumulation_steps=3,
        predict_with_generate=True,
        per_device_train_batch_size=int(args.train_batch_size),
        per_device_eval_batch_size=int(args.test_batch_size),
        warmup_steps=args.warmup_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{args.output_data_dir}/logs",
        learning_rate=float(args.learning_rate),
        load_best_model_at_end=True,
        report_to="tensorboard"
    )
    
    
        # create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # train model
    if get_last_checkpoint(args.output_data_dir) is not None:
        logger.info("***** continue training *****")
        last_checkpoint = get_last_checkpoint(args.output_data_dir)
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()
    
    trainer.train()

    # evaluate model
    eval_result = trainer.evaluate(eval_dataset=test_dataset)
    

     # writes eval result to file which can be accessed later in s3 ouput
    with open(os.path.join(os.environ["SM_MODEL_DIR"], "evaluation.json"), "w") as writer:
        print(f"***** Eval results *****")
        writer.write(json.dumps(eval_result))

    # Saves the model to s3 uses os.environ["SM_MODEL_DIR"] to make sure checkpointing works
    trainer.save_model(os.environ["SM_MODEL_DIR"]) 
    tokenizer.save_pretrained(os.environ["SM_MODEL_DIR"])
