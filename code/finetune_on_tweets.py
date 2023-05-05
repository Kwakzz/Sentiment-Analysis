# import libraries
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer
from datasets import load_dataset, load_metric
import numpy as np

# using the distilbert model's configuration and tokenizer
original_model_name = "social_media_sa"
model_tokenizer = "social_media_sa/tokenizer"
architecture = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=original_model_name)
tokenizer = AutoTokenizer.from_pretrained(model_tokenizer)

# load dataset
data_files = {"train": "twitter-train.csv", "test": "twitter-test.csv"}
twitter_dataset = load_dataset("csv", data_files=data_files)

# function for processing model input
def process_input (example):
    return tokenizer (example["Sequence"], truncation=True)


# collecting 5000 random samples
dataset_for_training = twitter_dataset["train"].shuffle(seed=42).select(range (5000))
dataset_for_testing = twitter_dataset["test"].shuffle(seed=42).select(range (5000))

# processed splits of the dataset
processed_train_split = dataset_for_training.map(process_input, batched = True)
processed_test_split = dataset_for_testing.map(process_input, batched = True)

# pad input to speed up processing
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# function for determining model's training metrics
def compute_metrics(eval_pred):
    load_accuracy = load_metric("accuracy")
    load_f1 = load_metric("f1")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
    return {"accuracy": accuracy, "f1": f1}


new_model_name = "scoial_media_sa_model_finetuned"
 
 # setting training hyperparameters (parameters for controlling learning process)
training_args = TrainingArguments(
   output_dir=new_model_name,
   learning_rate=2e-5,
   per_device_train_batch_size=16,
   per_device_eval_batch_size=16,
   num_train_epochs=2,
   weight_decay=0.01,
   save_strategy="epoch",
   push_to_hub=True,
)
 
  # creating trainer object
trainer = Trainer(
   model=architecture,
   args=training_args,
   train_dataset=processed_train_split,
   eval_dataset=processed_test_split,
   tokenizer=tokenizer,
   data_collator=data_collator,
   compute_metrics=compute_metrics,
)

# train model
trainer.train()