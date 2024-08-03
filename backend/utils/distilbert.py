import torch

# download IMDB dataset
from datasets import load_dataset
imdb = load_dataset("imdb")

# truncate datasets
small_train_dataset = imdb["train"].shuffle(seed=42).select([i for i in list(range(3000))])
small_test_dataset = imdb["test"].shuffle(seed=42).select([i for i in list(range(300))])

# distilbert tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# prepare text
def preprocess_function(examples):
   return tokenizer(examples["text"], truncation=True)
 
tokenized_train = small_train_dataset.map(preprocess_function, batched=True)
tokenized_test = small_test_dataset.map(preprocess_function, batched=True)

# convert training samples to pytorch tensors with padding
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# define distilbert as base model
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# define metrics for fine tuned model
import numpy as np
from datasets import load_metric
 
def compute_metrics(eval_pred):
   load_accuracy = load_metric("accuracy")
   load_f1 = load_metric("f1")
  
   logits, labels = eval_pred
   predictions = np.argmax(logits, axis=-1)
   accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
   f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
   return {"accuracy": accuracy, "f1": f1}

# define trainer
from transformers import TrainingArguments, Trainer
 
repo_name = "finetuning-sentiment-model-3000-samples"
 
training_args = TrainingArguments(
   output_dir=repo_name,
   learning_rate=2e-5,
   per_device_train_batch_size=16,
   per_device_eval_batch_size=16,
   num_train_epochs=2,
   weight_decay=0.01,
   save_strategy="epoch",
   push_to_hub=True,
)
 
trainer = Trainer(
   model=model,
   args=training_args,
   train_dataset=tokenized_train,
   eval_dataset=tokenized_test,
   tokenizer=tokenizer,
   data_collator=data_collator,
   compute_metrics=compute_metrics,
)

# train model
trainer.train()

# compute evaluation metrics
trainer.evaluate()

# Save the trained model
model.save_pretrained('./distilbert-sentiment')
tokenizer.save_pretrained('./distilbert-sentiment')

# Load the saved model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained('./distilbert-sentiment')
tokenizer = AutoTokenizer.from_pretrained('./distilbert-sentiment')

def predict(text):
    # Preprocess input text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    
    # Ensure the model is in evaluation mode
    model.eval()

    # Run the model and get predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Convert logits to probabilities
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
    # Get predicted class
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    
    # Return result
    return predicted_class, probabilities[0].tolist()

text = "I love this movie!"
predicted_class, probabilities = predict(text)
print(f'Predicted class: {predicted_class}')
print(f'Probabilities: {probabilities}')