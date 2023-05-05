# import modules for loading and deploying model.
from transformers import pipeline, AutoModelForSequenceClassification
import gradio as gr

# batch of sequences whose sentiments are to be analysed.
text = ["The chips are ugh.",
        "yikes.",
        "The chicken is yummy."]

# assigning model architecture to variable
model = AutoModelForSequenceClassification.from_pretrained("social_media_sa_finetuned_2")

# creating a sentiment analysis pipeline
sa = pipeline("sentiment-analysis", model=model, tokenizer = "social_media_sa_finetuned_2/tokenizer")
# print(sa(text))

# returns the sentiment of sequence.
def predict(sequence):
      return sa (sequence)

# header for model UI
description = """
Analyze the Sentiment of Your Social Media Comments
"""

# gradio user interface for model
interface = gr.Interface(
  fn=predict, 
  description=description,
  inputs='text',
  outputs='text',
  examples=["Kwaku loves food"]
)

# launch gradio user interface
interface.launch(share=True)