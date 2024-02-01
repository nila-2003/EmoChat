import streamlit as st
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer, RobertaTokenizer, RobertaForSequenceClassification
import torch
import warnings

warnings.filterwarnings("ignore", message="Some weights of the model checkpoint")

sa = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=0 if torch.cuda.is_available() else -1)

gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

roberta_model = RobertaForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
roberta_tokenizer = RobertaTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

def generate_response_with_sentiment(user_input):
    sentiment_result = sa(user_input)
    sentiment_label = sentiment_result[0]['label']
    
    if sentiment_label == "POSITIVE":
        user_input += " |positive| "
    elif sentiment_label == "NEGATIVE":
        user_input += " |negative| "
    else:
        user_input += " |neutral| "

    input_ids = gpt_tokenizer.encode(user_input, return_tensors="pt")
    output = gpt_model.generate(input_ids, max_length=150, num_return_sequences=1)
    response = gpt_tokenizer.decode(output[0], skip_special_tokens=True)

    return response

def main():
    st.title("Sentimental Chatbot with GPT-2")
    user_input = st.text_area("Enter your message:")
    if st.button("Generate Response"):
        if user_input:
            response = generate_response_with_sentiment(user_input)
            st.text("Bot's Response: " + response)

if __name__ == "__main__":
    main()
