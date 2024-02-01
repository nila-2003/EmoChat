import streamlit as st
from transformers import GPT2LMHeadModel
from transformers import pipeline
from transformers import GPT2Tokenizer
import openai
import torch
import warnings

openai.api_key = "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"

warnings.filterwarnings("ignore", message="Some weights of the model checkpoint")

sa = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=0 if torch.cuda.is_available() else -1)
# GPT2
gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# History tab
conversation_history = []

def generate_response_with_sentiment(user_input):
    # Sentiment Analysis
    sentiment_result = sa(user_input)
    sentiment_label = sentiment_result[0]['label']
    
    if sentiment_label == "POSITIVE":
        user_input += " |positive| "
    elif sentiment_label == "NEGATIVE":
        user_input += " |negative| "
    else:
        user_input += " |neutral| "
    # Generate response using gpt-2

    input_ids = gpt_tokenizer.encode(user_input, return_tensors="pt")
    output = gpt_model.generate(input_ids, max_length=150, num_return_sequences=1)
    response = gpt_tokenizer.decode(output[0], skip_special_tokens=True)

    # Fine Tuning the response using GPT-3.5 openai API.
    refined_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",  
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{user_input}"},
        ],
        max_tokens=150
    )

    refined_response_text = refined_response['choices'][0]['message']['content'].strip()
    conversation_history.append({"user": user_input, "emobot": refined_response_text})

    return refined_response_text

def main():
    st.title("Welcome To EmoChat!!")
    user_input = st.text_area("Let's Chat!!!")
    if st.button("Send"):
        if user_input:
            response = generate_response_with_sentiment(user_input)
            st.text("EmoChat: " + response)
            
    # Displaying the history tab
    st.sidebar.title("Your Past Conversations!")
    for conversation in conversation_history:
        st.sidebar.text(f"User: {conversation['user']}")
        st.sidebar.text(f"EmoChat: {conversation['emobot']}")
        st.sidebar.text("------")

if __name__ == "__main__":
    main()
