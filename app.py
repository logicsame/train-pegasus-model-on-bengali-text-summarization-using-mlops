import streamlit as st
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# Caching the model and tokenizer loading to speed up the app
@st.cache_resource
def load_model_and_tokenizer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-large')
    model.load_state_dict(torch.load('pegasus_bengali_epoch_2.pt'))  # Load the saved model
    model = model.to(device)
    
    tokenizer = PegasusTokenizer('bengali_tokenizer.model', do_lower_case=False)
    
    return model, tokenizer, device

# Summarization function
def summarize_bengali_text(model, tokenizer, device, text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
    
    # Generate the summary
    summary_ids = model.generate(
        inputs['input_ids'], 
        max_length=70, 
        num_beams=5, 
        length_penalty=2.0, 
        early_stopping=True
    )
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Streamlit UI design
def main():
    st.set_page_config(page_title="Bengali Text Summarizer", layout="centered", initial_sidebar_state="collapsed")
    
    # Title and description
    st.title("📜 Bengali Text Summarizer")
    st.write("Enter a Bengali paragraph and get a concise summary instantly.")
    
    # Sidebar info
    with st.sidebar:
        st.header("How to Use:")
        st.write("1. Enter your Bengali text in the input box below.")
        st.write("2. Click on **Summarize**.")
        st.write("3. Get the summary of your input text.")
    
    # User input
    bengali_text = st.text_area("Enter Bengali Text:", height=200, placeholder="আপনার বাংলা টেক্সট এখানে লিখুন...")

    # Load model and tokenizer
    model, tokenizer, device = load_model_and_tokenizer()

    # Summarization button
    if st.button("Summarize"):
        if bengali_text.strip():
            # Summarize the text
            with st.spinner("Summarizing..."):
                summary = summarize_bengali_text(model, tokenizer, device, bengali_text)
            
            # Display the result
            st.subheader("Summary:")
            st.write(summary)
        else:
            st.warning("Please enter some Bengali text to summarize!")
if __name__ == "__main__":
    main()
