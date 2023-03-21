import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelWithLMHead


def generate_text(prompt):
    model_name = "newcave/kogpt2-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelWithLMHead.from_pretrained(model_name)

    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text


st.title("KoChatGPT")
prompt = st.text_input("프롬프트를 입력하세요.")
if st.button("생성"):
    generated_text = generate_text(prompt)
    st.write(generated_text)
