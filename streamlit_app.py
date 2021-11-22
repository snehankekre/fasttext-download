import streamlit as st
import fasttext

# Cache trained model
@st.experimental_singleton
def get_model():
    model = fasttext.train_supervised(input="cooking.train")
    return model

# Save trained model to file
def save_model(model, path="saved_model.bin"):
    model.save_model(path)

model = get_model()
save_model(model, path="saved_model.bin")

# Download saved trained model
with open("saved_model.bin", "rb") as f:
    btn = st.download_button(
        label="Download trained fasttext model",
        data=f,
        file_name="fasttext_model.bin"
    )
