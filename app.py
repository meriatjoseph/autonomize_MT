import streamlit as st
import torch
import torch.nn.functional as F
from cpg_predictor import CpGPredictor

# Load the full pre-trained model
MODEL_PATH = 'model.pkl'
model = torch.load(MODEL_PATH)  # Load the entire model
model.eval()  # Set to evaluation mode

# Helper function to one-hot encode DNA sequence
def one_hot_encode(sequence, alphabet='NACGT'):
    # Map DNA to integers
    dna2int = {a: i for i, a in enumerate(alphabet)}
    try:
        int_sequence = [dna2int[base] for base in sequence]
    except KeyError:
        raise ValueError("Invalid character in sequence. Please use only A, C, G, T, or N.")

    # One-hot encode the sequence
    one_hot = F.one_hot(torch.tensor(int_sequence), num_classes=len(alphabet)).float()
    return one_hot

# Streamlit App
st.title("CpG Site Predictor")
st.write("Enter a DNA sequence to predict the number of CpG sites.")

# Input field for DNA sequence
sequence = st.text_input("DNA Sequence (e.g., ACGTACGTACGT):", "")

if sequence:
    try:
        # Encode the input DNA sequence
        encoded_sequence = one_hot_encode(sequence)

        # Reshape for the model (batch_size=1, seq_len, input_size=5)
        input_tensor = encoded_sequence.unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            prediction = model(input_tensor).item()

        # Display the result
        st.success(f"Predicted number of CpG sites: {round(prediction)}")

    except ValueError as e:
        st.error(str(e))
else:
    st.info("Please enter a DNA sequence to get started.")
