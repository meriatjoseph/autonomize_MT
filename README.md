# CpG Detector with LSTM

## Overview
This repository contains an implementation of an LSTM-based model for detecting CpG sites in DNA sequences. The task is to predict the number of CpG sites (consecutive 'CG' nucleotides) in a given DNA sequence. The project includes the model training, evaluation, and a visualization of the results.

## Features
- **LSTM Model**: An LSTM neural network is used to learn and predict CpG site counts.
- **Custom Dataset**: DNA sequences are encoded and used to train the model.
- **Training and Evaluation**: Includes a complete pipeline for model training and performance evaluation.
- **Visualization**: Scatter plot comparing actual vs. predicted CpG site counts.
- **Streamlit Web App (Planned)**: An interactive web interface for testing the model.

## Installation
### Requirements
- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- Streamlit (for the web app)

### Setup
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/cpg-detector.git
    cd cpg-detector
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Verify PyTorch installation:
    ```bash
    python -c "import torch; print(torch.__version__)"
    ```

## Usage
### Training the Model
1. Open the Jupyter Notebook provided and run the cells to:
   - Prepare the dataset
   - Train the LSTM model
   - Evaluate the model

2. Alternatively, run the Python script (if provided).

### Streamlit Web App
1. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
2. Input a DNA sequence (e.g., `NCACANNTNCGGAGGCGNA`) to get the predicted CpG count.

## File Structure
- `cpg_detector_lstm.py`: Core implementation of the LSTM model and training pipeline.
- `app.py`: Streamlit web application for testing the model.
- `requirements.txt`: Required Python packages.
- `README.md`: Project documentation.

## Example Results
After training the model, the evaluation results can be visualized using a scatter plot comparing actual vs. predicted CpG site counts.

![Model Evaluation](results/evaluation_plot.png)

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgments
Special thanks to the contributors and the open-source community for providing valuable tools and libraries.

