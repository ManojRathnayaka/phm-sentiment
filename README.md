# Sentiment Analysis with LSTM and Bi-LSTM

This project implements and compares Long Short-Term Memory (LSTM) and Bidirectional LSTM (Bi-LSTM) models for sentiment analysis on tweet data. The goal is to classify tweets based on their sentiment using deep learning techniques.

## Project Structure

- **`LSTM.ipynb`**: Jupyter Notebook containing the implementation of the standard LSTM model using TensorFlow/Keras.
- **`Bi-LSTM.ipynb`**: Jupyter Notebook containing the implementation of the Bidirectional LSTM model.
- **`phm_train.csv`**: The training dataset containing labeled tweets.
- **`phm_test.csv`**: The testing dataset for evaluating model performance.

## Dataset

The dataset consists of tweets labeled for sentiment analysis.
- **Columns:**
  - `tweet_id`: Unique identifier for the tweet.
  - `label`: Sentiment label (e.g., 0 or 1).
  - `tweet`: The text content of the tweet.

## Prerequisites

To run this project, you need Python installed along with the following libraries:

- TensorFlow
- Pandas
- NumPy
- Matplotlib
- Seaborn
- NLTK
- Scikit-learn

You can install the necessary packages using pip:

```bash
pip install tensorflow pandas numpy matplotlib seaborn nltk scikit-learn
```

## Usage

1.  Clone the repository or download the project files.
2.  Ensure the dataset files (`phm_train.csv` and `phm_test.csv`) are in the same directory as the notebooks.
3.  Open the Jupyter Notebooks (`LSTM.ipynb` or `Bi-LSTM.ipynb`) using Jupyter Lab or Jupyter Notebook:

    ```bash
    jupyter notebook
    ```

4.  Run the cells in the notebooks to preprocess the data, train the models, and view the evaluation results.

## Models

- **LSTM (Long Short-Term Memory):** A type of Recurrent Neural Network (RNN) capable of learning order dependence in sequence prediction problems.
- **Bi-LSTM (Bidirectional LSTM):** An extension of LSTM that trains two instead of one LSTM on the input sequence (one on the input sequence as-is and one on a reversed copy of the input sequence), allowing the network to have both backward and forward information about the sequence at every time step.
