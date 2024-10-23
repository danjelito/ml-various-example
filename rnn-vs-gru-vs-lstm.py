import csv

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from neattext.functions import clean_text
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

torch.random.manual_seed(0)


def split_dataset(x, y, test_size=0.2, val_size=0.1, random_state=None):
    """Split the dataset into train, validation, and test sets."""
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    # Adjust validation size to the remaining training set
    val_fraction_of_train = val_size / (1 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=val_fraction_of_train,
        random_state=random_state,
    )

    return x_train, x_val, x_test, y_train, y_val, y_test


def clean(text):
    """Clean dataset."""
    return clean_text(
        text,
        stopwords=True,
        puncts=True,
        emojis=True,
        special_char=True,
        phone_num=True,
        non_ascii=True,
        multiple_whitespaces=True,
        contractions=True,
    )


def preprocess(text, tokenizer, lemmatizer):
    text = tokenizer.tokenize(text)
    text = [lemmatizer.lemmatize(word) for word in text]
    return text


# Embedding
def load_glove_embeedings(path):
    """Load GloVe embeddings."""
    glove_embeddings = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]  # The word
            vector = np.asarray(values[1:], dtype="float32")  # The embedding vector
            glove_embeddings[word] = np.array(vector)
        return glove_embeddings


def get_word_embedding(word, glove_embeddings, embedding_shape):
    """Get word embedding of a word. If not exist, return zero vector."""
    return glove_embeddings.get(word, np.zeros(embedding_shape))


def get_sentence_embedding(sentence, glove_embeddings, embedding_shape):
    return np.array(
        [
            get_word_embedding(word, glove_embeddings, embedding_shape)
            for word in sentence
        ]
    )


def get_document_embedding(document, glove_embeddings, embedding_shape):
    return [
        get_sentence_embedding(sentence, glove_embeddings, embedding_shape)
        for sentence in document
    ]


def pad_or_truncate(dataset, max_sentence_length):
    dataset = dataset.copy()
    for i, _ in enumerate(dataset):
        current_sentence_length = dataset[i].shape[0]
        if current_sentence_length > max_sentence_length:  # Truncate
            dataset[i] = dataset[i][:max_sentence_length]
        elif current_sentence_length < max_sentence_length:  # Pad
            to_pad = max_sentence_length - current_sentence_length
            padding = np.zeros((to_pad, 50), dtype="float32")
            dataset[i] = np.concatenate((dataset[i], padding))
    return np.array(dataset)


def normalize_array(
    arr,
    target_min_value,
    target_max_value,
    current_min_value=None,
    current_max_value=None,
):
    """
    Normalize array to the given range. Optionally, provide current_min_value and current_max_value.
    If not supplied, they will be calculated from the array.
    """
    if current_min_value is None:
        current_min_value = arr.min()
    if current_max_value is None:
        current_max_value = arr.max()

    # Calculate ranges
    current_range = current_max_value - current_min_value
    target_range = target_max_value - target_min_value

    # Normalize the array to the target range
    scaled_arr = (
        (arr - current_min_value) / current_range * target_range
    ) + target_min_value
    return scaled_arr


class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = np.array(self.texts[index], dtype="float32")
        label = np.array(self.labels[index], dtype="int64")
        # label = np.expand_dims(label, axis=0)
        return text, label


class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNNClassifier, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.linear = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x):
        output, _ = self.rnn(x)
        output = output[:, -1, :]  # Last step output
        output = self.linear(output)
        return output


class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(GRUClassifier, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.linear = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x):
        output, _ = self.gru(x)
        output = output[:, -1, :]  # Last step output
        output = self.linear(output)
        return output


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.linear = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = output[:, -1, :]  # Last step output
        output = self.linear(output)
        return output


def calculate_class_probabilities(logits):
    probabilities = F.softmax(logits, dim=1)
    return probabilities


def get_predicted_class(probabilities):
    predicted_classes = torch.argmax(probabilities, dim=1)
    return predicted_classes


def train_one_epoch(model, optimizer, criterion, dataloader):
    model.train()
    running_loss = 0.0
    for batch_idx, (xs, ys) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(xs)
        loss = criterion(output, ys)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / (batch_idx + 1)


def val_one_epoch(model, criterion, dataloader):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch_idx, (xs, ys) in enumerate(dataloader):
            output = model(xs)
            loss = criterion(output, ys)
            running_loss += loss.item()
    return running_loss / (batch_idx + 1)


def print_loss(epoch, epochs, train_loss, val_loss, print_step):
    if (epoch + 1) % print_step == 0:
        print(
            f"  Epoch [{epoch + 1}/{epochs}], "
            f"  Train Loss: {train_loss:.4f}, "
            f"  Validation Loss: {val_loss:.4f}"
        )


def main(debug=False):
    # Read dataset
    print("Opening dataset...")
    x = []
    y = []
    with open(
        "dataset/tickets/all_tickets_processed_improved_v3.csv",
        mode="r",
        encoding="utf-8-sig",
    ) as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for i, row in enumerate(reader):
            if i > 0:
                raw_text = row[0]
                label = row[1]
                x.append(raw_text)
                y.append(label)

    if debug:
        # ! LIMIT DATASET TO 2000 samples
        x = x[:2000]
        y = y[:2000]

    # Encode y as float
    print("Encoding labels...")
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    y = np.array(y).astype("float")

    # Create train test val split
    print("Creating train val test splits...")
    x_train, x_val, x_test, y_train, y_val, y_test = split_dataset(x, y)
    assert (len(x_train) + len(x_val) + len(x_test)) == len(x)

    # Clean dataset
    print("Cleaning dataset...")
    if debug:
        for i in range(0, 3):
            print(f"    Before cleaning: {x_train[i]}")
            print(f"    After cleaning: {clean(x_train[i])}")
    x_train = [clean(x) for x in x_train]
    x_val = [clean(x) for x in x_val]
    x_test = [clean(x) for x in x_test]

    # Preprocessing
    print("Preprocessing dataset...")
    lemmatizer = WordNetLemmatizer()
    tokenizer = TreebankWordTokenizer()
    if debug:
        for i in range(0, 3):
            print(f"    Before preprocesing: {x_train[i]}")
            print(
                f"    After preprocesing: {preprocess(x_train[i], tokenizer, lemmatizer)}"
            )
    x_train = [preprocess(x, tokenizer, lemmatizer) for x in x_train]
    x_val = [preprocess(x, tokenizer, lemmatizer) for x in x_val]
    x_test = [preprocess(x, tokenizer, lemmatizer) for x in x_test]

    # Embedding
    print("Creating text embedding...")
    path = "/home/anj/Documents/practice/book practical nlp/data/glove/glove.6B/glove.6B.50d.txt"
    glove_embeddings = load_glove_embeedings(path)
    embedding_shape = (50,)
    x_train = get_document_embedding(x_train, glove_embeddings, embedding_shape)
    x_val = get_document_embedding(x_val, glove_embeddings, embedding_shape)
    x_test = get_document_embedding(x_test, glove_embeddings, embedding_shape)
    assert all([x.shape[1] == 50 for x in x_train])
    assert all([x.shape[1] == 50 for x in x_val])
    assert all([x.shape[1] == 50 for x in x_test])
    assert all([x.shape[0] > 0 for x in x_train])
    assert all([x.shape[0] > 0 for x in x_val])
    assert all([x.shape[0] > 0 for x in x_test])

    # Padding and truncating the text to same lengths
    print("Padding and truncating...")
    train_sentence_lengths = [x.shape[0] for x in x_train]
    # Set the maximum sentence length to 75th quartile
    max_sentence_length = int(
        pd.DataFrame(train_sentence_lengths).describe().loc["75%"].values[0]
    )
    x_train = pad_or_truncate(x_train, max_sentence_length)
    x_val = pad_or_truncate(x_val, max_sentence_length)
    x_test = pad_or_truncate(x_test, max_sentence_length)
    assert all([x.shape[0] == max_sentence_length for x in x_train])
    assert all([x.shape[0] == max_sentence_length for x in x_val])
    assert all([x.shape[0] == max_sentence_length for x in x_test])

    # MinMax scaling
    for dim in range(x_train.shape[2]):
        # Always use train min and max value to prevent data leaking
        train_min_value = x_train[:, :, dim].min()
        train_max_value = x_train[:, :, dim].max()
        # Train
        x_train[:, :, dim] = normalize_array(
            x_train[:, :, dim],
            target_min_value=0,
            target_max_value=1,
            current_min_value=train_min_value,
            current_max_value=train_max_value,
        )
        # Val
        x_val[:, :, dim] = normalize_array(
            x_val[:, :, dim],
            target_min_value=0,
            target_max_value=1,
            current_min_value=train_min_value,
            current_max_value=train_max_value,
        )
        # Test
        x_test[:, :, dim] = normalize_array(
            x_test[:, :, dim],
            target_min_value=0,
            target_max_value=1,
            current_min_value=train_min_value,
            current_max_value=train_max_value,
        )
    # Check that the scaling works
    tolerance = 0.5  # Tolerance threshold for floating-point comparisons
    # Assertions for train set
    assert np.all(np.isclose(x_train.min(axis=(0, 1)), 0.0, atol=tolerance))
    assert np.all(np.isclose(x_train.max(axis=(0, 1)), 1.0, atol=tolerance))
    # Assertions for validation set
    assert np.all(np.isclose(x_val.min(axis=(0, 1)), 0.0, atol=tolerance))
    assert np.all(np.isclose(x_val.max(axis=(0, 1)), 1.0, atol=tolerance))
    # Assertions for test set
    assert np.all(np.isclose(x_test.min(axis=(0, 1)), 0.0, atol=tolerance))
    assert np.all(np.isclose(x_test.max(axis=(0, 1)), 1.0, atol=tolerance))

    # Creating dataset and dataloader
    print("Creating dataset and dataloader...")
    train_dataset = TextDataset(x_train, y_train)
    val_dataset = TextDataset(x_val, y_val)
    test_dataset = TextDataset(x_test, y_test)
    batch_size = 4
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    if debug:
        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
            print(f"    Batch {batch_idx + 1}")
            # (batch_size, sequence_length, embedding_dim)
            print(f"    Inputs shape: {inputs.shape}")
            # (batch_size, 1)
            print(f"    Targets shape: {targets.shape}")
            # print(f"    Inputs:", inputs)
            # print(f"    Targets:", targets)
            break

    # Define model
    # RNN
    embedding_dim = 50
    n_class = len(set(y_train))
    rnn_classifier = RNNClassifier(
        input_size=embedding_dim,
        hidden_size=64,
        output_size=n_class,
        num_layers=2,
    )
    gru_classifier = GRUClassifier(
        input_size=embedding_dim,
        hidden_size=64,
        output_size=n_class,
        num_layers=2,
    )
    lstm_classifier = LSTMClassifier(
        input_size=embedding_dim,
        hidden_size=128,
        output_size=n_class,
        num_layers=1,
    )
    # Make sure the model is working
    if debug:
        with torch.no_grad():
            print(rnn_classifier(inputs).shape)
            print(calculate_class_probabilities(rnn_classifier(inputs)))
            print(
                get_predicted_class(
                    calculate_class_probabilities(rnn_classifier(inputs))
                )
            )
        with torch.no_grad():
            print(gru_classifier(inputs).shape)
            print(calculate_class_probabilities(gru_classifier(inputs)))
            print(
                get_predicted_class(
                    calculate_class_probabilities(gru_classifier(inputs))
                )
            )
        with torch.no_grad():
            print(lstm_classifier(inputs).shape)
            print(calculate_class_probabilities(lstm_classifier(inputs)))
            print(
                get_predicted_class(
                    calculate_class_probabilities(lstm_classifier(inputs))
                )
            )

    # Training
    epochs = 20
    print_step = 1
    # Training RNN
    lr = 0.0001
    rnn_optimizer = torch.optim.Adam(rnn_classifier.parameters(), lr=lr)
    rnn_criterion = nn.CrossEntropyLoss()
    print("Training model: RNN")
    for epoch in range(epochs):
        rnn_train_loss = train_one_epoch(
            rnn_classifier, rnn_optimizer, rnn_criterion, train_dataloader
        )
        rnn_val_loss = val_one_epoch(rnn_classifier, rnn_criterion, val_dataloader)
        print_loss(epoch, epochs, rnn_train_loss, rnn_val_loss, print_step)
    # Training GRU
    lr = 0.001
    gru_optimizer = torch.optim.Adam(gru_classifier.parameters(), lr=lr)
    gru_criterion = nn.CrossEntropyLoss()
    print("Training model: GRU")
    for epoch in range(epochs):
        gru_train_loss = train_one_epoch(
            gru_classifier, gru_optimizer, gru_criterion, train_dataloader
        )
        gru_val_loss = val_one_epoch(gru_classifier, gru_criterion, val_dataloader)
        print_loss(epoch, epochs, gru_train_loss, gru_val_loss, print_step)
    # Training LSTM
    lr = 0.0001
    lstm_optimizer = torch.optim.Adam(lstm_classifier.parameters(), lr=lr)
    lstm_criterion = nn.CrossEntropyLoss()
    print("Training model: LSTM")
    for epoch in range(epochs):
        lstm_train_loss = train_one_epoch(
            lstm_classifier, lstm_optimizer, lstm_criterion, train_dataloader
        )
        lstm_val_loss = val_one_epoch(lstm_classifier, lstm_criterion, val_dataloader)
        print_loss(epoch, epochs, lstm_train_loss, lstm_val_loss, print_step)


if __name__ == "__main__":

    debug = False
    if debug:
        print("Debugging is on.")
    else:
        print("Debugging is off.")

    main(debug)
