from ECGClassifier import ECGClassification
from visualize import *
from ECGLlamaClassifier import ECGLlamaClassification

import torch
from data.ecgdata import mock_data
from data.ecgdata import mock_data2
from data.ecgdata import mock_data3
from data.ecgdata import real_data

def main():
    # Initialize the ECG classifier model
    model = ECGClassification(num_labels = 4)

    # Load and preprocess the training data
    data = model.load_data(real_data)  # Load the training data
    pre_processed_data, labels = model.pre_process_data(data)

    # Convert the processed data and labels to tensors for DataLoader
    train_input_ids = pre_processed_data['input_ids']
    train_attention_mask = pre_processed_data['attention_mask']
    labels = torch.tensor(labels)

    # Create TensorDataset and DataLoader for batching
    dataset, dataloader = model.create_tensors_dataloader(train_input_ids, train_attention_mask, labels)

    # Fine-tune the model
    model.fine_tune_model(dataloader)

    # Now load and preprocess the test data
    test_data = model.load_data(data['test'])  # Load the test data
    pre_processed_test_data, test_labels = model.pre_process_data(test_data)

    #Plotting train and test data
    model.graph_ecg_signal(mock_data3, 'train')


    test_input_ids = pre_processed_test_data['input_ids']
    test_attention_mask = pre_processed_test_data['attention_mask']
    test_labels = torch.tensor(test_labels)
    test_dataset, test_dataloader = model.create_tensors_dataloader(test_input_ids, test_attention_mask, test_labels)


    # Evaluate the model on the test data
    model.evaluate_model(test_dataloader)

    model.model.eval()

    predictions=[]
    signals=[]
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch[0].to(model.device)
            attention_mask = batch[1].to(model.device)
            labels = batch[2].to(model.device)

            outputs = model.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class_01 = torch.argmax(probabilities, dim=-1)

            # Move tensors to CPU and convert to numpy for processing
            predicted_class_01 = predicted_class_01.cpu().numpy()
            batch_signals = batch[0].cpu().numpy()  # Assuming batch[0] contains ECG signal data

            predictions.extend(predicted_class_01)
            signals.extend(batch_signals)



    # Visualize the first N ECG signals with their predicted symbols
    N = 4  # Number of samples to visualize
    for i in range(min(N, len(signals))):
        ecg_signal = signals[i]  # Shape: (sequence_length,)
        predicted_label = int(predictions[i])
        reversed_predicted_label = model.reverse_real_class_mapping[predicted_label]
        symbol = model.class_mapping.get(predicted_label, reversed_predicted_label)
        print(symbol)

        # Convert the signal into a list of tuples (x, y)
        # Here, x is the time step, y is the amplitude
        ecg_tuples = [(x, y) for x, y in enumerate(ecg_signal)]

        # Choose a timestamp to place the symbol annotation
        # For demonstration, place it at the midpoint of the signal
        timestamp = len(ecg_tuples) // 2

        # Call the visualization function
        visualize_ecg_with_symbols(ecg_tuples, symbol, timestamp, model.class_mapping)

    test_embeddings = model.extract_embeddings(test_input_ids, test_attention_mask).mean(dim=1).numpy()
    test_true_labels = test_labels.cpu().numpy()

    visualize_embeddings(test_embeddings, test_true_labels, model.class_mapping)

def main_data():
    # Initialize the ECG classifier model
    model = ECGClassification(num_labels=4)
    data = model.load_data(real_data)
    # Define class mappings for real_data
    pre_processed_data, labels = model.pre_process_data(data)

    # Load and preprocess the training data

    # Convert processed data to tensors for DataLoader
    train_input_ids = pre_processed_data['input_ids']
    train_attention_mask = pre_processed_data['attention_mask']
    labels = torch.tensor(labels)

    # Create TensorDataset and DataLoader for batching
    dataset, dataloader = model.create_tensors_dataloader(train_input_ids, train_attention_mask, labels)

    # Fine-tune the model using real data
    model.fine_tune_model(dataloader)

    # Use the same data for testing since we don't have a separate test split
    pre_processed_test_data, test_labels = model.pre_process_data(real_data)


    test_input_ids = pre_processed_test_data['input_ids']
    test_attention_mask = pre_processed_test_data['attention_mask']
    test_labels = torch.tensor(test_labels)

    # Create DataLoader for evaluation
    test_dataset, test_dataloader = model.create_tensors_dataloader(test_input_ids, test_attention_mask, test_labels)

    # Evaluate the model
    model.evaluate_model(test_dataloader)

    # Set model to evaluation mode
    model.model.eval()

    predictions = []
    signals = []

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch[0].to(model.device)
            attention_mask = batch[1].to(model.device)
            labels = batch[2].to(model.device)

            # Get model predictions
            outputs = model.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1)

            # Move tensors to CPU for processing
            predicted_class = predicted_class.cpu().numpy()
            batch_signals = batch[0].cpu().numpy()

            predictions.extend(predicted_class)
            signals.extend(batch_signals)

    # Visualize the first N ECG signals with their predicted symbols
    N = 4  # Number of samples to visualize
    for i in range(min(N, len(signals))):
        ecg_signal = signals[i]  # Shape: (sequence_length,)
        predicted_label = int(predictions[i])
        predicted_symbol = model.reverse_real_class_mapping.get(predicted_label, "Unknown")  # Get the class label

        print(f"Predicted Symbol: {predicted_symbol}")

        # Convert the signal into a list of tuples (x, y)
        ecg_tuples = [(x, y) for x, y in enumerate(ecg_signal)]

        # Choose a timestamp to place the symbol annotation
        timestamp = len(ecg_tuples) // 2

        # Call the visualization function
        visualize_ecg_with_symbols(ecg_tuples, predicted_symbol, timestamp, model.real_class_mapping)

    # Extract embeddings and visualize them
    test_embeddings = model.extract_embeddings(test_input_ids, test_attention_mask).mean(dim=1).numpy()
    test_true_labels = test_labels.cpu().numpy()

    visualize_embeddings(test_embeddings, test_true_labels, model.real_class_mapping)


def main_llama():
    # Initialize the LLaMA-based ECG classifier model
    model = ECGLlamaClassification(num_labels = 4)

    # Load and preprocess the training data
    data = model.load_data(real_data)  # Load real ECG data
    pre_processed_data ,labels = model.pre_process_data_llama(data)

    train_input_ids = pre_processed_data['input_ids']
    train_attention_mask = pre_processed_data['attention_mask']
    labels = torch.tensor(labels)

    dataset ,dataloader = model.create_tensors_dataloader(train_input_ids ,train_attention_mask ,labels)

    # Fine-tune the LLaMA model using real ECG data
    model.fine_tune_model_llama(dataloader)

    pre_processed_test_data ,test_labels = model.pre_process_data_llama(real_data)

    test_input_ids = pre_processed_test_data['input_ids']
    test_attention_mask = pre_processed_test_data['attention_mask']
    test_labels = torch.tensor(test_labels)

    # Create DataLoader for evaluation
    test_dataset ,test_dataloader = model.create_tensors_dataloader(test_input_ids ,test_attention_mask ,test_labels)

    # Evaluate the model
    model.evaluate_model(test_dataloader)

    # Set model to evaluation mode
    model.model.eval()

    predictions = []
    signals = []

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch[0].to(model.device)
            attention_mask = batch[1].to(model.device)
            labels = batch[2].to(model.device)

            # Get model predictions
            outputs = model.model(input_ids = input_ids ,attention_mask = attention_mask)
            probabilities = torch.nn.functional.softmax(outputs.logits ,dim = -1)
            predicted_class = torch.argmax(probabilities ,dim = -1)

            # Move tensors to CPU for processing
            predicted_class = predicted_class.cpu().numpy()
            batch_signals = batch[0].cpu().numpy()

            predictions.extend(predicted_class)
            signals.extend(batch_signals)

    # Visualize the first N ECG signals with their predicted symbols
    N = 4  # Number of samples to visualize
    for i in range(min(N ,len(signals))):
        ecg_signal = signals[i]  # Shape: (sequence_length,)
        predicted_label = int(predictions[i])
        predicted_symbol = model.reverse_real_class_mapping.get(predicted_label ,"Unknown")  # Get the class label

        print(f"Predicted Symbol: {predicted_symbol}")

        # Convert the signal into a list of tuples (x, y)
        ecg_tuples = [(x ,y) for x ,y in enumerate(ecg_signal)]

        # Choose a timestamp to place the symbol annotation
        timestamp = len(ecg_tuples) // 2

        # Call the visualization function
        visualize_ecg_with_symbols(ecg_tuples ,predicted_symbol ,timestamp ,model.real_class_mapping)

    # Extract embeddings and visualize them
    test_embeddings = model.extract_embeddings(test_input_ids ,test_attention_mask).mean(dim = 1).cpu().numpy()
    test_true_labels = test_labels.cpu().numpy()

    visualize_embeddings(test_embeddings ,test_true_labels ,model.real_class_mapping)


if __name__ == '__main__':
    main_data()
