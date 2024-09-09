from ECGClassifier import ECGClassification
import torch


def main():
    # Initialize the ECG classifier model
    model = ECGClassification("bert-base-uncased", 2)

    # Load and preprocess the training data
    data = model.load_data()  # Load the training data
    pre_processed_data, labels = model.pre_process_data(data)

    # Convert the processed data and labels to tensors for DataLoader
    input_ids = pre_processed_data['input_ids']
    attention_mask = pre_processed_data['attention_mask']
    labels = torch.tensor(labels)

    # Create TensorDataset and DataLoader for batching
    dataset, dataloader = model.create_tensors_dataloader(input_ids, attention_mask, labels)

    # Fine-tune the model
    model.fine_tune_model(dataloader)

    # Now load and preprocess the test data
    test_data = model.load_data(data['test'])  # Load the test data
    pre_processed_test_data, test_labels = model.pre_process_data(test_data)

    test_input_ids = pre_processed_test_data['input_ids']
    test_attention_mask = pre_processed_test_data['attention_mask']
    test_labels = torch.tensor(test_labels)
    test_dataset, test_dataloader = model.create_tensors_dataloader(test_input_ids, test_attention_mask, test_labels)

    # Evaluate the model on the test data
    model.evaluate_model(test_dataloader)

if __name__ == '__main__':
    main()
