from ECGClassifier import ECGClassification
import torch
from data.ecgdata import mock_data
from data.ecgdata import mock_data2
from data.ecgdata import mock_data3

def main():
    # Initialize the ECG classifier model
    model = ECGClassification(num_labels = 5)

    # Load and preprocess the training data
    data = model.load_data(mock_data3)  # Load the training data
    pre_processed_data, labels = model.pre_process_data(data['train'])

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
    model.graph_results(data['train'],data['test'])

    test_input_ids = pre_processed_test_data['input_ids']
    test_attention_mask = pre_processed_test_data['attention_mask']
    test_labels = torch.tensor(test_labels)
    test_dataset, test_dataloader = model.create_tensors_dataloader(test_input_ids, test_attention_mask, test_labels)


    # Evaluate the model on the test data
    model.evaluate_model(test_dataloader)

    embeddings1 = model.extract_embeddings(train_input_ids ,train_attention_mask)
    embeddings2 = model.extract_embeddings(test_input_ids ,train_attention_mask)


    similarity = model.evaluate_embeddings(embeddings1.mean(dim = 1) ,embeddings2.mean(dim = 1))  # mean over sequence length to get a single vector per input
    # model.graph_evaluation_embeddings(similarity)

if __name__ == '__main__':
    main()
