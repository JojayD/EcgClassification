from ECGClassifier import ECGClassification
from data.ecgdata import mock_data
import torch
from torch.utils.data import DataLoader ,TensorDataset


def main():
	# Initialize the ECG classifier model
	model = ECGClassification("bert-base-uncased" ,2)

	# Load and preprocess the data
	data = model.load_data(mock_data)
	pre_processed_data ,labels = model.pre_process_data(data)

	# Convert the processed data and labels to tensors for DataLoader
	input_ids = pre_processed_data['input_ids']
	attention_mask = pre_processed_data['attention_mask']
	labels = torch.tensor(labels)

	# Create TensorDataset and DataLoader for batching
	dataset = TensorDataset(input_ids ,attention_mask ,labels)
	dataloader = DataLoader(dataset ,batch_size = 16 ,shuffle = True)  # Adjust batch_size as needed


	model.fine_tune_model(dataloader)
	model.evaluate_model(dataloader)

if __name__ == '__main__':
	main()
