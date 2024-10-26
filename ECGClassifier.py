# #Goal to develop neural symbolic methods for ECG classification.
import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from visualize import *




class ECGClassification(nn.Module):
	def __init__(self ,model_name='bert-base-uncased', num_labels=5):
		super(ECGClassification, self).__init__()
		"""
		Initialize the class with the necessary parameters.

		Parameters:
		- model_name: Name of the pre-trained model
		- num_labels: Number of classes (labels) in the ECG classification task.
		- data_path: Path to the ECG dataset (optional for now).
		"""
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		self.class_mapping = {
			"a": 0 ,  # Normal
			"b": 1 ,  # Myocardial Infarction
			"c": 2 ,  # ST/T Change
			"d": 3 ,  # Conduction Disturbance
			"e": 4  # Hypertrophy
		}

		self.num_labels = num_labels
		self.model_name = model_name
		self.model = self.load_model()
		self.tokenizer = self.load_tokenizer()

		# Define the classification layer (output size = num_classes)
		# The input size of this layer should match the hidden size of the BERT model
		hidden_size = self.model.config.hidden_size  # This is usually 768 for 'bert-base-uncased'
		print(hidden_size)
		self.classifier = nn.Linear(hidden_size ,self.num_labels)
		self.softmax = nn.Softmax(dim = 1)

	def forward(self, inputs_ids, attention_mask):
		#Pass the input through bert model
		outputs = self.model(input_ids= inputs_ids, attention_mask= attention_mask)

		# Extract the hidden state of the [CLS] token (first token)
		cls_token_rep = outputs.last_hidden_state[:,0,:]

		# Pass the [CLS] representation through the classification layer
		logits = self.classifier(cls_token_rep)  # Shape: (batch_size, num_classes)

		return logits

	def load_model(self):
		"""
		Load a pre-trained BERT model for sequence classification.
		"""
		from transformers import BertForSequenceClassification

		# Load the BERT model for classification, fine-tuned for ECG classification
		return BertForSequenceClassification.from_pretrained(self.model_name, num_labels = self.num_labels)

	def load_tokenizer(self):
		"""
		Load the tokenizer for the model.
		"""
		from transformers import BertTokenizer

		return BertTokenizer.from_pretrained(self.model_name)

	def load_data(self, data_path=None):
		"""
		Load and preprocess the ECG dataset. Here we assume a simplified version for now.
		"""
		if data_path is None:
			return {
				"train": {"a": [[1 ,2] ,[2 ,3]] ,"b": [[100 ,1] ,[101 ,102]]} ,  # Mock data
				"test": {"a": [[4 ,5] ,[5 ,6]] ,"b": [[110 ,11] ,[111 ,112]]}
			}
		else:
			return data_path

	def generate_mockdata(self, num_labels):
		"""
		Generate mock data of the ECG dataset for classification does it based of the map labelings
		:return:
		mock data of the outputed size
		"""
		self.class_mapping


	def pre_process_data(self ,raw_data):
		"""
		Preprocesses the raw ECG data into BERT-compatible format, including input_ids and attention_mask.
		"""
		processed_data = {'input_ids': [] ,'attention_mask': []}
		labels = []

		for label ,pairs in raw_data.items():
			print(label, pairs)
			for pair in pairs:
				# Convert pair to a string and tokenize it with attention mask
				pair_string = " ".join(map(str ,pair))

				encoding = self.tokenizer.encode_plus(
					pair_string ,
					return_tensors = 'pt' ,
					padding = 'max_length' ,  # Ensure all sequences have the same length
					max_length = 10 ,  # Define a max length, adjust based on your data
					truncation = True ,  # Truncate sequences that are too long
					return_attention_mask = True  # Generate attention mask
				)

				print(encoding)
				# Append the input_ids and attention_mask for this sample
				processed_data['input_ids'].append(encoding['input_ids'])
				processed_data['attention_mask'].append(encoding['attention_mask'])

				# Simplified label assignment
				labels.append(self.class_mapping[label])

		# Convert the list of tensors into a single tensor for input_ids and attention_mask
		processed_data['input_ids'] = torch.cat(processed_data['input_ids'] ,dim = 0)
		processed_data['attention_mask'] = torch.cat(processed_data['attention_mask'] ,dim = 0)
		labels = torch.tensor(labels)
		print("Here are the labels", labels)
		return processed_data ,labels

	def save_model(self):
		pass

	def create_tensors_dataloader(self , input_ids , attention_mask, labels):
		from torch.utils.data import DataLoader ,TensorDataset
		dataset = TensorDataset(input_ids ,attention_mask ,labels)
		dataloader = DataLoader(dataset ,batch_size = 16 ,shuffle = True)  # Adjust batch_size as needed
		return dataset,dataloader

	def fine_tune_model(self ,dataloader):
		"""
		Fine-tune the pre-trained BERT model on the ECG data.

		Parameters:
		- dataloader: A DataLoader object that provides batches of tokenized input data and labels.
		"""
		from transformers import AdamW
		import torch

		# Use AdamW optimizer for fine-tuning
		optimizer = AdamW(self.model.parameters() ,lr = 5e-5)
		loss_arr = []
		for epoch in range(10):  # Number of epochs
			print(f"Epoch {epoch + 1}")
			for idx ,batch in enumerate(dataloader):
				optimizer.zero_grad()

				input_ids = batch[0]
				attention_mask = batch[1]
				labels = batch[2]
				outputs = self.model(input_ids = input_ids ,attention_mask = attention_mask ,labels = labels)
				loss = outputs.loss
				loss_arr.append(loss.item())
				print(f"{outputs.loss.item()}")
				loss.backward()
				optimizer.step()

				print(f"Batch {idx + 1}, Loss: {loss.item()}")

			print(f"Epoch {epoch + 1} completed")

		print("Training complete!")
		self.graph_loss(loss_arr)

	def symbolic_reasoning(self ,output):
		"""
		Apply symbolic reasoning based on the model's output.

		Parameters:
		- output: The output of the neural network (logits, predictions).
		"""
		# This could be a set of rules or heuristics that interprets the model's predictions
		match output:
			case 0:
				return "a"
			case 1:
				return "b"
			case 2:
				return "c"
			case 3:
				return "d"
			case 4:
				return "e"



	def evaluate_model(self ,test_data):
		"""
		Test the model and evaluate its performance on unseen ECG data,
		applying symbolic reasoning after neural predictions.

		Parameters:
		- test_data: The preprocessed test data.
		"""
		correct = 0
		total = 0
		predictions = []
		true_labels = []
		signals = []
		for batch in test_data:
			with torch.no_grad():
				input_ids = batch[0]
				attention_mask = batch[1]
				labels = batch[2]

				outputs = self.model(input_ids = input_ids ,attention_mask = attention_mask)

				# Convert logits to probabilities (optional)
				probabilities = torch.nn.functional.softmax(outputs.logits ,dim = -1)
				predicted_class = torch.argmax(probabilities,dim = -1)
				print(f"This is the predicted class: {predicted_class}")

				correct += (predicted_class == labels).sum().item()
				total += labels.size(0)

				accuracy = correct / total
				predictions.extend(predicted_class.cpu().numpy())
				true_labels.extend(labels.cpu().numpy())
				signals.extend(batch[0].cpu().numpy())  # Assuming batch[0] contains ECG signal data

		symbolic= []
		for row in predicted_class.numpy():
			symbol = self.symbolic_reasoning(row)
			symbolic.append(symbol)

		print(f"Model Accuracy with Symbolic Reasoning: {accuracy:.2f}")
		print(f"Symbols of Symbolic Reasoning: {symbolic}")

		accuracy = accuracy_score(true_labels, predictions)
		precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='micro')
		cm = confusion_matrix(true_labels, predictions)
		print(f"Accuracy: {accuracy * 100:.2f}%")
		print(f"Precision: {precision * 100:.2f}%")
		print(f"Recall: {recall * 100:.2f}%")
		print(f"F1-Score: {f1 * 100:.2f}%")

		# Plot Confusion Matrix
		plt.figure(figsize=(6, 6))
		sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
					xticklabels=['Normal', 'Abnormal'],
					yticklabels=['Normal', 'Abnormal'])
		plt.ylabel('Actual')
		plt.xlabel('Predicted')
		plt.title('Confusion Matrix')
		plt.show()

		# Apply symbolic reasoning
		symbolic = [self.symbolic_reasoning(pred) for pred in predictions]

		return predictions, signals

	def extract_embeddings(self, input_ids , attention_mask):
		with torch.no_grad():
			outputs = self.model(input_ids = input_ids ,attention_mask = attention_mask, output_hidden_states=True)
			embeddings = outputs.hidden_states[-1]
		return embeddings

	def evaluate_embeddings(self ,embeddings1, embeddings2):
		similarity = cosine_similarity(embeddings1 ,embeddings2)
		return similarity

	"""
	
	Graph Evaluations
	
	"""




	def graph_ecg_signal(self, ecg_signal, test_or_train):
		plt.figure(figsize = (10,10))
		for label, points in ecg_signal[test_or_train].items():
			x_values = [x[0] for x in points]
			y_values = [y[1] for y in points]
			plt.title(f'Train data symbol {label}')
			plt.xlabel("")
			plt.plot(x_values ,y_values ,label = f'Class {label}' ,marker = 'o')  # Using marker='o' to highlight points
		# plt.legend()
		plt.show()


	def graph_loss(self, training_loss):
		plt.figure(figsize = (10,10))
		plt.figure(figsize = (10 ,6))
		plt.plot(training_loss ,label = 'Training Loss' ,color = 'blue')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.title('Training Loss Over Epochs')
		plt.legend()
		plt.grid(True)
		plt.show()




