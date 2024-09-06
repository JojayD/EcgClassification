#Goal to develop neural symbolic methods for ECG classification.
import torch
import torch.nn as nn
import random
from transformers import BertForSequenceClassification ,BertTokenizer
class ECGClassifier:
	def __init__(self, model_name, data_path=None):
		"""
		:param model_path: Name of the path of the trained model
		:param num_labels: Number of labels
		:param data_path: Path of the data, defaults to None
		"""
		self.model_name = model_name
		self.data_path = data_path
		self.tokenizer = self.load_tokenizer()
		self.model = self.load_model()
	def load_model(self):
		"""
		Load pretrained model
		:returns
			in this instance we are using BertForSequenceClassification
		"""
		number_of_classes = 2
		model = BertForSequenceClassification.from_pretrained('bert-base-uncased' ,num_labels = number_of_classes)
		return model

	def load_tokenizer(self):
		"""
		Load the tokenizer for the model.
		"""
		from transformers import BertTokenizer

		return BertTokenizer.from_pretrained(self.model_name)

	def generate_mock_data(num_samples ,num_classes=2 ,range_of_values=(0 ,200)):
		"""
		Generate mock data following the given pattern.

		Parameters:
			 num_samples (int): Number of samples per class.
			 num_classes (int): Number of distinct classes.
			 range_of_values (tuple): Min and max values for the generated coordinates.

		Returns:
			 dict: Generated data structured as class to list of coordinate pairs.
		"""
		data = {}
		class_labels = ['a' ,'b' ,'c' ,'d' ,'e' ,'f'][:num_classes]  # Extend this list if more than 6 classes are needed.

		for label in class_labels:
			# Each class will have a list of coordinate pairs
			data[label] = []
			for _ in range(num_samples):
				pair1 = [random.randint(*range_of_values) ,random.randint(*range_of_values)]
				pair2 = [random.randint(*range_of_values) ,random.randint(*range_of_values)]
				data[label].append(pair1)
				data[label].append(pair2)
		return data

	def load_data(self, data=None):
		if data is None:
			return self.generate_mock_data()
		else:
			return data


	def pre_process_data(self, raw_data):
		"""

		:param raw_data:
			raw_data in a form of key with an array of pair values
		:return:
			returns string data structured as class to list of coordinate pairs. For BERT to understand this data structure

		"""
		for label, points in raw_data.items():
			for pair in points:
				print(pair)
				tokenized_pair = self.tokenizer.encode(" ".join(map(str, pair)), return_tensors='pt')
				print(tokenized_pair)


