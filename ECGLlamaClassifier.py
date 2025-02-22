import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModel, AutoTokenizer, AdamW

class ECGLlamaClassification(nn.Module):
    def __init__(self, model_name="meta-llama/Llama-3.2-3B", num_labels=4):
        super(ECGLlamaClassification, self).__init__()

        self.num_labels = num_labels
        self.model_name = model_name
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")

        # Load the base model with automatic device mapping.
        self.base_model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"  # Automatically manages device placement
        )
        # Set device attribute based on the base model's parameters.
        self.device = next(self.base_model.parameters()).device

        # Mapping for classification labels.
        self.real_class_mapping = {
            "A": 0,  # Normal
            "V": 1,  # Ventricular issue
            "x": 2,  # Unknown class
            "J": 3   # Junctional rhythm
        }
        self.reverse_real_class_mapping = {v: k for k, v in self.real_class_mapping.items()}

        # Add a classification head without manually moving it.
        self.classifier = nn.Linear(self.base_model.config.hidden_size ,self.num_labels)
        self.classifier = self.classifier.to(self.device).half()

        # Load Tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Fix padding issue

    def create_tensors_dataloader(self ,input_ids ,attention_mask ,labels):
        from torch.utils.data import DataLoader ,TensorDataset
        dataset = TensorDataset(input_ids ,attention_mask ,labels)
        dataloader = DataLoader(dataset ,batch_size = 16 ,shuffle = True)  # Adjust batch_size as needed
        return dataset ,dataloader

    def pre_process_data_llama(self ,raw_data):
        """
		Preprocesses the raw ECG data into a tokenized format, including input_ids and attention_mask.
		Expects raw_data to be a dictionary with keys corresponding to labels and values as lists of data pairs.
		"""
        processed_data = {'input_ids': [] ,'attention_mask': []}
        labels = []

        for label ,pairs in raw_data.items():
            print(label ,pairs)
            for pair in pairs:
                # Convert the pair to a string and tokenize it.
                pair_string = " ".join(map(str ,pair))

                encoding = self.tokenizer.encode_plus(
                    pair_string ,
                    return_tensors = 'pt' ,
                    padding = 'max_length' ,  # Ensure all sequences have the same length.
                    max_length = 10 ,  # Adjust the max length as needed.
                    truncation = True ,  # Truncate sequences that are too long.
                    return_attention_mask = True
                )

                print(encoding)
                # Append the input_ids and attention_mask for this sample.
                processed_data['input_ids'].append(encoding['input_ids'])
                processed_data['attention_mask'].append(encoding['attention_mask'])

                # Assign the label using the mapping.
                labels.append(self.real_class_mapping[label])

        # Concatenate the list of tensors into single tensors.
        processed_data['input_ids'] = torch.cat(processed_data['input_ids'] ,dim = 0).to(self.device)
        processed_data['attention_mask'] = torch.cat(processed_data['attention_mask'] ,dim = 0).to(self.device)
        labels = torch.tensor(labels).to(self.device)
        print("Here are the labels" ,labels)
        return processed_data ,labels

    def forward(self, input_ids, attention_mask):
        # Pass inputs through the base model.
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        # Aggregate the hidden states via mean pooling.
        cls_output = torch.mean(outputs.last_hidden_state, dim=1)
        # Pass through the classification head.
        logits = self.classifier(cls_output)
        return logits

    def fine_tune_model_llama(self, dataloader, epochs=5, learning_rate=2e-5):
        # optimizer = AdamW(self.parameters(), lr=learning_rate)
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)

        loss_fn = nn.CrossEntropyLoss()
        self.train()

        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                # Move tensors in batch to the device (assumes batch is a tuple of tensors).
                input_ids, attention_mask, labels = [tensor.to(self.device) for tensor in batch]

                logits = self(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(logits, labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1}, Loss: {avg_loss}")

    def extract_embeddings(self, input_ids, attention_mask):
        """
        Extract embeddings from the base model.
        """
        with torch.no_grad():
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            embeddings = outputs.hidden_states[-1]
        return embeddings

    def evaluate_embeddings(self, embeddings1, embeddings2):
        """
        Compute cosine similarity between two embeddings.
        """
        similarity = cosine_similarity(embeddings1, embeddings2)
        return similarity

    def load_data(self ,data_path=None):
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

    def graph_ecg_signal(self, ecg_signal, test_or_train):
        """
        Plot ECG signal.
        """
        plt.figure(figsize=(10, 10))
        for label, points in ecg_signal[test_or_train].items():
            x_values = [x[0] for x in points]
            y_values = [y[1] for y in points]
            plt.title(f'Train data symbol {label}')
            plt.plot(x_values, y_values, label=f'Class {label}', marker='o')
        plt.legend()
        plt.show()

    def graph_loss(self, training_loss):
        """
        Plot loss graph.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(training_loss, label='Training Loss', color='blue')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()
