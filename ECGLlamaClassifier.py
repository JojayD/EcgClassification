import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModel, AutoTokenizer, AdamW

class ECGLlamaClassification(nn.Module):
    def __init__(self, model_name="codellama/CodeLlama-7b-Python-hf", num_labels=4):
        super(ECGLlamaClassification, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_labels = num_labels
        self.model_name = model_name

        # Load the base CodeLLaMA model
        self.base_model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"  # Adjust if using custom device mapping
        ).to(self.device)

        # Add a classification head
        self.classifier = nn.Linear(self.base_model.config.hidden_size, self.num_labels).to(self.device)

        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Fix padding issue

    def forward(self, input_ids, attention_mask):
        # Pass inputs through the base model
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)

        # Aggregate the hidden states (mean pooling as a robust alternative)
        cls_output = torch.mean(outputs.last_hidden_state, dim=1)

        # Pass through the classification head
        logits = self.classifier(cls_output)
        return logits

    def fine_tune_model_llama(self, dataloader, epochs=5, learning_rate=2e-5):
        optimizer = AdamW(self.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        self.train()

        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
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
        Extract embeddings from CodeLLaMA.
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
