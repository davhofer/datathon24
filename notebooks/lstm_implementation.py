import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# Global device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Forward pass through LSTM layer
        # x shape: (batch, sequence_length, input_size)
        out, (h_t, c_t) = self.lstm(x)

        # We use the last hidden state to predict the output
        # h_t shape: (num_layers, batch, hidden_size)
        # Take the last layer's hidden state
        h_t_last_layer = h_t[-1]

        # Forward pass through the fully connected layer
        out = self.fc(h_t_last_layer)

        return out

def initialize_data_loader(X, y, batch_size):
    # Convert list of numpy arrays to a list of torch tensors
    X_tensors = [torch.tensor(sequence, dtype=torch.float32) for sequence in X]
    
    # Since there is no need to pad, we can assume all sequences are of equal length or
    # handling variable length isn't required. We'll directly stack them into a single tensor.
    # If sequences are of different lengths and you decide later to handle it, adjustments will be needed here.
    X_tensor = torch.stack(X_tensors)
    
    # Convert list of single values into a torch tensor
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # Reshape y to be a column vector
    
    # Create tensor dataset
    dataset = TensorDataset(X_tensor, y_tensor)
    
    # Create data loader
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_model(model, data_loader, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(data_loader):.4f}')

def test_model(model, data_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    with torch.no_grad():  # No gradients needed
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()

    print(f'Test Loss: {total_loss / len(data_loader):.4f}')


# Hyperparameters
input_size = 1
hidden_size = 50
num_layers = 1
output_size = 1
learning_rate = 0.01
num_epochs = 100
batch_size = 64

# Example usage
X = [np.random.randn(10, 1) for _ in range(100)]  # list of numpy arrays
y = [np.random.randn() for _ in range(100)]  # corresponding single values

# Model setup
model = LSTMModel(input_size, hidden_size, num_layers, output_size)
model = model.to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Initialize DataLoader
data_loader = initialize_data_loader(X , y , batch_size)

# Train the model
train_model(model, data_loader, criterion, optimizer, num_epochs, device)



# Generate test data
X_test = [np.random.randn(10, 1) for _ in range(30)]  # list of numpy arrays for test
y_test = [np.random.randn() for _ in range(30)]  # corresponding single values for test

# Initialize DataLoader for test data
test_data_loader = initialize_data_loader(X_test, y_test, batch_size)


test_model(model, data_loader, criterion, device)
