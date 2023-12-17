import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable

class LSTM_MLP(nn.Module):
    def __init__(self, obstacle_features, hidden_size, mlp_output_size):
        super(LSTM_MLP, self).__init__()
        #LTSM for positional features (current position and goal)
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden_size, batch_first=True)

        # MLP for obstacle features with BatchNorm
        self.obstacle_mlp = nn.Sequential(
            nn.Linear(obstacle_features, 128), nn.PReLU(), nn.Dropout(),
            nn.Linear(128, 64), nn.PReLU(), nn.Dropout(),
            nn.Linear(64, hidden_size), nn.PReLU(), nn.Dropout()
        )

        # Final MLP layers after combining LTSM and MLP outputs with BatchNorm
        self.final_mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, 1280), nn.PReLU(), nn.Dropout(),
            nn.Linear(1280, 1024),nn.PReLU(),nn.Dropout(),
		    nn.Linear(1024, 896),nn.PReLU(),nn.Dropout(),
		    nn.Linear(896, 768),nn.PReLU(),nn.Dropout(),
		    nn.Linear(768, 512),nn.PReLU(),nn.Dropout(),
		    nn.Linear(512, 384),nn.PReLU(),nn.Dropout(),
		    nn.Linear(384, 256),nn.PReLU(), nn.Dropout(),
		    nn.Linear(256, 256),nn.PReLU(), nn.Dropout(),
		    nn.Linear(256, 128),nn.PReLU(), nn.Dropout(),
		    nn.Linear(128, 64),nn.PReLU(), nn.Dropout(),
		    nn.Linear(64, 32),nn.PReLU(),
            nn.Linear(32, mlp_output_size)
        )
    
    def forward(self, x):
        # Separate obstacle and positional features
        obstacle_features = x[:, :28] # First 28 features
        positional_features = x[:, 28:].reshape(-1, 2, 2) # Reshape to (batch, seq_len, features)

        # Process obstacle features
        obstacle_out = self.obstacle_mlp(obstacle_features)

        # Process positional features
        lstm_out, (hn, cn) = self.lstm(positional_features)
        lstm_out = lstm_out[:, -1, :] # Use the last hidden state

        # Combine LSTM and MLP outputs
        combined_features = torch.cat((obstacle_out, lstm_out), dim=1)

        # Final MLP layers
        out = self.final_mlp(combined_features)
        return out