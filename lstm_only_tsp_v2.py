import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# define scaler class
class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        self.mean = data.mean((0, 1))
        self.std = data.std((0, 1))

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / (std + 1e-8)

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean

scaler = StandardScaler()

# prepare data
input_seq_len = 128 # number of time steps for input
target = "nonfarm" # target col

df_raw = pd.read_csv("data/HNF.csv", delimiter="\t")
# remove date col and reorder cols
cols = list(df_raw.columns); cols.remove(target); cols.remove('date')
df_raw = df_raw[cols+[target]]
# split train val test sets
usable_data_len = len(df_raw) - input_seq_len + 1
num_train = int(usable_data_len * 0.7)
num_val = usable_data_len - num_train

all_data = []
for i in range(usable_data_len):
    all_data.append(df_raw.values[i:i + input_seq_len])

# save predict x here
pred_x = df_raw.values[-input_seq_len:].copy()
pred_x = torch.from_numpy(pred_x).float()

whole_series_raw = df_raw.values[:,-1] # save for drawing

# random.shuffle(all_data)
all_data = np.array(all_data)

# normalize
trainset = all_data[:num_train]
scaler.fit(trainset)
all_data = scaler.transform(all_data)
# splits sets
x_train, y_train = all_data[:num_train,:-1], all_data[:num_train,1:]
x_val, y_val = all_data[num_train:,:-1], all_data[num_train:,1:]


# defile a dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y

    def __getitem__(self, index):
        x = self.data_x[index]
        y = self.data_y[index]
        return x, y

    def __len__(self):
        return len(self.data_x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Convert the data into PyTorch tensors
x_train = torch.from_numpy(x_train).float().to(device)
y_train = torch.from_numpy(y_train).float().to(device)
x_val = torch.from_numpy(x_val).float().to(device)
y_val = torch.from_numpy(y_val).float().to(device)
pred_x = pred_x.to(device)

train_dataset = TimeSeriesDataset(x_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, nlayers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, nlayers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return output

    def predict(self, x, h, c):
        lstm_out, (h, c) = self.lstm(x, (h, c))
        output = self.fc(lstm_out)
        return output, (h, c)

# Set the hyperparameters
input_size = x_train.shape[-1]
output_size = x_train.shape[-1]
hidden_size = 128
nlayers = 2
num_epochs = 20
learning_rate = 0.001

# Instantiate the LSTM model
model = LSTMModel(input_size, hidden_size, nlayers, output_size)
model.to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print("======Training Starts======")
# Train the model
for epoch in range(num_epochs):
    # train
    model.train()
    for x, y in train_dataloader:
        outputs = model(x)
        loss = criterion(outputs, y.squeeze())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
    # eval
    model.eval()
    eval_loss = criterion(model(x_val), y_val.squeeze())
    print(f"Eval loss: {eval_loss.item():.4f}")


print("======Evaluation Starts======")
model.eval()
with torch.no_grad():
    eval_out = model(x_val)
    val_plot = np.ones_like(whole_series_raw) * np.nan
    val_plot[num_train + input_seq_len - 1:] = scaler.inverse_transform(eval_out)[:,-1,-1].cpu().detach().numpy()
    train_out = model(x_train)
    train_plot = np.ones_like(whole_series_raw) * np.nan
    train_plot[input_seq_len:num_train + input_seq_len] = scaler.inverse_transform(train_out)[:,-1,-1].cpu().detach().numpy()


print("======Prediction Starts======")
data_csv_path = "data/HNF_predict.csv"
df_raw = pd.read_csv(data_csv_path, delimiter="\t")
df_raw_pred = df_raw[cols+[target]].copy()
ext_variables = df_raw_pred.values[-16:]
ext_variables = scaler.transform(ext_variables)
# Make predictions
model.eval()
with torch.no_grad():
    # initial values
    x_prev = pred_x.unsqueeze(0)
    h = torch.randn(nlayers, 1, hidden_size).to(device)
    c = torch.randn(nlayers, 1, hidden_size).to(device)
    predictions = []
    # auto-regressive predictions
    for i in range(16):
        x, (h, c) = model.predict(x_prev, h, c)
        predictions.append(scaler.inverse_transform(x)[0, -1, -1].item())
        x_prev = torch.cat([x_prev[:,1:,:], x[:,-1,:].unsqueeze(-2)], dim=-2)
        # use the external variables in prediction
        for j in range(x_prev.shape[-1] - 1):
            if ext_variables[i, j] != np.nan:
                x_prev[0, -1, j] = ext_variables[i, j]
    

print("======Plotting Starts======")

date_col = df_raw["date"]
nf_col = df_raw["nonfarm"]


figure_exp_dir = "figures"
os.makedirs(figure_exp_dir, exist_ok=True)

date_series = date_col.values
true = nf_col.values

date_series_pred = date_series.copy()[-(len(predictions)):]
date_series_train_eval = date_series.copy()[:len(whole_series_raw)]

plt.plot(date_series, true, label=f"true", color="blue")
plt.plot(date_series_pred, predictions, label=f"pred", color="red")
plt.plot(date_series_train_eval, val_plot, label=f"eval", color="green")
plt.plot(date_series_train_eval, train_plot, label=f"train", color="pink")

plt.xlabel("Date")
plt.ylabel("NF")
plt.legend()

plt.xticks(range(len(date_series))[::48], date_series[::48])

plt.savefig(os.path.join(figure_exp_dir, f"plot_day.png"))
plt.clf()