from polygon import RESTClient
import pandas as pd
import tkinter as tk
from tkinter import Toplevel
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import messagebox, simpledialog, ttk
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime, timedelta

client = RESTClient('NFcVTTVW4_4AtBP4hkpJA9x78dAEa0QT')

data_dict = {}

def stock_data(option):
    datarequest = client.get_aggs(ticker = option, multiplier = 1, timespan = 'day', from_ = '2023-03-01', to = '2025-03-01')

    pricedata = pd.DataFrame(datarequest)
    pricedata['Date'] = pricedata['timestamp'].apply(lambda x: pd.to_datetime(x*1000000))
    pricedata = pricedata.set_index('Date')

    global data_dict
    for date, high in pricedata["high"].items():
        data_dict[date.strftime("%Y-%m-%d")] = high

    pricedata[['high']].to_csv(f"{option}.csv", mode = 'w')


def plot():
    option = st_combobox.get()
    stock_data(option)
    plot_window = Toplevel(app)
    plot_window.title("Stock Price Chart")
    plot_window.geometry("800x600")

    dates = [datetime.strptime(date, '%Y-%m-%d') for date in data_dict.keys()]
    prices = list(data_dict.values())

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(dates, prices, marker='', linestyle='-')
    ax.set_title(f"Stock Prices for {option}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid()

    canvas = FigureCanvasTkAgg(fig, master=plot_window)
    canvas.draw()
    canvas.get_tk_widget().pack()


def calculate_profit():
    stock = st_combobox.get()
    investment_amount = float(text_box1.get(1.0, tk.END).strip())
    investment_date = text_box2.get(1.0, tk.END).strip()


    stock_data(stock)

    try:
        datetime.strptime(investment_date, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        messagebox.showerror("Error", "Invalid date format. Use YYYY-MM-DD.")
        return

    try:
        data_dict[investment_date]
    except KeyError:
        messagebox.showerror("Error", "The stock market was closed on this date.")

    initial_price = data_dict[investment_date]
    latest_price = list(data_dict.values())[-1]


    shares_bought = investment_amount / initial_price
    current_value = shares_bought * latest_price
    profit = current_value - investment_amount

    messagebox.showinfo("Results", f"Initial investment: {investment_amount}\n" f"Current value: {current_value}\n" f"Profit / Loss: {profit}\n")


class LSTMModel(nn.Module):
    def __init__(self, input_dim = 1, hidden_dim = 50, layer_dim = 2, output_dim = 1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def prepare_data(prices, seq_length=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(np.array(prices).reshape(-1, 1)) #LSTM works better for normalized data
    X, Y = [], []
    for i in range(len(scaled_prices) - seq_length):
        X.append(scaled_prices[i:i + seq_length])
        Y.append(scaled_prices[i + seq_length])
    X, Y = np.array(X), np.array(Y)
    X_train, Y_train = X, Y #used all the data for training because the purpose is to test the accuracy of the model

    return X_train, Y_train, scaler


def train_lstm(X_train, Y_train, epochs=50, lr=0.001):
    X_train_torch = torch.tensor(X_train, dtype=torch.float32)
    Y_train_torch = torch.tensor(Y_train, dtype=torch.float32)
    dataset = TensorDataset(X_train_torch, Y_train_torch)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = LSTMModel()
    msqerr = nn.MSELoss() #mean square error
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for inputs, targets in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = msqerr(outputs, targets)
            loss.backward()
            optimizer.step()

    return model

def predict_stock_price():
    stock = st_combobox.get()

    stock_data(stock)

    prices = list(data_dict.values())
    dates = list(data_dict.keys())

    X_train, Y_train, scaler = prepare_data(prices)
    model = train_lstm(X_train, Y_train)

    input_seq = torch.tensor(X_train[-1], dtype=torch.float32).unsqueeze(0) #we add an extra batch at 0
    predictions = []

    for iterator in range(60):
        with torch.no_grad():
            pred = model(input_seq).detach().numpy()
        predictions.append(pred[0][0])
        new_input = np.roll(input_seq.numpy(), -1, axis=1) #left shift to make room
        new_input[0, -1, 0] = pred[0][0] #use already made prediction for the ones in the future
        input_seq = torch.tensor(new_input, dtype=torch.float32)

    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    last_date = datetime.strptime(dates[-61], "%Y-%m-%d")
    pred_dates = [(last_date + pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 61)]

    plot_predictions(pred_dates, predicted_prices, stock)


def plot_predictions(pred_dates, predicted_prices, stock):
    plot_window = Toplevel(app)
    plot_window.title("Stock Price Prediction")
    plot_window.geometry("800x600")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(pred_dates, predicted_prices,  label="Predicted Prices", marker='', linestyle='-', color='red')

    real_dates = []
    real_prices = []

    for date in pred_dates:
        if date in data_dict:
            real_dates.append(date)
            real_prices.append(data_dict[date])

    ax.plot(real_dates, real_prices, label="Real Prices", linestyle='-', color='blue')

    ax.set_xticks(pred_dates[::10])

    ax.set_title(f"Stock Price Prediction for {stock}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid()

    canvas = FigureCanvasTkAgg(fig, master=plot_window)
    canvas.draw()
    canvas.get_tk_widget().pack()


app = tk.Tk()
app.title("Stock Market Analysis")
app.geometry("800x600")

button_frame = tk.Frame(app)
button_frame.pack(pady=10)

combo_frame = tk.Frame(app)
combo_frame.pack(pady=10)

st_label = tk.Label(combo_frame, text="Select a stock:")
st_label.grid(row=0, column=0, padx=5)
st_combobox = ttk.Combobox(combo_frame, values=["AAPL", "IBM", "NVDA"])
st_combobox.set("")
st_combobox.grid(row=0, column=1, padx=5)

view_button = tk.Button(button_frame, text="View Chart", command=plot)
view_button.pack(side=tk.LEFT, padx=5)

investment_frame = tk.Frame(app)
investment_frame.pack(pady=10)

text_box1_label = tk.Label(investment_frame, text="Enter the amount of money you invested ($):")
text_box1_label.grid(row=0, column=0, padx=5, pady=5)
text_box1 = tk.Text(investment_frame, height=1, width=10)
text_box1.grid(row=0, column=1, padx=5, pady=5)

date_frame = tk.Frame(app)
date_frame.pack(pady=10)

text_box2_label = tk.Label(date_frame, text="Enter the date when you invested (YYYY-MM-DD):")
text_box2_label.grid(row=1, column=0, padx=5, pady=5)
text_box2 = tk.Text(date_frame, height=1, width=10)
text_box2.grid(row=1, column=1, padx=5, pady=5)

calc_button = tk.Button(button_frame, text="Calculate profit", command=calculate_profit)
calc_button.pack(side=tk.LEFT, padx=5)

predict_button = tk.Button(button_frame, text="Predict Price", command=predict_stock_price)
predict_button.pack(side=tk.LEFT, padx=5)

app.mainloop()