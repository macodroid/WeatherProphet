import numpy as np
import torch
from dataset import TimeSeriesDataset
from torch.utils.data import DataLoader
from dojo import Dojo
from mlp import ClassicWeatherProphet

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    train_data = np.load("dataset/train_data.npy")
    val_data = np.load("dataset/val_data.npy")
    test_dataset = np.load("dataset/test_data.npy")
    mean, std = np.load("dataset/stat.npy")

    window_size = 6
    train_dataset = TimeSeriesDataset(train_data, window_size, mean[-1], std[-1])
    val_dataset = TimeSeriesDataset(val_data, window_size, mean[-1], std[-1])
    test_dataset = TimeSeriesDataset(test_dataset, window_size, mean[-1], std[-1])

    batch_size = 32
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    input_size = window_size * train_data.shape[1]
    # predict temperature for the next hour
    output_size = 1
    # define model
    model = ClassicWeatherProphet(input_size=input_size, output_size=output_size)
    model.to(device)
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    trainer = Dojo(
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        device=device,
    )
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_train_accs = []
    epoch_val_accs = []
    epochs = 50

    for e in range(epochs):
        print(f"Epoch {e + 1}\n-------------------------------")
        train_loss, val_loss = trainer.train()
        # scheduler.step()
        print("\nTrain loss at epoch {}: {}".format(e, train_loss))
        print("Val loss at epoch {}: {}".format(e, val_loss))
        epoch_train_losses.append(train_loss)
        epoch_val_losses.append(val_loss)

    acc_test = trainer.test()
    for ac in acc_test:
        print(f"Test Accuracy of the model: {ac * 100:.2f}")
    # print(
    #     f"\n-------------------------------\nTest Accuracy of the model: {acc_test * 100:.2f}"
    # )

    # [ ] TODO: Create MLP model
    # [ ] TODO: For loop for training
    # [ ] TODO: Create GRU model
