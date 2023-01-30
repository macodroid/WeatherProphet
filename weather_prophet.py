import numpy as np
import torch
from utils import create_plot, get_device
from dataset import TimeSeriesDataset
from torch.utils.data import DataLoader
from dojo import Dojo
from mlp import ClassicWeatherProphet
from torch.optim.lr_scheduler import StepLR

if __name__ == "__main__":
    name = "mlp_jit_L1"
    device = get_device()
    print(device)

    train_features = np.load("dataset/train_features.npy")
    train_labels = np.load("dataset/train_labels.npy")

    val_features = np.load("dataset/val_features.npy")
    val_labels = np.load("dataset/val_labels.npy")

    test_features = np.load("dataset/test_features.npy")
    test_labels = np.load("dataset/test_labels.npy")

    mean, std = np.load("dataset/stat.npy")

    window_size = 6
    train_dataset = TimeSeriesDataset(train_features, train_labels, window_size)
    val_dataset = TimeSeriesDataset(val_features, val_labels, window_size)
    test_dataset = TimeSeriesDataset(test_features, test_labels, window_size)

    batch_size = 512
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    input_size = window_size * train_features.shape[1]
    # predict temperature for the next hour
    output_size = 1
    # define model
    model = ClassicWeatherProphet(input_size=input_size, output_size=output_size)
    model.to(device)
    loss_function = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))
    scheduler = StepLR(optimizer, step_size=65, gamma=0.1)
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
    epochs = 300

    for e in range(epochs):
        print(f"Epoch {e + 1}\n-------------------------------")
        train_loss, val_loss = trainer.train()
        scheduler.step()
        print("\nTrain loss at epoch {}: {}".format(e, train_loss))
        print("Val loss at epoch {}: {}".format(e, val_loss))
        epoch_train_losses.append(train_loss)
        epoch_val_losses.append(val_loss)

    test_loss, predicted = trainer.test()
    np.save("predicted.npy", predicted)
    print(f"Test MSE of the model: {np.mean(test_loss, axis=0):.4f}")
    scripted_model = torch.jit.script(model)
    torch.jit.save(scripted_model, f"{name}.pt")
    create_plot(epoch_train_losses, epoch_val_losses, test_loss, name, epochs)
    with open("scores.txt", "a") as file:
        file.write("\n\n" + name)
        file.write("\n" + "Last train loss: " + str(epoch_train_losses[-1]))
        file.write("\n" + "Last val loss: " + str(epoch_val_losses[-1]))
        file.write("\n" + "Test loss: " + str(np.mean(test_loss, axis=0)))
