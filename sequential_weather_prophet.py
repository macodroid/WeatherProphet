import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR

from dojo import Dojo
from gru import GRUWeatherProphet
from utils import (
    create_plot,
    get_device,
    load_train_data,
    load_validation_data,
    load_test_data,
    load_mean_std,
)
from dataset import TimeSeriesDataset
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # HYPER-PARAMETERS
    hidden_size = 64
    num_layers = 2

    name = "wtf_gru_1"
    device = get_device()
    print(device)

    train_features, train_labels = load_train_data()
    val_features, val_labels = load_validation_data()
    test_features, test_labels = load_test_data()
    mean, std = load_mean_std()

    window_size = 6
    train_dataset = TimeSeriesDataset(train_features, train_labels, window_size)
    val_dataset = TimeSeriesDataset(val_features, val_labels, window_size)
    test_dataset = TimeSeriesDataset(test_features, test_labels, window_size)

    batch_size = 256
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    gru_model = GRUWeatherProphet(
        input_size=8, hidden_size=hidden_size, output_size=1, num_layers=num_layers
    )
    gru_model.to(device)
    loss_function = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(gru_model.parameters(), lr=0.1, betas=(0.9, 0.999))
    scheduler = StepLR(optimizer, step_size=65, gamma=0.1)
    train_losses = []
    val_losses = []
    epochs = 150

    trainer = Dojo(
        model=gru_model,
        loss_function=loss_function,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        device=device,
    )

    for e in range(epochs):
        print(f"Epoch {e + 1}\n-------------------------------")
        train_loss, val_loss = trainer.train()
        scheduler.step()
        print("\nTrain loss at epoch {}: {}".format(e, train_loss))
        print("Val loss at epoch {}: {}".format(e, val_loss))
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    test_loss, predicted = trainer.test()
    # np.save("predicted.npy", predicted)
    print(f"Test MSE of the model: {np.mean(test_loss, axis=0):.4f}")
    torch.save(gru_model, f"models/WP{epochs}-{name}.pt")
    create_plot(train_losses, val_losses, test_loss, name, epochs)
    with open("scores_gru.txt", "a") as file:
        file.write("\n\n" + name)
        file.write("\n" + "Last train loss: " + str(train_losses[-1]))
        file.write("\n" + "Last val loss: " + str(val_losses[-1]))
        file.write("\n" + "Test loss: " + str(np.mean(test_loss, axis=0)))
