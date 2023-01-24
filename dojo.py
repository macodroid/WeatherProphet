import numpy as np
import torch


class Dojo:
    def __init__(
        self,
        model,
        loss_function,
        optimizer,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        device,
    ):
        self.model = model
        self.loss_fn = loss_function
        self.optimizer = optimizer
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader
        self.test_dl = test_dataloader
        self.device = device

    def train(self):
        train_losses = []
        val_losses = []
        self.model.train()
        for i, batch in enumerate(self.train_dl):
            x = batch[0].type(torch.FloatTensor).to(self.device)
            y = batch[1].type(torch.FloatTensor).to(self.device).reshape(-1, 1)
            self.optimizer.zero_grad()

            y_hat = self.model(x)
            loss = self.loss_fn(y_hat, y)
            loss.backward()
            train_losses.append(loss.item())
            self.optimizer.step()
            # if i % 100 == 0:
            #     print("Training loss at step {}: {}".format(i, loss.item()))
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.val_dl):
                x = batch[0].type(torch.FloatTensor).to(self.device)
                y = batch[1].type(torch.FloatTensor).to(self.device).reshape(-1, 1)

                out = self.model(x)
                loss = self.loss_fn(out, y)
                val_losses.append(loss.item())

        return np.mean(train_losses), np.mean(val_losses)

    def test(self):
        losses = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.test_dl):
                x = batch[0].type(torch.FloatTensor).to(self.device)
                y = (
                    batch[1]
                    .type(torch.FloatTensor)
                    .to(self.device)
                    .reshape(-1, 1)
                )

                y_hat = self.model(x)
                loss = self.loss_fn(y_hat, y)
                losses.append(loss.item())
        return losses
