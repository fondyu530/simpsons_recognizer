import torch


class Trainer:
    """Class for torch.nn model training."""

    def __init__(self, device=torch.device("cpu")):
        """device - torch.device object, current device."""
        self.device = device

    def compute_accuracy(self, model, test_loader):
        """Computes accuracy of pretrained model

        model - torch.nn object, neural net model;
        test_loader - torch.utils.data.DataLoader object, consists of X labels - images, y labels - correct class;

        return - float, computed accuracy."""

        model.eval()  # evaluation mode

        total_correct = 0
        total_samples = 0

        for x, y in test_loader:
            x_gpu = x.to(self.device)
            y_gpu = y.to(self.device)

            prediction = model(x_gpu)

            _, indices = torch.max(prediction, 1)
            del prediction
            torch.cuda.empty_cache()
            total_correct += torch.sum(indices == y_gpu)
            total_samples += len(y_gpu)
        return float(total_correct / total_samples)

    def train_model(self, model, train_loader, val_loader, num_epochs, optimizer, loss, scheduler=None):
        """Trains neural net with custom optimizer, scheduler and loss function

        model - model - torch.nn object, neural net model;
        train_loader - torch.utils.data.DataLoader object, consists of data for model training;
        val_loader - torch.utils.data.DataLoader object, consists of data for model validation;
        num_epochs - int, number of epochs;
        optimizer - optimizer from torch.optim module;
        loss - loss function from torch.nn module;
        scheduler - scheduler from torch.optim.lr_scheduler module;
        device - torch.device object, current device;

        return:
            loss_history - list, list of average loss values for every training epoch;
            train_history - list, list of accuracy values on training data for every epoch;
            val_history - list, list of accuracy values on validation data for every epoch."""

        loss_history = []
        train_history = []
        val_history = []

        for epoch in range(num_epochs):
            model.train()  # training mode

            loss_accumulator = 0
            correct_samples = 0
            total_samples = 0

            for x, y in train_loader:
                x_gpu = x.to(self.device)
                y_gpu = y.to(self.device)

                prediction = model(x_gpu)
                loss_value = loss(prediction, y_gpu)
                loss_accumulator += loss_value

                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()

                _, indices = torch.max(prediction, 1)
                correct_samples += torch.sum(indices == y_gpu)
                total_samples += len(y_gpu)

            if scheduler:
                scheduler.step()

            aver_loss = loss_accumulator / len(train_loader)
            train_accuracy = float(correct_samples) / total_samples
            val_accuracy = self.compute_accuracy(model, val_loader)

            loss_history.append(aver_loss)
            train_history.append(train_accuracy)
            val_history.append(val_accuracy)

            print(f"Epoch: {epoch + 1}, loss: {aver_loss}")
            print(f"Train accuracy: {train_accuracy}, val_accuracy: {val_accuracy}", end="\n\n")

        return loss_history, train_history, val_history
