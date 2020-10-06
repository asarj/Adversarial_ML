from data_loader import ImageDataset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime

class CNN(nn.Module):

    device = None
    lr = 1e-3
    limit = 8

    def __init__(self, dataset:ImageDataset, epochs:int, limit=6, verbose=1):
        """
        Class to create a CNN using pytorch with training/evaluation scripts

        :param ImageDataset epochs: the image dataset used to load the data, includes PyTorch dataloaders
        :param int epochs: number of epochs to train the model for
        :param int limit: patience level for early stopping
        """

        super(CNN, self).__init__()
        self.dataset = dataset
        self.epochs = epochs
        self.limit = limit
        self.verbose = verbose
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        self.to(self.device)

    def forward(self, x):
        """
        fill in later

        :param x:
        :return:
        """

        x = x.view(x.size(0), 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), 512)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def run_training_script(self):

        self.train()

        if self.verbose:
            print("Starting training script...")

        train_data_loader = self.dataset.train
        valid_data_laoder = self.dataset.validation

        train_loss, train_acc, valid_loss, valid_acc = list(), list(), list(), list()
        best, no_change = 0, 0

        start = datetime.now()
        for epoch in range(self.epochs):

            if self.verbose:
                print(f"Epoch {epoch + 1} / {self.epochs}:")


            # Eval on train data
            tot, t_loss, t_acc = 0, 0, 0

            for data, labels in train_data_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self(data)
                predictions = outputs.argmax(dim=1)
                loss = self.loss_func(outputs, labels)
                loss.backward()
                self.optimizer.step()

                t_acc += (predictions == labels).sum().item()
                t_loss += loss.item() * len(data)

                end = datetime.now()

                if self.verbose:
                    tot += len(data)
                    print(f'[{tot} / {self.dataset.train_size}] - {(end - start)} -',
                          f'train loss = {(t_loss / tot):.4f},',
                          f'train acc = {(t_acc / tot):.4f}',
                          end='\r'
                          )

            t_loss = t_loss / self.dataset.train_size
            t_acc = t_acc / self.dataset.train_size
            train_acc.append(t_acc)
            train_loss.append(t_loss)

            # Eval on validation
            v_loss, v_acc = 0, 0
            with torch.no_grad():
                for data, labels in train_data_loader:
                    data = data.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self(data)
                    predictions = outputs.argmax(dim=1)

                    loss = self.loss_func(outputs, labels)

                    v_acc += (predictions == labels).sum().item()
                    v_loss += loss.item() * len(data)

            v_loss = v_loss / self.dataset.train_size
            v_acc = v_acc / self.dataset.train_size
            valid_acc.append(v_acc)
            valid_loss.append(v_loss)

            end = datetime.now()

            if self.verbose:
                print(f'[{tot} / {self.dataset.train_size}] - {(end - start)} -',
                      f'train loss = {t_loss:.4f},',
                      f'train acc = {t_acc:.4f},',
                      f'valid loss = {v_loss:.4f},',
                      f'valid acc = {v_acc:.4f}'
                      )

            if v_acc > best:
                no_change = 0
                best = v_acc
            else:
                no_change += 1

            if no_change >= self.limit:
                if self.verbose:
                    print("Early stopping...")
                break

        end = datetime.now()

        if self.verbose:
            print(f"Total time taken to train the model: {end - start}")

        return train_loss, train_acc, valid_loss, valid_acc

    def evaluate(self, data, labels):
        self.eval()

        with torch.no_grad():
            data = torch.FloatTensor(data).to(self.device)
            labels = torch.LongTensor(labels).to(self.device)

            outputs = self(data)
            predictions = outputs.argmax(dim=1)

            loss = self.loss_func(outputs, labels).item()
            acc = (predictions == labels).double().mean().item()

            if self.verbose:
                print(f'test loss: {loss:.4f}, test acc: {acc:.4f}')

            return loss, acc

    def predict(self, data):
        self.eval()
        with torch.no_grad():
            data = torch.FloatTensor(data).to(self.device)
            outputs = self(data)
            predictions = outputs.argmax(dim=1).cpu().numpy()

            return predictions


if __name__ == "__main__":
    mnist = ImageDataset(type='TORCH_MNIST')
    n_train = len(mnist.train.dataset)
    n_valid = len(mnist.validation.dataset)
    n_test = len(mnist.test.dataset)
    image_shape = mnist.shape
    n_classes = mnist.num_classes

    print("Number of training examples =", n_train)
    print("Number of testing examples =", n_test)
    print("Number of validation examples =", n_valid)
    print("Image data shape =", image_shape)
    print("Number of classes =", n_classes)

    epochs = 50
    limit = 5
    verbose = 1
    model_path = "mnist_model.pt"

    cnn = CNN(dataset=mnist, epochs=epochs, limit=limit, verbose=verbose)

    # For creating from scratch
    # cnn.run_training_script()
    # cnn.evaluate(mnist.x_test, mnist.y_test)
    # torch.save(cnn, model_path)

    # from working with pretrained model
    cnn = torch.load(model_path)