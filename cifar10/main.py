import torch as t
import torchvision as tvision
import numpy as np
import matplotlib.pyplot as plt
import ipdb


class Net(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = t.nn.Linear(32 * 32 * 3, 2000)
        self.linear2 = t.nn.Linear(2000, 1000)
        self.linear3 = t.nn.Linear(1000, 10)

    def forward(self, x):
        x = self.linear1(x)
        x = t.nn.functional.relu(x)
        x = self.linear2(x)
        x = t.nn.functional.relu(x)
        x = self.linear3(x)
        # TODO
        return x


def train(net, trainset, testset, batch_size, device, epochs=10):
    # TODO: aggiungere l'ottimizzatore
    optimizer = t.optim.Adam(net.parameters(), lr=0.001)
    loss_function = t.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for x, y in trainset:
            optimizer.zero_grad()  # resetta i gradienti, te la spiego dopo
            ipdb.set_trace()
            x = x.view(batch_size, 32 * 32 * 3).to(device)
            y = y.to(device)
            y_pred = net(x)  # shape (batch_size, 10)
            loss = loss_function(y_pred, y)
            loss.backward()  # calcola i gradienti, te la spiego dopo
            optimizer.step()  # aggiorna i parametri della rete


def main():
    # TODO: aggiungere il training
    # TODO: aggiungere il test
    # TODO: usare la GPU
    batch_size = 128
    # loads CIFAR10 dataset
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    trainset = t.utils.data.DataLoader(
        tvision.datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=tvision.transforms.ToTensor(),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    testset = t.utils.data.DataLoader(
        tvision.datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=tvision.transforms.ToTensor(),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    net = Net().to(device)
    train(net, trainset, testset, batch_size, device, epochs=10)


if __name__ == "__main__":
    main()
