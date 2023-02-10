import torch as t
import torchvision as tvision
import numpy as np
import matplotlib.pyplot as plt
import ipdb
from torch.utils.tensorboard import SummaryWriter

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
        return x


def test(net, loss_function, testset, device):
    with t.no_grad():
        correct = 0
        total = 0
        tot_loss = 0
        for x, y in testset:
            x = x.view(x.shape[0], 32 * 32 * 3).to(device)
            y = y.to(device)
            y_pred = net(x)
            loss = loss_function(y_pred, y)
            tot_loss += loss.detach().item()
            _, predicted = t.max(y_pred.data, 1)
            total += y.shape[0]
            correct += (predicted == y).sum().item()
    print(f"Test loss: {tot_loss/total} Accuracy: {correct / total}")

def train(net, trainset, testset, device, epochs=10):
    # TODO: aggiungere l'ottimizzatore
    optimizer = t.optim.Adam( # algoritmo che aggiorna i parametri della rete
            net.parameters(), 
            lr=0.001 # iperparametro: la velocita' con cui i parametri vengono aggiornati
    ) # 
    loss_function = t.nn.CrossEntropyLoss() # funzione di loss, cioe' quella che calcola quanto la rete sta sbagliando
    writer = SummaryWriter(log_dir="runs/cifar10")
    total = 0
    for epoch in range(epochs):
        tot_loss = 0
        for x, y in trainset:
            optimizer.zero_grad()  # resetta i gradienti, te la spiego dopo
            x = x.view(x.shape[0], 32 * 32 * 3).to(device)
            y = y.to(device)
            y_pred = net(x)  # shape (batch_size, 10)
            loss = loss_function(y_pred, y)
            loss.backward()  # calcola i gradienti, te la spiego dopo
            optimizer.step()  # aggiorna i parametri della rete
            tot_loss += loss.detach().item()
            total += y.shape[0]
            writer.add_scalar("Loss/train", loss, total)
        print(f"Epoch {epoch} train loss: {tot_loss/total}")
        test(net, loss_function, testset, device)
        t.save(net.state_dict(), 'checkpoint.pt')



def main():
    
    batch_size = 128 # iperparametro: il numero di esempi che vengono processati insieme

    # loads CIFAR10 dataset
    device = t.device("cuda" if t.cuda.is_available() else "cpu") # dove i dati vengono processati (cpu o gpu)
    print(f"Using device {device}")
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
    train(net, 
          trainset,
          testset, 
          device, 
          epochs=20 # iperparametro: il numero di volte che il modello vede tutti i dati
    )


if __name__ == "__main__":
    main()
