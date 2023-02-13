import json
import ipdb
import torch as t
from torch.utils.tensorboard import SummaryWriter


# TODO: sistemare la rete
class Net(t.nn.Module): #costruisco la mia rete neurale
    def __init__(self):
        super().__init__()
        self.linear1 = t.nn.Linear(3*107, 2000) #immagine formato 32x32 
        self.Linear2 = t.nn.Linear(2000, 2000) #ricevo in input 2000 e faccio uscire 20000 
        self.Linear3 = t.nn.Linear(1000, 10) #input 1000 e output 10
    
    def forward(self, x):
        x = self.linear1(x)


def preprocess(data):
    rna = 'ACGU'
    structure = '().'
    # S: paired "Stem" M: Multiloop I: Internal loop B: Bulge H: Hairpin loop E: dangling End X: eXternal loop
    predicted_loop_type = 'SMBHIEX'
    xs = []
    ys = []
    for d in data:
        x = d['sequence']
        x_0 = [rna.index(c) for c in x]
        x_1 = [structure.index(c) for c in d['structure']]
        x_2 = [predicted_loop_type.index(c) for c in d['predicted_loop_type']]
        x = [x_0, x_1, x_2]
        y_0 = d['reactivity']
        y_1 = d['deg_Mg_pH10']
        y_2 = d['deg_Mg_50C']
        y_3 = d['deg_pH10']
        y_4 = d['deg_50C']
        y = [y_0, y_1, y_2, y_3, y_4]
        xs.append(x)
        ys.append(y)
    xs = t.tensor(xs, dtype=t.float32)
    ys = t.tensor(ys, dtype=t.float32)
    return xs, ys


def test(net, loss_function, testset, device):
    with t.no_grad():
        correct = 0
        total = 0
        tot_loss = 0
        for x, y in testset:
            x = x.view(x.shape[0], 3*107).to(device)
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
    # loss_function = t.nn.CrossEntropyLoss() # funzione di loss, cioe' quella che calcola quanto la rete sta sbagliando
    loss_function = t.nn.MSELoss()
    writer = SummaryWriter(log_dir="runs/cifar10")
    total = 0
    for epoch in range(epochs):
        tot_loss = 0
        for x, y in trainset:
            optimizer.zero_grad()  # resetta i gradienti, te la spiego dopo
            x = x.view(x.shape[0], 3*107).to(device)
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
    lines = []
    with open('train.json') as f:
        for line in f:
            lines.append(json.loads(line))

    xs, ys = preprocess(lines)
    rand_index = t.randperm(len(xs))
    train_size = int(len(xs) * 0.8)
    train_index = rand_index[:train_size]
    test_index = rand_index[train_size:] # [4, 5, 6, 7, 8, 9]
    train_xs = xs[train_index]
    train_ys = ys[train_index]
    test_xs = xs[test_index]
    test_ys = ys[test_index]

    trainset = t.utils.data.TensorDataset(train_xs, train_ys)
    testset = t.utils.data.TensorDataset(test_xs, test_ys)
    trainset = t.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
    testset = t.utils.data.DataLoader(testset, batch_size=32, shuffle=True)

    for x, y in trainset:
        print(x.shape, y.shape)
        break

    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    net = Net().to(device)
    train(net, trainset, testset, device, epochs=10)


if __name__ == '__main__':
    main()






