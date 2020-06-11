
from sklearn.metrics import accuracy_score
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from weak_deepards.dataset import ApneaECGDataset
from weak_deepards.models.base.dey import DeyNet


def train(model, loader, optim, epochs):
    with torch.enable_grad():
        loss_tally = 0
        n = 1
        sum_loss = 0
        for ep in range(epochs):
            for i, record, data, y in loader:
                x = Variable(data.float()).cuda()
                y = y.cuda()
                y_hat = model(x)
                loss = F.binary_cross_entropy_with_logits(y_hat, y.float())
                sum_loss += loss
                print("loss: {} average: {}".format(loss, sum_loss / n))
                loss.backward()
                optim.step()
                optim.zero_grad()
                n += 1


def test(model, loader):
    with torch.no_grad():
        acc_sum = 0
        n = 1
        for i, record, data, y in loader:
            x = data.float().cuda()
            y_hat = model(x).cpu()
            acc = accuracy_score(y.argmax(dim=1), F.softmax(y_hat, dim=1).argmax(dim=1))
            acc_sum += acc
            print('acc: {}, av: {}'.format(acc, acc_sum/n))
            n += 1


epochs = 10
model = DeyNet().cuda()
optim = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9, weight_decay=.00001)
dataset = ApneaECGDataset('/fastdata/apnea-ecg/physionet.org/files/apnea-ecg/1.0.0/', 'train', 'main', 'inter')
loader = DataLoader(dataset, batch_size=64, shuffle=True)
train(model, loader, optim, epochs)
dataset = ApneaECGDataset('/fastdata/apnea-ecg/physionet.org/files/apnea-ecg/1.0.0/', 'test', 'main', 'inter')
loader = DataLoader(dataset, batch_size=64, shuffle=True)
test(model, loader)
