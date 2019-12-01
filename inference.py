import torch
import math
import numpy as np
from lista_dataset import LISTADataset
from lista_model import LISTA

train_datasets = LISTADataset(train=True)
test_datasets = LISTADataset(train=False)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

A = train_datasets.A
T = 10
lam = 0.4
model = LISTA(A,T,lam)
model.to(device)
checkpoint=torch.load("lista.pt")
model.load_state_dict(checkpoint)


criterion = torch.nn.MSELoss()

def test(model, device, test_loader):
    model.eval()
    test_loss = [0] * model._T
    test_denom = 0
    with torch.no_grad():
        for y_, x_ in test_loader:
            y_, x_ = y_.to(device), x_.to(device)
            # 算二范数
            denom = torch.norm(x_)
            test_denom += denom.item()
            xhs_ = model(y_)
            for t in range(model._T):
                loss = criterion(xhs_[t], x_)
                test_loss[t] += loss.item()
    test_loss = np.array(test_loss)
    test_dB = 10 * np.log10(test_loss / test_denom)
    print("test_loss: {:.3e}, test_denom: {:.3e}".format(test_loss[-1], test_denom))
    np.save('test_dB.npy', test_dB)

test(model,device,test_datasets)


