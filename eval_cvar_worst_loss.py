import argparse
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from algs import find_opt_lbd
import pandas as pd
from config import preprocess_compas, get_dataset, populate_config
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

def worst_group_loss(args, lbd):
    if 'compas' in args.file:
        df = pd.read_csv('compas-scores-two-years.csv')
        X, y = preprocess_compas(df)
        input_dim = len(X.columns)
        X, y = X.to_numpy().astype('float32'), y.to_numpy()
        X[:, 4] /= 10
        X[X[:, 7] > 0, 7] = 1 # Race: White (0) and Others (1)
        domain_fn = [
        lambda x: (x[:, 7] == 0) & (x[:, 6] == 1), # White and Female
        lambda x: (x[:, 7] == 1) & (x[:, 6] == 0), # Other and Male
        lambda x: (x[:, 7] == 0) & (x[:, 6] == 0), # White and Male
        lambda x: (x[:, 6] == 1) & (x[:, 7] == 1), # Other and Female
        ]
        # Split the dataset: train-test = 70-30
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                            random_state=42, shuffle=True)                
    if 'cifar10_unbal' in args.file:
        args.dataset = 'cifar10_unbal'
        dataset_train, dataset_test_train, dataset_valid, dataset_test, model, label_id = get_dataset(args)
        X_train = dataset_train.data
        Y_train = dataset_train.targets
        X_test = dataset_test.data
        Y_test = dataset_test.targets
        domain_fn = []
        for i in range(10):
            domain_fn.append(lambda y: y == i)
    elif 'cifar100' in args.file:
        args.dataset = 'cifar100'
        dataset_train, dataset_test_train, dataset_valid, dataset_test, model, label_id = get_dataset(args)
        X_train = dataset_train.data
        Y_train = dataset_train.targets
        X_test = dataset_test.data
        Y_test = dataset_test.targets
        domain_fn = []
        for i in range(100):
            domain_fn.append(lambda y: y == i)
    elif 'cifar10' in args.file:
        args.dataset = 'cifar10'
        dataset_train, dataset_test_train, dataset_valid, dataset_test, model, label_id = get_dataset(args)
        X_train = dataset_train.data
        Y_train = dataset_train.targets
        X_test = dataset_test.data
        Y_test = dataset_test.targets
        domain_fn = []
        for i in range(10):
            domain_fn.append(lambda y: y == i)
    
    if 'synthetic' in args.file:
        args.dataset = 'synthetic'
        dataset_train, dataset_test_train, dataset_valid, dataset_test, model, label_id = get_dataset(args)
        X_train = dataset_train.X
        Y_train = dataset_train.y
        X_test = dataset_test.X
        Y_test = dataset_test.y
        domain_fn = []
        for i in range(2):
            domain_fn.append(lambda y: y == i)
    num_domains = len(domain_fn)
    dct = {}
    for inputs in ['train', 'test']:
        group_correct = np.zeros((num_domains,), dtype=int)
        group_loss = np.zeros((num_domains,), dtype=float)
        group_num = np.zeros((num_domains,), dtype=int)
        for i in range(num_domains):
            if 'cifar10_unbal' in args.file or 'cifar10' in args.file or 'synthetic' in args.file or 'cifar100' in args.file:
                g = domain_fn[i](eval('Y_' + inputs))
            else:
                g = domain_fn[i](eval('X_' + inputs))
            group_correct[i] += (lbd @ mat[inputs + '_correct'])[g].sum().item()
            group_loss[i] += (lbd @ (1 - mat[inputs + '_correct']))[g].sum().item()
            group_num[i] += g.sum().item()
        dct[inputs] = [group_correct, group_loss, group_num]
    
    print("Train Sizes", dct['train'][2], "Test Sizes", dct['test'][2], '\n')
    top = ""
    for i in range(num_domains):
        top += "{:.3f} / {:.3f} || ".format(dct['train'][1][i] / dct['train'][2][i], dct['test'][1][i] / dct['test'][2][i])
    top += "max: {:.3f} / {:.3f}".format((dct['train'][1] / dct['train'][2]).max(), (dct['test'][1] / dct['test'][2]).max())
    print(top)

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str)
parser.add_argument('--data_root', type=str, default="data/")
parser.add_argument('--n_val', type=int, default=5000)
parser.add_argument('--download', type=bool, default=False)
parser.add_argument('--width', type=int, default=1)
args = parser.parse_args()

mat = sio.loadmat(args.file)

train_correct = mat['train_correct']
print("train_correct", train_correct.mean(axis=1) * 100)
val_correct = mat['val_correct']
test_correct = mat['test_correct']
T = train_correct.shape[0]
print("T", T)
if len(mat['gamevalues']) > 0:
    best_t = np.argmin(mat['gamevalues'])
else:
    best_t = 0
print("Best t:", best_t)

P_erm = np.zeros((T,))
P_erm[0] = 1

P_uniform = np.ones((T,)) / T

zo_loss = 1 - train_correct.mean(axis=1)
P_adaboost = np.log((1 - zo_loss) / zo_loss)
P_adaboost[P_adaboost < 0] = 0
P_adaboost = P_adaboost / P_adaboost.sum()

if 'rai' in args.file:
    P_rai = mat['hypothesis_weight_history'][:, best_t][0]
    P_rai = np.concatenate([P_rai, np.zeros((T - P_rai.shape[0], 1))], axis=0)
    P_rai = P_rai.reshape(-1)
else:
    P_rai = np.zeros((T,))
P_rai = P_rai / P_rai.sum()
if 'compas' in args.file:
    P_lp = find_opt_lbd(train_correct, 0.7)
else:
    P_lp = find_opt_lbd(val_correct, 0.7)


print("\n\nTrain Size: {}, Val Size: {}, Test Size: {}".format(train_correct.shape[1], val_correct.shape[1], test_correct.shape[1]))

def get_cvar_loss(lbd, correct, alpha):
    avg_loss = 1 - lbd @ correct
    avg_loss = np.sort(avg_loss)[::-1]
    n = avg_loss.shape[0]
    return format(avg_loss[:int(alpha * n)].mean(), '.3f')

aps = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
print("Train/Val/Test CVaR Loss for alpha = {}, {}, {}, {}, {}, {}".format(0.1, 0.3, 0.5, 0.7, 0.9, 1.0))

print('{:>20} {:>20} {:>20} {:>20} {:>20} {:>20}'.format(*aps))
ord = ['erm', 'uniform', 'adaboost', 'lpboost', 'rai']
for i, lbd in enumerate([P_erm, P_uniform, P_adaboost, P_lp, P_rai]):
    if lbd is None:
        continue
    print("Setting:", ord[i])
    for alpha in aps:
        print(get_cvar_loss(lbd, train_correct, alpha), "/", get_cvar_loss(lbd, val_correct, alpha), "/", get_cvar_loss(lbd, test_correct, alpha), end=" || ")
    print()
    # worst_group_loss(args, lbd)

print("\n\nWorst Case Group Loss")
for i, lbd in enumerate([P_erm, P_uniform, P_adaboost, P_lp, P_rai]):
    if lbd is None:
        continue
    print("Setting:", ord[i])
    worst_group_loss(args, lbd)