import os
import pandas
import regex as re
import time
from tqdm import tnrange, tqdm_notebook
import tqdm
import numpy
import torch
from scipy.sparse import coo_matrix

from matplotlib import transforms
import networkx as nx
import numpy as np
import pandas as pd
import os
import sys
from glob import glob
import torch
from torch_geometric.nn import GCNConv
import os.path as osp
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import train_test_split_edges
from torch_geometric.loader import DataLoader
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score
import torch_geometric.transforms as T


path = os.getcwd() + '/raw_name'

os.chdir(path)
df = []
def read_text_file(file_path):
    file = open(file_path, 'r')
    for line in file:
        df.append(line)
  
  
# iterate through all file
for file in os.listdir():
    # Check whether file is in text format or not
    if file.endswith(".txt"):
        file_path = f"{path}/{file}"
  
        # call read text file function
        read_text_file(file_path)

lst = []
lst.append("a.b c.dsf.dd")

dn = '(\w\n ,'

x = df[0].split()


pattern1 = r'\w*\s*\w+(?=\.)'
pattern2 = '(?<=\.)\w*\s*\w+$'
ptrn = [pattern1, pattern2]
ptn = re.compile(r'(?<=\.)\w*\s*\w+$')

matches = []
for pat in ptrn:
    matches += re.findall(pat, x[0])

# re.findall('(%s|%s)' % (pattern1,pattern2), x[0] )
re.compile("(%s|%s)" % (pattern1, pattern2)).findall(x[0])

re.split(r"\.", x[0])

count = 0
for i in df:
    count = count + 1
count*2


lst = []
count = 0
for combined in df:
    comb = combined.split()
    
    for sentence in comb:
        lst += re.compile('(%s|%s)' % (pattern1, pattern2)).findall(sentence)
        count = count + 1
        

lst = list(set(lst))


# we have to make a list of list 
splitted_df = []

for i in df:
    comb = i.split()
    for sp in comb:
        splitted_df.append(sp)

matrix = []
for call in splitted_df:
    call_break = re.split(r'\.', call)
    print(call_break)
    ans = []
    for word in lst:
        if word in call_break:
            ans.append(1)
        else:
            ans.append(0)
    
    matrix.append(ans)
    break

count = 0
for i in matrix[0]:
    if(i == 1):
        count = count+1

# check if size is same of not
# size of first call break = 7

matrix = []
with tqdm.tqdm(total=len(splitted_df)) as t:
    for call in splitted_df:
        call_break = re.split(r'\.', call)
        ans = []
        for word in lst:
            if word in call_break:
                ans.append(1)
            else:
                ans.append(0)

        matrix.append(ans)
        t.update(1)

matrix_array = numpy.array([numpy.array(xi) for xi in matrix])

os.chdir("../")
os.getcwd()


device = "cpu"

root = "Dataset/callNetwork"

sys.stdout = open("results - 3 AUC curve.txt", 'w')


class CallNetworks(Dataset):
    def __init__(self, root, tranform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, tranform, pre_transform, pre_filter)
    @property
    def raw_file_names(self):
        files = os.listdir(self.root+"/raw")
        return files
    @property
    def processed_file_names(self):
        processed_files = [f"data_{i}.pt" for i in range(len(self.raw_file_names))]
        return processed_files
    def process(self):
        index = 0
        for raw_path in self.raw_paths:
            print(raw_path)
            temp = list(np.unique(pd.read_csv(raw_path, delimiter=" ").values.reshape((1,-1))))
            G = nx.read_edgelist(raw_path, delimiter=" ",create_using = nx.Graph)
            # print(G.nodes)
            # print(temp)
            tmp_nodes = list(G.nodes)
            for i in temp:
                if i not in tmp_nodes:
                    G.add_node((i))
            x = torch.zeros((len(G.nodes), 5))
            edge_index = torch.from_numpy(pd.read_csv(raw_path, delimiter=" ").values).reshape((2,-1))
            for i, key in enumerate(dict(G.nodes).keys()):
                """
                Difference in Approach 2 and 3
                """
                for j in range(matrix.shape[1]):
                    x[i,j] = matrix[i,j]
                x[i,j+1] = index
            data = Data(x=x, edge_index=edge_index)
            torch.save(data, osp.join(self.processed_dir,f"data_{index}.pt"))
            index += 1
    def len(self):
        return len(self.processed_file_names)
    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f"data_{idx}.pt"))
        return data

transform = T.RandomLinkSplit(num_val=0.05, num_test=0.1,is_undirected=True, add_negative_train_samples=True)
dataset = CallNetworks(root=root)
train_data, val_data, test_data = transform(dataset[0])


class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)#.relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


model = Net(dataset.num_features, 128, 64).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()


def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)

    # We perform a new round of negative sampling for every training epoch:
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat([
        train_data.edge_label,
        train_data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)

    out = model.decode(z, edge_label_index).view(-1)
    loss = criterion(out, edge_label)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())


best_val_auc = final_test_auc = 0
for epoch in range(1, 101):
    loss = train()
    val_auc = test(val_data)
    test_auc = test(test_data)
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        final_test_auc = test_auc
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, '
          f'Test: {test_auc:.4f}')

print(f'Final Test: {final_test_auc:.4f}')

z = model.encode(test_data.x, test_data.edge_index)
final_edge_index = model.decode_all(z)

    