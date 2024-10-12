import os

import scipy
import torch
import dgl
from dgl.data import RedditDataset
from dgl.distributed import partition_graph
import torch.distributed as dist
import time
from contextlib import contextmanager
from ogb.nodeproppred import DglNodePropPredDataset
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from typing import Dict, List, Tuple, Union
from dgl.data.dgl_dataset import DGLDataset
import scipy.sparse as sp

def load_ogb_dataset(name):
    dataset = DglNodePropPredDataset(name=name, root='./dataset/')
    split_idx = dataset.get_idx_split()
    g, label = dataset[0]
    n_node = g.num_nodes()

    node_data = g.ndata
    node_data['label'] = label.view(-1).long()
    node_data['train_mask'] = torch.zeros(n_node, dtype=torch.bool)

    
    node_data['val_mask'] = torch.zeros(n_node, dtype=torch.bool)

    
    node_data['test_mask'] = torch.zeros(n_node, dtype=torch.bool)

    
    node_data['train_mask'][split_idx["train"]] = True
    node_data['val_mask'][split_idx["valid"]] = True
    node_data['test_mask'][split_idx["test"]] = True


    return g


def load_yelp():
    prefix = './dataset/yelp/'

    with open(prefix + 'class_map.json') as f:
        class_map = json.load(f)
    with open(prefix + 'role.json') as f:
        role = json.load(f)

    adj_full = scipy.sparse.load_npz(prefix + 'adj_full.npz')
    feats = np.load(prefix + 'feats.npy')
    n_node = feats.shape[0]

    g = dgl.from_scipy(adj_full)
    node_data = g.ndata

    label = list(class_map.values())
    node_data['label'] = torch.tensor(label)

    node_data['train_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['val_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['test_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['train_mask'][role['tr']] = True
    node_data['val_mask'][role['va']] = True
    node_data['test_mask'][role['te']] = True

    assert torch.all(torch.logical_not(torch.logical_and(node_data['train_mask'], node_data['val_mask'])))
    assert torch.all(torch.logical_not(torch.logical_and(node_data['train_mask'], node_data['test_mask'])))
    assert torch.all(torch.logical_not(torch.logical_and(node_data['val_mask'], node_data['test_mask'])))
    assert torch.all(
        torch.logical_or(torch.logical_or(node_data['train_mask'], node_data['val_mask']), node_data['test_mask']))

    train_feats = feats[node_data['train_mask']]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    feats = scaler.transform(feats)

    node_data['feat'] = torch.tensor(feats, dtype=torch.float)

    return g

# Include the AmazonProducts class definition here

class AmazonProducts(DGLDataset):
    def __init__(self, raw_dir: str=None, force_reload: bool=False, verbose: bool=False):
        _url = 'https://docs.google.com/uc?export=download&id={}&confirm=t'
        super(AmazonProducts, self).__init__(name='amazonProducts', url=_url, raw_dir=raw_dir, force_reload=force_reload, verbose=verbose)
    
    def download(self):
        adj_full_id = '17qhNA8H1IpbkkR-T2BmPQm8QNW5do-aa'
        feats_id = '10SW8lCvAj-kb6ckkfTOC5y0l8XXdtMxj'
        class_map_id = '1LIl4kimLfftj4-7NmValuWyCQE8AaE7P'
        role_id = '1npK9xlmbnjNkV80hK2Q68wTEVOFjnt4K'
        
        if os.path.exists(self.raw_path):  # pragma: no cover
            return

        os.makedirs(self.raw_path, exist_ok=True)

        # 下载 adj_full.npz
        path = download_url(self.url.format(adj_full_id), self.raw_path)
        adj_path = os.path.join(self.raw_path, 'adj_full.npz')
        os.rename(path, adj_path)
        print(f"Downloaded and saved adj_full.npz at {adj_path}")

        # 下载 feats.npy
        path = download_url(self.url.format(feats_id), self.raw_path)
        feats_path = os.path.join(self.raw_path, 'feats.npy')
        os.rename(path, feats_path)
        print(f"Downloaded and saved feats.npy at {feats_path}")

        # 下载 class_map.json
        path = download_url(self.url.format(class_map_id), self.raw_path)
        class_map_path = os.path.join(self.raw_path, 'class_map.json')
        os.rename(path, class_map_path)
        print(f"Downloaded and saved class_map.json at {class_map_path}")

        # 下载 role.json
        path = download_url(self.url.format(role_id), self.raw_path)
        role_path = os.path.join(self.raw_path, 'role.json')
        os.rename(path, role_path)
        print(f"Downloaded and saved role.json at {role_path}")

    def process(self):
        f = np.load(os.path.join(self.raw_path, 'adj_full.npz'))
        # graph
        adj = sp.csr_matrix((f['data'], f['indices'], f['indptr']), f['shape'])
        adj = adj.tocoo()
        self._graph = dgl.from_scipy(adj)
        # features and labels
        amazon_data = np.load(os.path.join(self.raw_path, 'feats.npy'))
        amazon_data = torch.from_numpy(amazon_data).to(torch.float32)
        ys = [-1] * amazon_data.size(0)
        with open(os.path.join(self.raw_path, 'class_map.json')) as f:
            class_map = json.load(f)
            for key, item in class_map.items():
                ys[int(key)] = item
        labels = torch.tensor(ys, dtype=torch.float32)
        # train/val/test indices
        with open(os.path.join(self.raw_path, 'role.json')) as f:
            role = json.load(f)
        train_mask = torch.zeros(amazon_data.size(0), dtype=torch.bool)
        train_mask[torch.tensor(role['tr'])] = True
        val_mask = torch.zeros(amazon_data.size(0), dtype=torch.bool)
        val_mask[torch.tensor(role['va'])] = True
        test_mask = torch.zeros(amazon_data.size(0), dtype=torch.bool)
        test_mask[torch.tensor(role['te'])] = True
        # add all the data to graph 
        self._graph.ndata['train_mask'] = train_mask
        self._graph.ndata['val_mask'] = val_mask
        self._graph.ndata['test_mask'] = test_mask
        self._graph.ndata['feat'] = amazon_data
        self._graph.ndata['label'] = labels
        # reorder graph
        self._graph = dgl.reorder_graph(self._graph, node_permute_algo='rcmk', edge_permute_algo='dst', store_ids=False)  

    @property
    def num_classes(self):
        return 107
    
    @property
    def num_labels(self):
        return self.num_classes
    
    def __getitem__(self, idx):
        assert idx == 0, "AmazonProducts Dataset only has one graph"
        return self._graph
    
    def __len__(self):
        return 1


def load_amazon_products():
    # 实例化 AmazonProducts 数据集对象
    dataset = AmazonProducts(raw_dir='./dataset/')
    
    # 加载图数据
    g = dataset[0]
    
    # 获取节点数量
    n_node = g.num_nodes()
    print(f"AmazonProducts 数据集的节点数量: {n_node}")

    # 获取节点数据
    node_data = g.ndata

    # 确保标签是长整型
    node_data['label'] = node_data['label'].long()

    # 检查训练、验证和测试集的掩码是否正确设置
    print(f"训练节点数: {g.ndata['train_mask'].int().sum().item()}")
    print(f"验证节点数: {g.ndata['val_mask'].int().sum().item()}")
    print(f"测试节点数: {g.ndata['test_mask'].int().sum().item()}")

    # 如果需要对特征进行标准化
    train_feats = node_data['feat'][node_data['train_mask']]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    node_data['feat'] = torch.tensor(scaler.transform(node_data['feat']), dtype=torch.float)

    return g

def load_data(dataset):
    if dataset == 'reddit':
        data = RedditDataset(raw_dir='./dataset/')
        g = data[0]
    elif dataset == 'ogbn-products':
        g = load_ogb_dataset('ogbn-products')
    elif dataset == 'ogbn-papers100m':
        g = load_ogb_dataset('ogbn-papers100M')
    elif dataset == 'yelp':
        g = load_yelp()
    elif dataset == 'amazonProducts':
        g = load_amazon_products()    
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))

    n_feat = g.ndata['feat'].shape[1]
    if g.ndata['label'].dim() == 1:
        n_class = g.ndata['label'].max().item() + 1
    else:
        n_class = g.ndata['label'].shape[1]

    g.edata.clear()
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    return g, n_feat, n_class


def load_partition(args, rank):
    graph_dir = 'partitions/' + args.graph_name + '/'
    part_config = graph_dir + args.graph_name + '.json'

    print('loading partitions')

    subg, node_feat, _, gpb, _, node_type, _ = dgl.distributed.load_partition(part_config, rank)
    node_type = node_type[0]
    node_feat[dgl.NID] = subg.ndata[dgl.NID]
    if 'part_id' in subg.ndata:
        node_feat['part_id'] = subg.ndata['part_id']
    node_feat['inner_node'] = subg.ndata['inner_node'].bool()
    node_feat['label'] = node_feat[node_type + '/label']
    node_feat['feat'] = node_feat[node_type + '/feat']
    node_feat['in_degree'] = node_feat[node_type + '/in_degree']
    node_feat['out_degree'] = node_feat[node_type + '/out_degree']
    node_feat['train_mask'] = node_feat[node_type + '/train_mask'].bool()
    node_feat.pop(node_type + '/label')
    node_feat.pop(node_type + '/feat')
    node_feat.pop(node_type + '/in_degree')
    node_feat.pop(node_type + '/train_mask')
    if not args.inductive:
        node_feat['val_mask'] = node_feat[node_type + '/val_mask'].bool()
        node_feat['test_mask'] = node_feat[node_type + '/test_mask'].bool()
        node_feat.pop(node_type + '/val_mask')
        node_feat.pop(node_type + '/test_mask')
    if args.dataset == 'ogbn-papers100m':
        node_feat.pop(node_type + '/year')
    subg.ndata.clear()
    subg.edata.clear()

    return subg, node_feat, gpb

# 沈练分区
def graph_partition(g, args):
    graph_dir = 'partitions/' + args.graph_name + '/'
    part_config = graph_dir + args.graph_name + '.json'

    # 确保目录存在
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)
    
    if not os.path.exists(part_config):
        with g.local_scope():
            if args.inductive:
                g.ndata.pop('val_mask', None)
                g.ndata.pop('test_mask', None)
            
            # 添加节点入度信息
            g.ndata['in_degree'] = g.in_degrees()
            g.ndata['out_degree'] = g.out_degrees()

            # 检查张量类型并转换为布尔型（如果必要）
            if 'inner_node' in g.ndata:
                g.ndata['inner_node'] = g.ndata['inner_node'].bool()

            # 使用partition_graph进行图分区，尝试启用边平衡
            partition_graph(
                g, args.graph_name, args.n_partitions, graph_dir, 
                part_method=args.partition_method,
                balance_edges=True,  # 尝试启用边平衡
                objtype=args.partition_obj
            )

def get_layer_size(n_feat, n_hidden, n_class, n_layers):
    layer_size = [n_feat]
    layer_size.extend([n_hidden] * (n_layers - 1))
    layer_size.append(n_class)
    return layer_size


def get_boundary(node_dict, gpb):
    rank, size = dist.get_rank(), dist.get_world_size()
    device = 'cuda'
    boundary = [None] * size

    for i in range(1, size):
        left = (rank - i + size) % size
        right = (rank + i) % size
        belong_right = (node_dict['part_id'] == right)
        num_right = belong_right.sum().view(-1)
        if dist.get_backend() == 'gloo':
            num_right = num_right.cpu()
            num_left = torch.tensor([0])
        else:
            num_left = torch.tensor([0], device=device)
        req = dist.isend(num_right, dst=right)
        dist.recv(num_left, src=left)
        start = gpb.partid2nids(right)[0].item()
        v = node_dict[dgl.NID][belong_right] - start
        if dist.get_backend() == 'gloo':
            v = v.cpu()
            u = torch.zeros(num_left, dtype=torch.long)
        else:
            u = torch.zeros(num_left, dtype=torch.long, device=device)
        req.wait()
        req = dist.isend(v, dst=right)
        dist.recv(u, src=left)
        u, _ = torch.sort(u)
        if dist.get_backend() == 'gloo':
            boundary[left] = u.cuda()
        else:
            boundary[left] = u
        req.wait()

    return boundary


def get_one_hops(data, recv_shape, tag=0, dtype=torch.long):

    rank, size = dist.get_rank(), dist.get_world_size()
    msg, res = [None] * size, [None] * size

    for i in range(1, size):
        idx = (rank + i) % size
        key = 'dst%d_tag%d' % (idx, tag)
        if key not in _recv_cpu:
            _send_cpu[key] = torch.zeros_like(data[idx], dtype=dtype, device='cpu', pin_memory=True)
            _recv_cpu[key] = torch.zeros(recv_shape[idx], dtype=dtype, pin_memory=True)
        msg[idx] = _send_cpu[key]
        res[idx] = _recv_cpu[key]

    for i in range(1, size):
        left = (rank - i + size) % size
        right = (rank + i) % size
        msg[right].copy_(data[right])
        req = dist.isend(msg[right], dst=right, tag=tag)
        dist.recv(res[left], src=left, tag=tag)
        res[left] = res[left].cuda(non_blocking=True)
        req.wait()

    return res

def data_transfer(data, recv_shape, backend, dtype=torch.float, tag=0):
    rank, size = dist.get_rank(), dist.get_world_size()
    res = [None] * size

    for i in range(1, size):
        left = (rank - i + size) % size
        if backend == 'gloo':
            res[left] = torch.zeros(torch.Size([recv_shape[left], data[left].shape[1]]), dtype=dtype)
        else:
            res[left] = torch.zeros(torch.Size([recv_shape[left], data[left].shape[1]]), dtype=dtype, device='cuda')

    for i in range(1, size):
        left = (rank - i + size) % size
        right = (rank + i) % size
        if backend == 'gloo':
            req = dist.isend(data[right].cpu(), dst=right, tag=tag)
        else:
            req = dist.isend(data[right], dst=right, tag=tag)
        dist.recv(res[left], src=left, tag=tag)
        res[left] = res[left].cuda()
        req.wait()

    return res

# 通用的传播，用于传播出度等
def data_transfer_degree(data, recv_shape, tag=1, dtype=torch.float):

    rank, size = dist.get_rank(), dist.get_world_size()
    msg, res = [None] * size, [None] * size
    _send_cpu, _recv_cpu = {}, {}
    for i in range(1, size):
        idx = (rank + i) % size
        key = 'dst%d_tag%d' % (idx, tag)
        if key not in _recv_cpu:
            _send_cpu[key] = torch.zeros_like(data[idx], dtype=dtype, device='cpu', pin_memory=True)
            _recv_cpu[key] = torch.zeros(recv_shape[idx], dtype=dtype, pin_memory=True)
        msg[idx] = _send_cpu[key]
        res[idx] = _recv_cpu[key]

    for i in range(1, size):
        left = (rank - i + size) % size
        right = (rank + i) % size
        msg[right].copy_(data[right])
        req = dist.isend(msg[right], dst=right, tag=tag)
        dist.recv(res[left], src=left, tag=tag)
        res[left] = res[left].cuda(non_blocking=True)
        req.wait()

    return res


def merge_feature(feat, recv):
    size = len(recv)
    for i in range(size - 1, 0, -1):
        if recv[i] is None:
            recv[i] = recv[i - 1]
            recv[i - 1] = None
    recv[0] = feat
    return torch.cat(recv)


def inductive_split(g):
    g_train = g.subgraph(g.ndata['train_mask'])
    g_val = g.subgraph(g.ndata['train_mask'] | g.ndata['val_mask'])
    g_test = g
    return g_train, g_val, g_test


def minus_one_tensor(size, device=None):
    if device is not None:
        return torch.zeros(size, dtype=torch.long, device=device) - 1
    else:
        return torch.zeros(size, dtype=torch.long) - 1


def nonzero_idx(x):
    return torch.nonzero(x, as_tuple=True)[0]


def print_memory(s):
    torch.cuda.synchronize()
    print(s + ': current {:.2f}MB, peak {:.2f}MB, reserved {:.2f}MB'.format(
        torch.cuda.memory_allocated() / 1024 / 1024,
        torch.cuda.max_memory_allocated() / 1024 / 1024,
        torch.cuda.memory_reserved() / 1024 / 1024
    ))


@contextmanager
def timer(s):
    rank, size = dist.get_rank(), dist.get_world_size()
    t = time.time()
    yield
    print('(rank %d) running time of %s: %.3f seconds' % (rank, s, time.time() - t))



# '''
# *************************************************
# ***** quantization/dequantization functions *****
# *************************************************
# '''

# # 这个函数的作用是计算输入张量的每行的最小值和最大值，并返回这些最小值和最大值。

# # 让我们来逐步解释这个函数：

# # input: Tensor：这是一个输入张量，其中每行代表一个样本，每列代表样本的特征。
# # -> Tuple[Tensor, Tensor]：函数的返回类型是一个包含两个张量的元组。
# # rmin, rmax = torch.min(input, dim=1)[0], torch.max(input, dim=1)[0]：
# # 这行代码使用 PyTorch 的 min 和 max 函数计算输入张量 input 沿着第一个维度（即行）的最小值和最大值。这两个函数返回一个元组，其中第一个元素是最小值，第二个元素是对应的索引。我们在这里通过索引 [0] 只获取最小值本身，而不关心索引。rmin 存储了每行的最小值，rmax 存储了每行的最大值。
# # return rmin, rmax：最后，将最小值和最大值作为元组返回。
# # 这个函数通常用于归一化或标准化数据，以便在训练神经网络时提高训练效果

# def compute_minmax_params(input: Tensor) -> Tuple[Tensor, Tensor]:
#     rmin, rmax = torch.min(input, dim=1)[0], torch.max(input, dim=1)[0]
#     return rmin, rmax


# # 数据形状变化：
# # 在输入之前，数据张量的形状为 [N, F]，其中 N 是批量大小，F 是特征维度。
# # 在量化之后，数据张量的形状可能会发生变化，变成了 [N/(8/bits)*F] 的形状，其中 (8/bits) 是每个量化数所占的比特位数。
# # 数据值变化：
# # 在输入之前，数据张量的值是原始数据，可以是任意浮点数值。
# # 在量化之后，数据张量的值被量化为整数，取值范围通常在 [0, 2^bits - 1] 之间，具体取值由量化函数和输入数据的范围决定。
# # 整数值的范围根据 bits 参数确定，通常 bits 越大，可表示的数值范围越广，但精度会降低。
# # 缩放因子变化：
# # 在量化过程中，会计算每个节点的缩放因子 scale，用于将原始数据缩放到量化后的整数值范围内。
# # scale 的形状与 rmin 和 rmax 相同，都是 [N]，其中 N 是数据批量大小。
# # 缩放因子 scale 是一个浮点数张量，用于将原始数据缩放到量化后的整数值范围内，以确保量化后的数据保持尽可能接近原始数据的范围和分布。
# # 综上所述，整数量化函数会将输入的浮点数数据张量转换为整数数据张量，并伴随着缩放因子的计算，以保证数据的尽可能准确的量化。

# def integer_quantize(data: Tensor, bits: int, rmin: Tensor, rmax: Tensor, stochastic: bool = True) -> Tuple[Tensor, Tensor]:
#     '''
#     `input`
#         data: shape: [N, F], where N is the batch_size, F is the feature dimension.

#         bits: type: int, quantization bit width.

#         rmin: shape: [N], min value per node, serve as zero point.

#         rmax: shape: [N], max value per node.
#     `return`
#         q_data: [N/(8/bits)*F]

#         scale: [N]
#     '''
#     assert type(bits) == int
#     quant_func = integer_quantizer.pack_single_precision
#     scale = (2 ** bits - 1) / (rmax - rmin)  # shape: [N]
#     q_data = quant_func(data, rmin, rmax, scale.to(data.dtype), bits, stochastic)
#     return q_data, scale

# def integer_dequantize(q_data: Tensor, shape: torch.Size, bits: int, scale: Tensor, rmin: Tensor) -> Tensor:
#     r'''
#     input
#         data: shape: [N/(8/bits)*F], where N is the batch_size, bits is the quantization bits,  F is the feature dimension. (already on device)

#         shape: the tempinal shape of q_data

#         bits: type: int, quantization bit width.

#         scale: shape: [N], quantization scale per node. (already on device)

#         rmin: shape: [N], min value per node, serve as zero point.

#     return
#         data: shape: [N, F], where N is the batch_size, F is the feature dimension.
#     '''
#     N = shape[0]
#     num_features = shape[1]
#     # unpack bit stream
#     assert type(bits) == int
#     dequant_func = integer_quantizer.unpack_single_precision
#     data = dequant_func(q_data, bits, scale, rmin, N, num_features)
#     return data

# @contextmanager
# def timer(s):
#     rank, size = dist.get_rank(), dist.get_world_size()
#     t = time.time()
#     yield
#     print('(rank %d) running time of %s: %.3f seconds' % (rank, s, time.time() - t))

# def message_quantization(input: Tensor, bits: int, stochastic: bool) -> Tuple[Tensor, Tensor, Tensor, torch.Size]:
#     rmin, rmax = compute_minmax_params(input)
#     q_input, q_scale = integer_quantize(input, bits, rmin, rmax, stochastic=stochastic)
#     # transfer with bfloat16
#     if input.dtype == torch.float32:
#         return q_input, q_scale.to(torch.bfloat16), rmin.to(torch.bfloat16), input.shape
#     else:
#         return q_input, q_scale, rmin, input.shape

# def message_dequantization(q_input: Tensor, q_scale: Tensor, rmin: Tensor, input_tempin_shape: torch.Size, bits):
#     if q_scale.dtype == torch.bfloat16:
#         q_scale = q_scale.to(torch.float32)
#         rmin = rmin.to(torch.float32)
#     input = integer_dequantize(q_input, input_tempin_shape, bits, q_scale, rmin)
#     return input.contiguous()




