import torch.nn.functional as F
from module.model import *
from helper.utils import *
import torch.distributed as dist
import time
import copy
from multiprocessing.pool import ThreadPool
from sklearn.metrics import f1_score
import csv
import os
import time as sleep
import uuid
from datetime import datetime
from config import *

def save_results_to_csv(unique_id, timestamp, dataset, model, method, partition, partition_method, epoch, computation, communication, reduce, accuracy, bits, 
                        stale, stale_threshold, group_num, wait_reduce, debug, count_time, transfer_debug, enable_pipeline,
                        neu_time, forward_time, backward_time, qt_time, dq_time, pack_time, unpack_time,
                        true_comm_time, forward_comm_time, backward_comm_time, transfer_num, forward_transfer_num, backward_transfer_num, qt_on_comp, adaptive, csv_file='results.csv'):
    """
    保存训练结果到CSV文件

    参数:
    unique_id (str): 唯一标识符
    timestamp (str): 时间戳
    dataset (str): 数据集名称
    model (str): 使用的模型
    method (str): 训练方式
    partition (int): 分区数量
    partition_method (str): 分区方法
    epoch (int): 当前Epoch
    computation (float): 计算时间
    communication (float): 通信时间
    reduce (float): 归约时间
    accuracy (float): 准确率
    bits (int): 数据传输的位数
    stale (bool): 是否启用过期数据
    stale_threshold (float): 过期数据阈值
    group_num (int): 组的数量
    wait_reduce (bool): 是否等待归约
    debug (bool): 是否启用调试模式
    count_time (bool): 是否计算时间
    transfer_debug (bool): 是否启用传输调试
    enable_pipeline (bool): 是否启用流水线
    neu_time (float): 神经网络时间
    forward_time (float): 前向传播时间
    backward_time (float): 反向传播时间
    qt_time (float): 量化时间
    dq_time (float): 解量化时间
    pack_time (float): 打包时间
    unpack_time (float): 解包时间
    true_comm_time (float): 真正通信时间
    forward_comm_time (float): 前向传播通信时间
    backward_comm_time (float): 反向传播通信时间
    transfer_num (int): 数据传输总数
    forward_transfer_num (int): 前向传播传输次数
    backward_transfer_num (int): 反向传播传输次数
    csv_file (str): CSV文件路径
    qt_on_comp (bool): 在计算或者通讯时进行量化
    adaptive (bool): 是否自行自适应位宽调整
    """
    # 检查CSV文件是否存在，不存在则创建并写入表头
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['timestamp', 'unique_id', 'dataset', 'model', 'method', 'partition', 'partition-method', 'bits', 'epoch', 
                             'computation', 'communication', 'reduce', 'accuracy', 'stale', 'stale_threshold', 'group_num', 
                             'wait_reduce', 'debug', 'count_time', 'transfer_debug', 'enable_pipeline',
                             'neu_time', 'forward_time', 'backward_time', 'qt_time', 'dq_time', 'pack_time', 'unpack_time',
                             'true_comm_time', 'forward_comm_time', 'backward_comm_time', 'transfer_num', 'forward_transfer_num', 'backward_transfer_num', 'qt_on_comp', 'adaptive'])
        writer.writerow([timestamp, unique_id, dataset, model, method, partition, partition_method, bits, epoch, computation, 
                         communication, reduce, accuracy, stale, stale_threshold, group_num, wait_reduce, debug, count_time, 
                         transfer_debug, enable_pipeline, neu_time, forward_time, backward_time, qt_time, dq_time, pack_time, unpack_time,
                         true_comm_time, forward_comm_time, backward_comm_time, transfer_num, forward_transfer_num, backward_transfer_num, qt_on_comp, adaptive])


def collect_out_degree(node_dict, boundary, args):
    rank, size = dist.get_rank(), dist.get_world_size()
    out_deg = node_dict['out_degree']
    send_info = []
    for i, b in enumerate(boundary):
        if i == rank:
            send_info.append(None)
            continue
        else:
            send_info.append(out_deg[b])
    recv_shape = []
    for i in range(size):
        if i == rank:
            recv_shape.append(None)
            continue
        else:
            s = (node_dict['part_id'] == i).int().sum()
            recv_shape.append(torch.Size([s]))
    recv_out_deg = data_transfer_degree(send_info, recv_shape, tag=1, dtype=torch.long)
    return merge_feature(out_deg, recv_out_deg)


def calc_acc(logits, labels):
    if labels.dim() == 1:
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() / labels.shape[0]
    else:
        return f1_score(labels, logits > 0, average='micro')



@torch.no_grad()
def evaluate_induc(name, model, g, mode, epoch, rank, result_file_name=None):
    """
    mode: 'val' or 'test'
    """
    model.eval()
    model.cpu()
    feat, labels = g.ndata['feat'], g.ndata['label']
    mask = g.ndata[mode + '_mask']
    logits = model(g, feat, BITS, epoch, rank)
    logits = logits[mask]
    labels = labels[mask]
    acc = calc_acc(logits, labels)
    buf = "{:s} | Accuracy {:.2%}".format(name, acc)
    print(buf)
    return model, acc


@torch.no_grad()
def evaluate_trans(name, model, g, epoch, rank, result_file_name=None):
    model.eval()
    model.cpu()
    feat, labels = g.ndata['feat'], g.ndata['label']
    val_mask, test_mask = g.ndata['val_mask'], g.ndata['test_mask']
    logits = model(g, feat, BITS, epoch, rank)
    val_logits, test_logits = logits[val_mask], logits[test_mask]
    val_labels, test_labels = labels[val_mask], labels[test_mask]
    val_acc = calc_acc(val_logits, val_labels)
    test_acc = calc_acc(test_logits, test_labels)
    buf = "{:s} | Validation Accuracy {:.2%} | Test Accuracy {:.2%}".format(name, val_acc, test_acc)
    print(buf)
    return model, test_acc




# def move_to_cuda(graph, part, node_dict):

#     for key in node_dict.keys():
#         node_dict[key] = node_dict[key].cuda()
#     graph = graph.int().to(torch.device('cuda'))
#     part = part.int().to(torch.device('cuda'))

#     return graph, part, node_dict

def move_to_cuda(graph, part, in_graph, out_graph, node_dict):
    rank, size = dist.get_rank(), dist.get_world_size()
    for key in node_dict.keys():
        node_dict[key] = node_dict[key].cuda()
    graph = graph.int().to(torch.device('cuda'))
    part = part.int().to(torch.device('cuda'))
    in_graph = in_graph.int().to(torch.device('cuda'))
    out_graph = out_graph.int().to(torch.device('cuda'))
    return graph, part, in_graph, out_graph, node_dict 


def get_pos(node_dict, gpb):
    pos = []
    rank, size = dist.get_rank(), dist.get_world_size()
    for i in range(size):
        if i == rank:
            pos.append(None)
        else:
            part_size = gpb.partid2nids(i).shape[0]
            start = gpb.partid2nids(i)[0].item()
            p = minus_one_tensor(part_size, 'cuda')
            in_idx = nonzero_idx(node_dict['part_id'] == i)
            out_idx = node_dict[dgl.NID][in_idx] - start
            p[out_idx] = in_idx
            pos.append(p)
    return pos


def get_recv_shape(node_dict):
    rank, size = dist.get_rank(), dist.get_world_size()
    recv_shape = []
    for i in range(size):
        if i == rank:
            recv_shape.append(None)
        else:
            t = (node_dict['part_id'] == i).int().sum().item()
            recv_shape.append(t)
    return recv_shape

# 在分布式图神经网络训练中，这段代码的主要作用是创建一个只包含内部节点及其相互连接的子图
def create_inner_graph(graph, node_dict):
    u, v = graph.edges()
    sel = torch.logical_and(node_dict['inner_node'].bool()[u], node_dict['inner_node'].bool()[v])
    u, v = u[sel], v[sel]
    return dgl.graph((u, v))

# 帮助在分布式图神经网络训练中有效组织节点和邻居节点的信息，优化数据结构和计算过程
def order_graph(part, graph, gpb, node_dict, pos):
    rank, size = dist.get_rank(), dist.get_world_size()
    one_hops = []
    for i in range(size):
        if i == rank:
            one_hops.append(None)
            continue
        start = gpb.partid2nids(i)[0].item()
        nodes = node_dict[dgl.NID][node_dict['part_id'] == i] - start
        nodes, _ = torch.sort(nodes)
        one_hops.append(nodes)
    return one_hops, construct(part, graph, pos, one_hops)

# 重新排列图中的节点，使得训练节点排在前面
def move_train_first(graph, node_dict, boundary):
    train_mask = node_dict['train_mask']
    num_train = torch.count_nonzero(train_mask).item()
    num_tot = graph.num_nodes('_V')

    new_id = torch.zeros(num_tot, dtype=torch.int, device='cuda')
    new_id[train_mask] = torch.arange(num_train, dtype=torch.int, device='cuda')
    new_id[torch.logical_not(train_mask)] = torch.arange(num_train, num_tot, dtype=torch.int, device='cuda')

    u, v = graph.edges()
    u[u < num_tot] = new_id[u[u < num_tot].long()]
    v = new_id[v.long()]
    graph = dgl.heterograph({('_U', '_E', '_V'): (u, v)})

    for key in node_dict:
        node_dict[key][new_id.long()] = node_dict[key][0:num_tot].clone()

    for i in range(len(boundary)):
        if boundary[i] is not None:
            boundary[i] = new_id[boundary[i]].long()

    return graph, node_dict, boundary




def precompute(graph, node_dict, boundary, recv_shape, args):
    rank, size = dist.get_rank(), dist.get_world_size()
    in_size = node_dict['inner_node'].bool().sum().item()  # 获取实际的数值
    feat = node_dict['feat']
    send_info = []
    for i, b in enumerate(boundary):
        if i == rank:
            send_info.append(None)
        else:
            send_info.append(feat[b])

    recv_feat = data_transfer(send_info, recv_shape, args.backend, dtype=torch.float)
    
    if args.model == 'graphsage':
        with graph.local_scope():
            graph.nodes['_U'].data['h'] = merge_feature(feat, recv_feat)
            graph['_E'].update_all(fn.copy_u('h', 'm'), fn.sum(msg='m', out='h'), etype='_E')
            # graph['_E'].update_all(fn.copy_u('h', 'm'), fn.mean(msg='m', out='h'), etype='_E')
            mean_feat = graph.nodes['_V'].data['h'] / node_dict['in_degree'][0:in_size].unsqueeze(1)
        return torch.cat([feat, mean_feat[:in_size]], dim=1)
    
    elif args.model == 'gnn':
        with graph.local_scope():
            h = merge_feature(feat, recv_feat)
            graph.nodes['_U'].data['h'] = h
            graph['_E'].update_all(fn.copy_u('h', 'm'), fn.sum(msg='m', out='h'), etype='_E')
            sum_feat = graph.nodes['_V'].data['h']
            combined_feat = torch.cat([feat, sum_feat[:in_size]], dim=1)
        return combined_feat
    
    elif args.model == 'gcn':
        in_norm = torch.sqrt(node_dict['in_degree'])
        out_norm = torch.sqrt(node_dict['out_degree'])
        with graph.local_scope():
            graph.nodes['_U'].data['h'] = merge_feature(feat, recv_feat)
            graph.nodes['_U'].data['h'] /= out_norm.unsqueeze(-1)
            graph['_E'].update_all(fn.copy_u(u='h', out='m'),
                                   fn.sum(msg='m', out='h'),
                                   etype='_E')
            return graph.nodes['_V'].data['h'] / in_norm.unsqueeze(-1)    
    elif args.model == 'gat':
        return merge_feature(feat, recv_feat)
    else:
        raise Exception("Unsupported model type")

def construct_graph(part, graph, pos, one_hops):
    rank, size = dist.get_rank(), dist.get_world_size()
    tot = part.num_nodes()
    u, v = part.edges()
    u_list, v_list = [u], [v]
    for i in range(size):
        if i == rank:
            continue
        else:
            u = one_hops[i]
            if u.shape[0] == 0:
                continue
            u = pos[i][u]
            u_ = torch.repeat_interleave(graph.out_degrees(u.int()).long()) + tot
            tot += u.shape[0]
            _, v = graph.out_edges(u.int())
            u_list.append(u_.int())
            v_list.append(v)
    u = torch.cat(u_list)
    v = torch.cat(v_list)
    g = dgl.heterograph({('_U', '_E', '_V'): (u, v)})

    if g.num_nodes('_U') < tot:
        g.add_nodes(tot - g.num_nodes('_U'), ntype='_U')

    return g


def create_model(layer_size, args):
    if args.model == 'graphsage':
        return GraphSAGE(layer_size, F.relu, use_pp=args.use_pp, norm=args.norm, dropout=args.dropout,
                         n_linear=args.n_linear, train_size=args.n_train)
    elif args.model == 'gnn':
        return CustomGNN(layer_size, F.relu, use_pp=args.use_pp, norm=args.norm, dropout=args.dropout,
                        n_linear=args.n_linear, train_size=args.n_train)
    elif args.model == 'gcn':
        return GCN(layer_size, F.relu, use_pp=args.use_pp, norm=args.norm, dropout=args.dropout,
                n_linear=args.n_linear, train_size=args.n_train)
    elif args.model == 'gat':
        return GAT(layer_size, F.relu, use_pp=args.use_pp, heads=args.heads, norm=args.norm, dropout=args.dropout, n_linear=args.n_linear, train_size=args.n_train)

    else:
        raise NotImplementedError


def reduce_hook(rank, param, name, n_train):
    def fn(grad):
        ctx.reducer.reduce(rank, param, name, grad, n_train)
    return fn


def construct(part, graph, pos, one_hops):
    rank, size = dist.get_rank(), dist.get_world_size()
    tot = part.num_nodes()
    u, v = part.edges()
    u_list, v_list = [u], [v]
    for i in range(size):
        if i == rank:
            continue
        else:
            u = one_hops[i]
            if u.shape[0] == 0:
                continue
            u = pos[i][u]
            u_ = torch.repeat_interleave(graph.out_degrees(u.int()).long()) + tot
            tot += u.shape[0]
            _, v = graph.out_edges(u.int())
            u_list.append(u_.int())
            v_list.append(v)
    u = torch.cat(u_list)
    v = torch.cat(v_list)
    g = dgl.heterograph({('_U', '_E', '_V'): (u, v)})
    if g.num_nodes('_U') < tot:
        g.add_nodes(tot - g.num_nodes('_U'), ntype='_U')
    return g

def construct_feat(num, feat, pos, one_hops):
    rank, size = dist.get_rank(), dist.get_world_size()
    res = [feat[0:num]]
    for i in range(size):
        if i == rank:
            continue
        else:
            u = one_hops[i]
            if u.shape[0] == 0:
                continue
            u = pos[i][u]
            res.append(feat[u])

    return torch.cat(res)

def get_in_out_graph(graph, node_dict):
    in_graph = dgl.node_subgraph(graph, node_dict['inner_node'].bool())
    in_graph.ndata.clear()
    in_graph.edata.clear()

    out_graph = graph.clone()
    out_graph.ndata.clear()
    out_graph.edata.clear()
    in_nodes = torch.arange(in_graph.num_nodes())
    out_graph.remove_edges(out_graph.out_edges(in_nodes, form='eid'))
    return in_graph, out_graph

def construct_out_norm(num, norm, pos, one_hops):
    rank, size = dist.get_rank(), dist.get_world_size()
    out_norm_list = [norm[0:num]]
    for i in range(size):
        if i == rank:
            continue
        else:
            out_norm_list.append(norm[pos[i][one_hops[i]]])
    return torch.cat(out_norm_list)



def run(graph, node_dict, gpb, args):
    
    rank, size = dist.get_rank(), dist.get_world_size()
    # if args.model == 'gcn' or args.model == 'gat':
    in_graph, out_graph = get_in_out_graph(graph, node_dict)
        # in_graph, out_graph = move_to_cuda(graph, in_graph, out_graph)
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)
    # 在评估阶段准备验证集和测试集数据
    if rank == 0 and args.eval:
        full_g, n_feat, n_class = load_data(args.dataset)
        if args.inductive:
            _, val_g, test_g = inductive_split(full_g)
        else:
            val_g, test_g = full_g.clone(), full_g.clone()
        del full_g

    # 创建一个只包含内部节点的子图
    part = create_inner_graph(graph.clone(), node_dict)
    # 内部节点的数量
    num_in = node_dict['inner_node'].bool().sum().item()
    part.ndata.clear()
    part.edata.clear()
    # 打印进程信息
    print(f'Process {rank} has {graph.num_nodes()} nodes, {graph.num_edges()} edges '
          f'{part.num_nodes()} inner nodes, and {part.num_edges()} inner edges.')
    
    graph, part, in_graph, out_graph, node_dict = move_to_cuda(graph, part, in_graph, out_graph, node_dict)
    # 获取边界节点
    # 与BNS-GCN中的boundary是一致的

    boundary = get_boundary(node_dict, gpb)
    # 计算层大小

    layer_size = get_layer_size(args.n_feat, args.n_hidden, args.n_class, args.n_layers)
    # 计算每个节点在全局分区的位置
    pos = get_pos(node_dict, gpb)
    # 帮助在分布式图神经网络训练中有效组织节点和邻居节点的信息，优化数据结构和计算过程
    one_hops, graph = order_graph(part, graph, gpb, node_dict, pos)      
    
    in_deg = node_dict['in_degree']
    # 重新排列图中的节点，使得训练节点排在前面
    graph, node_dict, boundary = move_train_first(graph, node_dict, boundary)
    # 一个一维数组，记录当前进程，需要与i进程的多少个节点进行通信
    # 类比BNS-GCN中的RECV-Shape，但不需要采样
    recv_shape = get_recv_shape(node_dict) 

    # one_hops = get_one_hops(boundary, recv_shape, dtype=torch.long)
    # 相比BNS-GCN，多了recv_shape、corr_feat、corr_grad、corr_momentum、pileline，少了send_size和recv_size(需采样)
    # 还提前计算了boundary
    ctx.buffer.init_buffer(num_in, graph.num_nodes('_U'), boundary, recv_shape, layer_size[:args.n_layers-args.n_linear],
                           use_pp=args.use_pp, backend=args.backend, pipeline=args.enable_pipeline, bits=BITS)
    if args.model == 'gcn':
        node_dict['out_degree'] = collect_out_degree(node_dict, boundary, args)
    if args.use_pp:
        node_dict['feat'] = precompute(graph, node_dict, boundary, recv_shape, args)
    
    # 
    labels = node_dict['label'][node_dict['train_mask']]
    train_mask = node_dict['train_mask']
    # 训练集的实际大小
    part_train = train_mask.int().sum().item()
    
    #


    del boundary
    del part
    # del pos

    # 生成唯一标识符和时间戳，用于记录
    unique_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 设置随机种子
    torch.manual_seed(args.seed)

    model = create_model(layer_size, args)
    model.cuda()
    
    ctx.reducer.init(model)

    # '''
    # 注册梯度钩子:
    # 遍历模型的所有参数，为每个参数注册一个梯度钩子(reduce_hook)。这个钩子负责在反向传播过程中拦截梯度，并执行分布式规约操作，
    # 以确保所有进程中参数的梯度保持同步。这是实现分布式梯度下降的关键机制。
    # '''
    for i, (name, param) in enumerate(model.named_parameters()):
        param.register_hook(reduce_hook(rank, param, name, args.n_train))

    # '''
    # 初始化模型性能记录器:
    # best_model, best_acc = None, 0初始化最佳模型和最高准确率的变量，用于记录训练过程中的最优模型状态和性能。
    # '''
    best_model, best_acc = None, 0

    if args.dataset == 'yelp':
        loss_fcn = torch.nn.BCEWithLogitsLoss(reduction='sum')
    else:
        loss_fcn = torch.nn.CrossEntropyLoss(reduction='sum')
    
    # 初始化优化器
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    # '''
    # 训练时间和通信时间的跟踪列表:
    # 初始化几个列表(train_dur, comm_dur, reduce_dur, barr_dur)用于跟踪训练、通信、规约和屏障等待的时间，帮助分析训练过程中的性能瓶颈
    # '''
    # 初始化存储时间的列表
    train_dur, reduce_dur, comm_dur = [], [], []
    # true_comm_dur = []
    # forward_comm_dur, backward_comm_dur = [], []
    # qt_dur, dq_dur, pack_dur, unpack_dur = [], [], [], []
    neu_dur, forward_dur, backward_dur = [], [], []

    torch.cuda.reset_peak_memory_stats()
    thread = None
    pool = ThreadPool(processes=1)

    feat = node_dict['feat']

    node_dict.pop('train_mask')
    node_dict.pop('inner_node')
    node_dict.pop('part_id')
    node_dict.pop(dgl.NID)
    # gcn需要out_norm和归一化
    if args.model == 'gcn':
        in_norm = torch.sqrt(node_dict['in_degree'])
        out_norm = torch.sqrt(node_dict['out_degree'])

    if not args.eval:
        node_dict.pop('val_mask')
        node_dict.pop('test_mask')
    if(DEBUG):
        print("进入训练")
    # 训练循环
    for epoch in range(args.n_epochs):
        t0 = time.time()
        
        # if args.model == 'gcn' or args.model == 'gat':
        #     one_hops = get_one_hops(boundary, recv_shape, dtype=torch.long)
        #     graph = construct_graph(in_graph, out_graph, pos, one_hops)


        # 设置模型为训练模式
        model.train()
        
        # 只能使用graphsage
        if args.model == 'graphsage':
            # 模型训练
            # BNS-GCN中用的是动态图g
            logits = model(graph, feat, BITS, epoch, rank, in_deg)
        elif args.model == 'gnn':
            # 模型训练
            logits = model(graph, feat, BITS, epoch, rank, in_deg)
            # if rank == 0:
            #     print(logits)
        elif args.model == 'gcn':
            # 模型训练
            # BNS-GCN中用的是动态图g
            # out_norm_ = construct_out_norm(graph.num_nodes('_V'), out_norm, pos, one_hops)
            logits = model(graph, feat, BITS, epoch, rank, in_norm, out_norm)
        elif args.model == 'gat':
            logits = model(graph, construct_feat(graph.num_nodes('_V'), feat, pos, one_hops), BITS, epoch, rank)
        else:
            raise Exception
        # 在归纳模式下，使用所有节点的预测结果计算损失；否则，仅使用训练节点的预测结果计算损失
        if args.inductive:
            loss = loss_fcn(logits, labels)
        else:
            loss = loss_fcn(logits[train_mask], labels)
        # 清理logits以释放显存
        del logits
        # 梯度清零
        if(DEBUG):
            print("梯度清零")
        optimizer.zero_grad(set_to_none=True)
        # 反向传播：计算梯度
        # 在反向传播中，会利用到feat_tansfer注册的梯度钩子
        # 还会利用到reduce_hook
        # 加了一句！retain_graph = True
        with ctx.neu_timer.timer(f"backward_{epoch}_"):
            with ctx.backward_timer.timer(f"backward_{epoch}_"):
                loss.backward()
        # 
        
        # 把迭代轮次加一
        ctx.buffer.next_epoch()
        if(DEBUG):
            print("同步梯度")        
        # 同步梯度,并计算reduce time
        # 实际上是通讯和计算结束，剩下reduce还没结束的时间，因此需要在
        pre_reduce = time.time()
        # 里面要等待
        # 加了barrier之后，时间加了很多
        
        torch.cuda.current_stream().synchronize()
        ctx.reducer.synchronize(epoch, rank)
        reduce_time = time.time() - pre_reduce
        if(DEBUG):
            print("优化")
        # 执行一次优化步骤,通过梯度下降法来更新参数的值
        # 根据规约后的梯度计算参数
        optimizer.step()

        if(DEBUG):
            print("打印时间")
        # 打印时间
        if epoch >= 5 and epoch % args.log_every != 0:
            train_dur.append(time.time() - t0)
            # 在适当的代码部分记录时间
            comm_dur.append(ctx.comm_timer.tot_time())
            reduce_dur.append(reduce_time)
            # if not STALE:
            #     true_comm_dur.append(ctx.true_comm_timer.tot_time())
            #     forward_comm_dur.append(ctx.forward_comm_timer.tot_time())
            #     backward_comm_dur.append(ctx.backward_comm_timer.tot_time())
            #     qt_dur.append(ctx.qt_timer.tot_time())
            #     dq_dur.append(ctx.dq_timer.tot_time())
            #     pack_dur.append(ctx.pack_timer.tot_time())
            #     unpack_dur.append(ctx.unpack_timer.tot_time())

            neu_dur.append(ctx.neu_timer.tot_time())
            forward_dur.append(ctx.forward_timer.tot_time())
            backward_dur.append(ctx.backward_timer.tot_time())


        if (epoch + 1) % 10 == 0:
            avg_train_time = np.mean(train_dur)
            avg_comm_time = np.mean(comm_dur)
            avg_reduce_time = np.mean(reduce_dur)
            avg_comp_time = avg_train_time - avg_comm_time - avg_reduce_time
            train_loss = loss.item() / part_train
            # print("Process {:03d} | Epoch {:05d} | Time(s) {:.4f} | Comm(s) {:.4f} | Reduce(s) {:.4f} | Loss {:.4f}".format(
            #       rank, epoch, np.mean(train_dur), np.mean(comm_dur), np.mean(reduce_dur), loss.item() / part_train))
            print("Process {:03d} | Epoch {:05d} | Time(s) {:.4f} | Comm(s) {:.4f} | Reduce(s) {:.4f} | Loss {:.4f}".format(
                  rank, epoch, avg_train_time, avg_comm_time, avg_reduce_time, train_loss))
        
        # 每个epoch都会重新计算timer
        # 
        ctx.comm_timer.clear()
        ctx.neu_timer.clear()
        ctx.forward_timer.clear()
        ctx.backward_timer.clear()
        # if not STALE:
        #     ctx.true_comm_timer.clear()
        #     ctx.forward_comm_timer.clear()
        #     ctx.backward_comm_timer.clear()
        #     ctx.qt_timer.clear()
        #     ctx.dq_timer.clear()
        #     ctx.pack_timer.clear()
        #     ctx.unpack_timer.clear()



        del loss
        if(DEBUG):
            print("模型评估")        
        # 模型评估
        if rank == 0 and args.eval and (epoch + 1) % args.log_every == 0:
            if thread is not None:
                model_copy, val_acc = thread.get()
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_model = model_copy
            model_copy = copy.deepcopy(model)
            # 计算时间，用于保存至csv
            # np.mean用于求均值
            avg_train_time = np.mean(train_dur)
            avg_comm_time = np.mean(comm_dur)
            avg_reduce_time = np.mean(reduce_dur)
            avg_comp_time = avg_train_time - avg_comm_time - avg_reduce_time
            avg_comm_time = np.mean(comm_dur)
            # avg_true_comm_time = np.mean(true_comm_dur)
            # avg_forward_comm_time = np.mean(forward_comm_dur)
            # avg_backward_comm_time = np.mean(backward_comm_dur)
            avg_true_comm_time = ctx.true_comm_timer.avg_time(size)
            avg_forward_comm_time = ctx.forward_comm_timer.avg_time(size)
            avg_backward_comm_time = ctx.backward_comm_timer.avg_time(size)
            if Count_Forward_Time:
                print(f"true_comm time : {avg_true_comm_time}")
                print(f"forward time : {avg_forward_comm_time}")
                print(f"backward time : {avg_backward_comm_time}")
            # avg_qt_time = np.mean(qt_dur)
            # avg_dq_time = np.mean(dq_dur)
            # avg_pack_time = np.mean(pack_dur)
            # avg_unpack_time = np.mean(unpack_dur)
            avg_qt_time = ctx.qt_timer.avg_time(size)
            avg_dq_time = ctx.dq_timer.avg_time(size)
            avg_pack_time = ctx.pack_timer.avg_time(size)
            avg_unpack_time = ctx.unpack_timer.avg_time(size)
            if Count_Forward_Time:
                print(f"QT time: {avg_qt_time}")
                print(f"DQ time: {avg_dq_time}")
                print(f"Pack time: {avg_pack_time}")
                print(f"Unpack time: {avg_unpack_time}")
            avg_neu_time = np.mean(neu_dur)

            avg_forward_time = np.mean(forward_dur)
            avg_backward_time = np.mean(backward_dur)
            if Count_Forward_Time:
                print(f"Neural network time: {avg_neu_time}")
                print(f"Forward propagation time: {avg_forward_time}")
                print(f"Backward propagation time: {avg_backward_time}")
            acc = 0

            # 获取通信次数
            transfer_num = ctx.true_comm_timer.get_transfer_num()
            forward_transfer_num = ctx.forward_comm_timer.get_transfer_num()
            backward_transfer_num = ctx.backward_comm_timer.get_transfer_num()
            
            
            if not args.inductive:
                thread = pool.apply_async(evaluate_trans, args=('Epoch %05d' % epoch, model_copy,
                                                                val_g, epoch, rank))
                if WRITE_CSV:
                    try:
                        result = thread.get()
                        acc = result[1]
                        print("Function returned:", acc)
                    except Exception as e:
                        print("An error occurred:", e)                
            else:
                thread = pool.apply_async(evaluate_induc, args=('Epoch %05d' % epoch, model_copy,
                                                                val_g, 'val', epoch, rank))
                if WRITE_CSV:
                    try:
                        result = thread.get()
                        acc = result[1]
                        print("Function returned:", acc)
                    except Exception as e:
                        print("An error occurred:", e)  
            if WRITE_CSV:   
                # 确定训练方式
                if FORWARD_QUANTIZATION and BACKWARD_QUANTIZATION:
                    method = 'full_quantization'
                elif FORWARD_QUANTIZATION and not BACKWARD_QUANTIZATION:
                    method = 'forward_quantization'
                elif not FORWARD_QUANTIZATION and BACKWARD_QUANTIZATION:
                    method = 'backward_quantization'
                else:
                    method = 'no_quantization'
                # 保存到csv文件 
                # 记得在这增加内容
                save_results_to_csv(unique_id, timestamp, args.dataset, args.model, method, args.n_partitions, args.partition_method, epoch + 1, avg_comp_time, avg_comm_time, 
                            avg_reduce_time, acc, BITS, STALE, STALE_THRESHOLD, GROUP_NUM, WAIT_REDUCE, DEBUG, Count_Time, TRANFER_DEBUG, args.enable_pipeline,
                            avg_neu_time, avg_forward_time, avg_backward_time, avg_qt_time, avg_dq_time, avg_pack_time, avg_unpack_time,
                            avg_true_comm_time, avg_forward_comm_time, avg_backward_comm_time, transfer_num, forward_transfer_num, backward_transfer_num, QT_ON_COMP, ADAPTIVE, CSV_NAME)

    
    if(DEBUG):
        print("最佳模型跟踪")
    # 最佳模型跟踪
    if args.eval and rank == 0:
        if thread is not None:
            model_copy, val_acc = thread.get()
            if val_acc > best_acc:
                best_acc = val_acc
                best_model = model_copy
                
        # 检查并创建 model 文件夹
        model_dir = "model"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        file_name = f"{model_dir}/{args.graph_name}_final_fq{FORWARD_QUANTIZATION}_bq{BACKWARD_QUANTIZATION}_bits{BITS}_stale{STALE}_qt{QT_ON_COMP}_stale_thr{STALE_THRESHOLD}.pth.tar"
        torch.save(best_model.state_dict(), file_name)
        print('model saved')
        print("Validation accuracy {:.2%}".format(best_acc))
        # _, acc = evaluate_induc('Test Result', best_model, test_g, epoch, rank, 'test')


def check_parser(args):
    if args.norm == 'none':
        args.norm = None


def init_processes(rank, size, args):
    """ Initialize the distributed environment. """
    # gloo
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = '%d' % args.port
    # 设置时间
    # os.environ['GLOO_MESSAGING_TIMEOUT'] = '600'
    
    # 
    dist.init_process_group(args.backend, rank=rank, world_size=size)
    rank, size = dist.get_rank(), dist.get_world_size()
    check_parser(args)
    g, node_dict, gpb = load_partition(args, rank)
    run(g, node_dict, gpb, args)
