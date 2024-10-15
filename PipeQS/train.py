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

def create_inner_graph(graph, node_dict):
    u, v = graph.edges()
    sel = torch.logical_and(node_dict['inner_node'].bool()[u], node_dict['inner_node'].bool()[v])
    u, v = u[sel], v[sel]
    return dgl.graph((u, v))

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
    in_size = node_dict['inner_node'].bool().sum().item()  
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
    in_graph, out_graph = get_in_out_graph(graph, node_dict)
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)
    if rank == 0 and args.eval:
        full_g, n_feat, n_class = load_data(args.dataset)
        if args.inductive:
            _, val_g, test_g = inductive_split(full_g)
        else:
            val_g, test_g = full_g.clone(), full_g.clone()
        del full_g
    part = create_inner_graph(graph.clone(), node_dict)
    num_in = node_dict['inner_node'].bool().sum().item()
    part.ndata.clear()
    part.edata.clear()
    print(f'Process {rank} has {graph.num_nodes()} nodes, {graph.num_edges()} edges '
          f'{part.num_nodes()} inner nodes, and {part.num_edges()} inner edges.')
    
    graph, part, in_graph, out_graph, node_dict = move_to_cuda(graph, part, in_graph, out_graph, node_dict)
    boundary = get_boundary(node_dict, gpb)
    layer_size = get_layer_size(args.n_feat, args.n_hidden, args.n_class, args.n_layers)
    pos = get_pos(node_dict, gpb)
    one_hops, graph = order_graph(part, graph, gpb, node_dict, pos)      
    in_deg = node_dict['in_degree']
    graph, node_dict, boundary = move_train_first(graph, node_dict, boundary)
    recv_shape = get_recv_shape(node_dict) 
    ctx.buffer.init_buffer(num_in, graph.num_nodes('_U'), boundary, recv_shape, layer_size[:args.n_layers-args.n_linear],
                           use_pp=args.use_pp, backend=args.backend, pipeline=args.enable_pipeline, bits=BITS)
    if args.model == 'gcn':
        node_dict['out_degree'] = collect_out_degree(node_dict, boundary, args)
    if args.use_pp:
        node_dict['feat'] = precompute(graph, node_dict, boundary, recv_shape, args)
    
    labels = node_dict['label'][node_dict['train_mask']]
    train_mask = node_dict['train_mask']
    part_train = train_mask.int().sum().item()
    del boundary
    del part
    unique_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    torch.manual_seed(args.seed)
    model = create_model(layer_size, args)
    model.cuda()
    ctx.reducer.init(model)
    for i, (name, param) in enumerate(model.named_parameters()):
        param.register_hook(reduce_hook(rank, param, name, args.n_train))
    best_model, best_acc = None, 0

    if args.dataset == 'yelp':
        loss_fcn = torch.nn.BCEWithLogitsLoss(reduction='sum')
    else:
        loss_fcn = torch.nn.CrossEntropyLoss(reduction='sum')
    
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    train_dur, reduce_dur, comm_dur = [], [], []
    neu_dur, forward_dur, backward_dur = [], [], []

    torch.cuda.reset_peak_memory_stats()
    thread = None
    pool = ThreadPool(processes=1)

    feat = node_dict['feat']

    node_dict.pop('train_mask')
    node_dict.pop('inner_node')
    node_dict.pop('part_id')
    node_dict.pop(dgl.NID)
    if args.model == 'gcn':
        in_norm = torch.sqrt(node_dict['in_degree'])
        out_norm = torch.sqrt(node_dict['out_degree'])

    if not args.eval:
        node_dict.pop('val_mask')
        node_dict.pop('test_mask')

    for epoch in range(args.n_epochs):
        t0 = time.time()
        model.train()
        if args.model == 'graphsage':
            logits = model(graph, feat, BITS, epoch, rank, in_deg)
        elif args.model == 'gnn':
            logits = model(graph, feat, BITS, epoch, rank, in_deg)
        elif args.model == 'gcn':

            logits = model(graph, feat, BITS, epoch, rank, in_norm, out_norm)
        else:
            raise Exception
        if args.inductive:
            loss = loss_fcn(logits, labels)
        else:
            loss = loss_fcn(logits[train_mask], labels)
        del logits
        optimizer.zero_grad(set_to_none=True)
        with ctx.neu_timer.timer(f"backward_{epoch}_"):
            with ctx.backward_timer.timer(f"backward_{epoch}_"):
                loss.backward()
        ctx.buffer.next_epoch()
        pre_reduce = time.time()
        torch.cuda.current_stream().synchronize()
        ctx.reducer.synchronize(epoch, rank)
        reduce_time = time.time() - pre_reduce
        optimizer.step()

        if epoch >= 5 and epoch % args.log_every != 0:
            train_dur.append(time.time() - t0)
            comm_dur.append(ctx.comm_timer.tot_time())
            reduce_dur.append(reduce_time)

            neu_dur.append(ctx.neu_timer.tot_time())
            forward_dur.append(ctx.forward_timer.tot_time())
            backward_dur.append(ctx.backward_timer.tot_time())


        if (epoch + 1) % 10 == 0:
            avg_train_time = np.mean(train_dur)
            avg_comm_time = np.mean(comm_dur)
            avg_reduce_time = np.mean(reduce_dur)
            avg_comp_time = avg_train_time - avg_comm_time - avg_reduce_time
            train_loss = loss.item() / part_train
            print("Process {:03d} | Epoch {:05d} | Time(s) {:.4f} | Comm(s) {:.4f} | Reduce(s) {:.4f} | Loss {:.4f}".format(
                  rank, epoch, avg_train_time, avg_comm_time, avg_reduce_time, train_loss))

        ctx.comm_timer.clear()
        ctx.neu_timer.clear()
        ctx.forward_timer.clear()
        ctx.backward_timer.clear()
        del loss
        if rank == 0 and args.eval and (epoch + 1) % args.log_every == 0:
            if thread is not None:
                model_copy, val_acc = thread.get()
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_model = model_copy
            model_copy = copy.deepcopy(model)
            avg_train_time = np.mean(train_dur)
            avg_comm_time = np.mean(comm_dur)
            avg_reduce_time = np.mean(reduce_dur)
            avg_comp_time = avg_train_time - avg_comm_time - avg_reduce_time
            avg_comm_time = np.mean(comm_dur)
            avg_true_comm_time = ctx.true_comm_timer.avg_time(size)
            avg_forward_comm_time = ctx.forward_comm_timer.avg_time(size)
            avg_backward_comm_time = ctx.backward_comm_timer.avg_time(size)
            avg_qt_time = ctx.qt_timer.avg_time(size)
            avg_dq_time = ctx.dq_timer.avg_time(size)
            avg_pack_time = ctx.pack_timer.avg_time(size)
            avg_unpack_time = ctx.unpack_timer.avg_time(size)
            avg_neu_time = np.mean(neu_dur)

            avg_forward_time = np.mean(forward_dur)
            avg_backward_time = np.mean(backward_dur)
            acc = 0

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
                if FORWARD_QUANTIZATION and BACKWARD_QUANTIZATION:
                    method = 'full_quantization'
                elif FORWARD_QUANTIZATION and not BACKWARD_QUANTIZATION:
                    method = 'forward_quantization'
                elif not FORWARD_QUANTIZATION and BACKWARD_QUANTIZATION:
                    method = 'backward_quantization'
                else:
                    method = 'no_quantization'
                save_results_to_csv(unique_id, timestamp, args.dataset, args.model, method, args.n_partitions, args.partition_method, epoch + 1, avg_comp_time, avg_comm_time, 
                            avg_reduce_time, acc, BITS, STALE, STALE_THRESHOLD, GROUP_NUM, False, False, False, False, args.enable_pipeline,
                            avg_neu_time, avg_forward_time, avg_backward_time, avg_qt_time, avg_dq_time, avg_pack_time, avg_unpack_time,
                            avg_true_comm_time, avg_forward_comm_time, avg_backward_comm_time, transfer_num, forward_transfer_num, backward_transfer_num, QT_ON_COMP, ADAPTIVE, CSV_NAME)

    
    if args.eval and rank == 0:
        if thread is not None:
            model_copy, val_acc = thread.get()
            if val_acc > best_acc:
                best_acc = val_acc
                best_model = model_copy
                
        model_dir = "model"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        file_name = f"{model_dir}/{args.graph_name}_final_fq{FORWARD_QUANTIZATION}_bq{BACKWARD_QUANTIZATION}_bits{BITS}_stale{STALE}_qt{QT_ON_COMP}_stale_thr{STALE_THRESHOLD}.pth.tar"
        torch.save(best_model.state_dict(), file_name)
        print('model saved')
        print("Validation accuracy {:.2%}".format(best_acc))


def check_parser(args):
    if args.norm == 'none':
        args.norm = None


def init_processes(rank, size, args):
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = '%d' % args.port
    dist.init_process_group(args.backend, rank=rank, world_size=size)
    rank, size = dist.get_rank(), dist.get_world_size()
    check_parser(args)
    g, node_dict, gpb = load_partition(args, rank)
    run(g, node_dict, gpb, args)
