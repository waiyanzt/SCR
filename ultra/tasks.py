from functools import reduce
from torch_scatter import scatter_add
from torch_geometric.data import Data
from collections import defaultdict
import torch
from torch.utils import data as torch_data

def edge_match(edge_index, query_index):
    # O((n + q)logn) time
    # O(n) memory
    # edge_index: big underlying graph
    # query_index: edges to match

    # preparing unique hashing of edges, base: (max_node, max_relation) + 1
    base = edge_index.max(dim=1)[0] + 1
    # we will map edges to long ints, so we need to make sure the maximum product is less than MAX_LONG_INT
    # idea: max number of edges = num_nodes * num_relations
    # e.g. for a graph of 10 nodes / 5 relations, edge IDs 0...9 mean all possible outgoing edge types from node 0
    # given a tuple (h, r), we will search for all other existing edges starting from head h
    assert reduce(int.__mul__, base.tolist()) < torch.iinfo(torch.long).max
    scale = base.cumprod(0)
    scale = scale[-1] // scale

    # hash both the original edge index and the query index to unique integers
    edge_hash = (edge_index * scale.unsqueeze(-1)).sum(dim=0)
    edge_hash, order = edge_hash.sort()
    query_hash = (query_index * scale.unsqueeze(-1)).sum(dim=0)

    # matched ranges: [start[i], end[i])
    start = torch.bucketize(query_hash, edge_hash)
    end = torch.bucketize(query_hash, edge_hash, right=True)
    # num_match shows how many edges satisfy the (h, r) pattern for each query in the batch
    num_match = end - start

    # generate the corresponding ranges
    offset = num_match.cumsum(0) - num_match
    range = torch.arange(num_match.sum(), device=edge_index.device)
    range = range + (start - offset).repeat_interleave(num_match)

    return order[range], num_match

def drop_edges(graph, drop_ratio):
    # drop edges randomly with a given ratio
    num_edges = graph.edge_type.size(0)
    num_drop = int(num_edges * (1-drop_ratio))
    drop_index = torch.randperm(num_edges)[:num_drop]
    graph.edge_index = graph.edge_index[:, drop_index]
    graph.edge_type = graph.edge_type[drop_index]
    graph = build_relation_graph(graph)
    return graph

def negative_sampling(data, batch, num_negative, strict=True):
    batch_size = len(batch)
    pos_h_index, pos_t_index, pos_r_index = batch.t()

    # strict negative sampling vs random negative sampling
    if strict:
        t_mask, h_mask = strict_negative_mask(data, batch)
        t_mask = t_mask[:batch_size // 2]
        neg_t_candidate = t_mask.nonzero()[:, 1]
        num_t_candidate = t_mask.sum(dim=-1)
        # draw samples for negative tails
        rand = torch.rand(len(t_mask), num_negative, device=batch.device)
        index = (rand * num_t_candidate.unsqueeze(-1)).long()
        index = index + (num_t_candidate.cumsum(0) - num_t_candidate).unsqueeze(-1)
        neg_t_index = neg_t_candidate[index]

        h_mask = h_mask[batch_size // 2:]
        neg_h_candidate = h_mask.nonzero()[:, 1]
        num_h_candidate = h_mask.sum(dim=-1)
        # draw samples for negative heads
        rand = torch.rand(len(h_mask), num_negative, device=batch.device)
        index = (rand * num_h_candidate.unsqueeze(-1)).long()
        index = index + (num_h_candidate.cumsum(0) - num_h_candidate).unsqueeze(-1)
        neg_h_index = neg_h_candidate[index]
    else:
        neg_index = torch.randint(data.num_nodes, (batch_size, num_negative), device=batch.device)
        neg_t_index, neg_h_index = neg_index[:batch_size // 2], neg_index[batch_size // 2:]

    h_index = pos_h_index.unsqueeze(-1).repeat(1, num_negative + 1)
    t_index = pos_t_index.unsqueeze(-1).repeat(1, num_negative + 1)
    r_index = pos_r_index.unsqueeze(-1).repeat(1, num_negative + 1)
    t_index[:batch_size // 2, 1:] = neg_t_index
    h_index[batch_size // 2:, 1:] = neg_h_index

    return torch.stack([h_index, t_index, r_index], dim=-1)


def all_negative(data, batch):
    pos_h_index, pos_t_index, pos_r_index = batch.t()
    r_index = pos_r_index.unsqueeze(-1).expand(-1, data.num_nodes)
    # generate all negative tails for this batch
    all_index = torch.arange(data.num_nodes, device=batch.device)
    h_index, t_index = torch.meshgrid(pos_h_index, all_index, indexing="ij")  # indexing "xy" would return transposed
    t_batch = torch.stack([h_index, t_index, r_index], dim=-1)
    # generate all negative heads for this batch
    all_index = torch.arange(data.num_nodes, device=batch.device)
    t_index, h_index = torch.meshgrid(pos_t_index, all_index, indexing="ij")
    h_batch = torch.stack([h_index, t_index, r_index], dim=-1)

    return t_batch, h_batch

def label_negative(data, batch, labels):
    pos_h_index, pos_t_index, pos_r_index = batch.t()
    r_index = pos_r_index.unsqueeze(-1).expand(-1, len(labels))
    # generate all negative tails for this batch
    all_index = labels #torch.arange(data.num_nodes, device=batch.device)
    h_index, t_index = torch.meshgrid(pos_h_index, all_index, indexing="ij")  # indexing "xy" would return transposed
    t_batch = torch.stack([h_index, t_index, r_index], dim=-1)
    # generate all negative heads for this batch
    all_index = labels #torch.arange(data.num_nodes, device=batch.device)
    t_index, h_index = torch.meshgrid(pos_t_index, all_index, indexing="ij")
    h_batch = torch.stack([h_index, t_index, r_index], dim=-1)

    return t_batch, h_batch


def strict_negative_mask(data, batch):
    # this function makes sure that for a given (h, r) batch we will NOT sample true tails as random negatives
    # similarly, for a given (t, r) we will NOT sample existing true heads as random negatives

    pos_h_index, pos_t_index, pos_r_index = batch.t()

    # part I: sample hard negative tails
    # edge index of all (head, relation) edges from the underlying graph
    edge_index = torch.stack([data.edge_index[0], data.edge_type])
    # edge index of current batch (head, relation) for which we will sample negatives
    query_index = torch.stack([pos_h_index, pos_r_index])
    # search for all true tails for the given (h, r) batch
    edge_id, num_t_truth = edge_match(edge_index, query_index)
    # build an index from the found edges
    t_truth_index = data.edge_index[1, edge_id]
    sample_id = torch.arange(len(num_t_truth), device=batch.device).repeat_interleave(num_t_truth)
    t_mask = torch.ones(len(num_t_truth), data.num_nodes, dtype=torch.bool, device=batch.device)
    # assign 0s to the mask with the found true tails
    t_mask[sample_id, t_truth_index] = 0
    t_mask.scatter_(1, pos_t_index.unsqueeze(-1), 0)

    # part II: sample hard negative heads
    # edge_index[1] denotes tails, so the edge index becomes (t, r)
    edge_index = torch.stack([data.edge_index[1], data.edge_type])
    # edge index of current batch (tail, relation) for which we will sample heads
    query_index = torch.stack([pos_t_index, pos_r_index])
    # search for all true heads for the given (t, r) batch
    edge_id, num_h_truth = edge_match(edge_index, query_index)
    # build an index from the found edges
    h_truth_index = data.edge_index[0, edge_id]
    sample_id = torch.arange(len(num_h_truth), device=batch.device).repeat_interleave(num_h_truth)
    h_mask = torch.ones(len(num_h_truth), data.num_nodes, dtype=torch.bool, device=batch.device)
    # assign 0s to the mask with the found true heads
    h_mask[sample_id, h_truth_index] = 0
    h_mask.scatter_(1, pos_h_index.unsqueeze(-1), 0)

    return t_mask, h_mask

def compute_ranking(pred, target, mask=None):
    pos_pred = pred.gather(-1, target.unsqueeze(-1))
    if mask is not None:
        # filtered ranking
        ranking = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1
    else:
        # unfiltered ranking
        ranking = torch.sum(pos_pred <= pred, dim=-1) + 1
    return ranking
    
def build_selfloop_edges(graph):
    device = graph.edge_index.device
    rel_mask = graph.edge_type != graph.num_relations
    
    unique_ents = torch.unique(torch.cat([graph.edge_index[0],graph.edge_index[1]]), dim=0) # 2
    self_edge_index = torch.stack([unique_ents, unique_ents], dim=0)
    self_edge_type = torch.ones(len(unique_ents)).to(device).long() * graph.num_relations.long()

    total_edge_index = torch.cat([graph.edge_index.t()[rel_mask].t(),self_edge_index],dim=1)
    total_edge_type = torch.cat([graph.edge_type[rel_mask],self_edge_type],dim=0)
    graph.edge_index = total_edge_index
    graph.edge_type = total_edge_type
    return graph


def build_relation_types(graph):
    # n_rel = graph.num_relations // 2
    n_rel = torch.div(graph.num_relations, 2, rounding_mode='floor').item()
    #####################
    rel_mask = graph.edge_type != graph.num_relations # remove self-loop first
    #####################
    edge_index, edge_type = graph.edge_index.t()[rel_mask].t(), graph.edge_type[rel_mask]
    
    triples = torch.vstack([edge_index[0], edge_type, edge_index[1]]).T.detach().cpu().numpy().tolist()
    filter_dict = defaultdict(lambda: defaultdict(lambda: set()))
    for h, r, t in triples:
        filter_dict[r][h].add(t)

    bern_prob = torch.zeros(n_rel * 2).long()
    rel_type = torch.zeros(n_rel * 2).long()
    rel_type_mat = torch.zeros(n_rel * 2, 4).long()
    for r in range(0, n_rel):
        r2 = r + n_rel
        rel_dict = filter_dict[r]
        rel_dict2 = filter_dict[r2]

        tph, hpt = 0, 0
        if len(rel_dict) > 0:
            tph = sum(len(tails) for tails in rel_dict.values()) / len(rel_dict)
        if len(rel_dict2) > 0:
            hpt = sum(len(heads) for heads in rel_dict2.values()) / len(rel_dict2)
        bern_prob[r] = tph / max(tph + hpt, 1)

        if tph < 1.5 and hpt < 1.5:
            type = 0; type2 = 0  # 1-1
        elif tph >= 1.5 and hpt < 1.5:
            type = 1; type2 = 2  # 1-n
        elif tph < 1.5 and hpt >= 1.5:
            type = 2; type2 = 1  # n-1
        elif tph >= 1.5 and hpt >= 1.5:
            type = 3; type2 = 3  # n-n
        # print(r, tph, hpt, type, r2, type2)
        rel_type[r] = type
        rel_type[r2] = type2
        rel_type_mat[r, type] = 1
        rel_type_mat[r2, type2] = 1
    # rel_type[n_rel * 2] = 0
    graph.relation_types = rel_type
    graph.relation_type_mat = rel_type_mat
    return graph

# now we have two solutions: 1. pretrain without self-relation information (as the second training task?), 2. in prediction, generate virtual triples to replace the self-relation triples

def custom_similarity(A, B, rel_pair):
    # return torch.sparse.mm(A, B).coalesce()
    # K = {0: 0, 5: 0, 1: 0, 6: 1} # A=1,B=1 is the only score
    K = {0: 0, 5: 0, 1: -1, 6: 1} # A=1,B=0 is the worst
    A = A.to_dense()
    B = B.t().to_dense()
    A[:, rel_pair] = 0
    B[:, rel_pair] = 0
    mat = (A + B * 5).to_sparse_coo()
    values = mat._values()
    indices = mat._indices()
    mapped_values = torch.tensor([K[val.item()] for val in values]).to(A.device)
    mat2 = torch.sparse_coo_tensor(indices, mapped_values, mat.shape)
    score_vec = mat2.float().sum(1).to_dense()
    score_vec = score_vec / B.sum()
    return score_vec
    
# how can we convert the relation graph to the virtual triples of one relation    
def build_virtual_triples_forrel(graph, rel, virtual_value = 0.7, triple_num = 100):
    edge_index, edge_type = graph.edge_index, graph.edge_type
    redge_index, redge_type = graph.relation_graph.edge_index, graph.relation_graph.edge_type
    n_rel = graph.num_relations[0].item()//2
    num_nodes, num_rels = graph.num_nodes, graph.num_relations
    device = edge_index.device
    
    Eh = torch.vstack([edge_index[0], edge_type]).T.unique(dim=0)  # (num_edges, 2)
    # Dh = scatter_add(torch.ones_like(Eh[:, 1]), Eh[:, 0])
    Eh = torch.sparse_coo_tensor(
        Eh.T, torch.ones(Eh.shape[0], device=device), # / Dh[Eh[:, 0]], 
        (num_nodes, num_rels)
    )
    Et = torch.vstack([edge_index[1], edge_type]).T.unique(dim=0)  # (num_edges, 2)
    # Dt = scatter_add(torch.ones_like(Et[:, 1]), Et[:, 0])
    Et = torch.sparse_coo_tensor(
        Et.T, torch.ones(Et.shape[0], device=device), # / Dt[Et[:, 0]], 
        (num_nodes, num_rels)
    )
    
    redge_mask = redge_index[0] == rel
    rel_inverse = rel+n_rel if rel < n_rel else rel-n_rel
    R = torch.zeros((num_rels, 4)).float().to(device)
    R[redge_index[1][redge_mask], redge_type[redge_mask]] = 1
    
    Rhh = R[:, :1].to_sparse_coo()
    Rtt = R[:, 1:2].to_sparse_coo()
    Rht = R[:, 2:3].to_sparse_coo()
    Rth = R[:, 3:].to_sparse_coo()
    
    Ahh = custom_similarity(Eh, Rhh, [rel, rel_inverse]) # for head  
    Att = custom_similarity(Et, Rtt, [rel, rel_inverse]) # for tail
    Aht = custom_similarity(Et, Rht, [rel, rel_inverse]) # for head
    Ath = custom_similarity(Eh, Rth, [rel, rel_inverse]) # for tail
    Ah = (Ahh + Aht).to_dense()
    At = (Att + Ath).to_dense() # [N_e]
    
    hh_max = Ahh.max().item() * virtual_value
    ht_max = Aht.max().item() * virtual_value
    tt_max = Att.max().item() * virtual_value
    th_max = Ath.max().item() * virtual_value

    maskH = (Ah >= hh_max+ht_max) | (Ahh >=hh_max) | (Aht >=ht_max)
    maskT = (At >= tt_max+th_max) | (Att >=tt_max) | (Ath >=th_max)
    
    select_entH = torch.nonzero(maskH, as_tuple=False).squeeze(1)
    select_entT = torch.nonzero(maskT, as_tuple=False).squeeze(1)
    print("select_entH", len(select_entH), len(select_entT))
        
    if select_entH.dim() == 0 or select_entT.dim() == 0:
        return None, None
    elif len(select_entH) <= 0 or len(select_entT) <= 0:
        return None, None
    
    if False:
        hid_vec = torch.randint(low=0, high=len(select_entH), size=(triple_num,))
        tid_vec = torch.randint(low=0, high=len(select_entT), size=(triple_num,))
    else:
        # build the distance matrix of nodes
        A = torch.vstack([edge_index[0], edge_index[1]]).T.unique(dim=0)  # (num_edges, 2)
        D = scatter_add(torch.ones_like(A[:, 1]), A[:, 0])
        M = torch.sparse_coo_tensor(
            A.T, torch.ones(A.shape[0], device=device) / D[A[:, 0]], 
            (num_nodes, num_nodes)
        )
        M2 = torch.sparse.mm(M, M).coalesce()
        M3 = torch.sparse.mm(M2, M).coalesce()
        subM2 = M3.to_dense()[select_entH, :][:, select_entT]
        cand_indices = torch.nonzero(subM2, as_tuple=True)
        cand_values = subM2[cand_indices[0], cand_indices[1]]
        cand_values = cand_values[cand_values > 0]
        topk_vs, topk_ids = torch.topk(cand_values, k = min(len(cand_values), triple_num))
        hid_vec = cand_indices[0][topk_ids] 
        tid_vec = cand_indices[1][topk_ids]
    
    virtual_edges = torch.vstack([select_entH[hid_vec], select_entT[tid_vec]]).unique(dim=1)
    virtual_edge_index = torch.cat([virtual_edges, torch.flip(virtual_edges,dims=[0])], dim=1).to(device)
    virtual_edge_type = torch.ones(virtual_edges.shape[1])
    virtual_edge_type = torch.cat([virtual_edge_type * rel, virtual_edge_type * rel_inverse], dim=0).to(device)
    return virtual_edge_index, virtual_edge_type


def build_virtual_triples(graph):
    edge_index, edge_type = graph.edge_index, graph.edge_type
    redge_index, redge_type = graph.relation_graph.edge_index, graph.relation_graph.edge_type
    n_rel = graph.num_relations[0].item()//2
    num_nodes, num_rels = graph.num_nodes, graph.num_relations
    device = edge_index.device
    
    Eh = torch.vstack([edge_index[0], edge_type]).T.unique(dim=0)  # (num_edges, 2)
    Dh = scatter_add(torch.ones_like(Eh[:, 1]), Eh[:, 0])
    Eh = torch.sparse_coo_tensor(
        Eh.T, torch.ones(Eh.shape[0], device=device), # / Dh[Eh[:, 0]], 
        (num_nodes, num_rels)
    )
    Et = torch.vstack([edge_index[1], edge_type]).T.unique(dim=0)  # (num_edges, 2)
    Dt = scatter_add(torch.ones_like(Et[:, 1]), Et[:, 0])
    Et = torch.sparse_coo_tensor(
        Et.T, torch.ones(Et.shape[0], device=device), # / Dt[Et[:, 0]], 
        (num_nodes, num_rels)
    )
    print(Eh.shape, Et.shape)
    
    for rel in range(n_rel * 2): # h-h:0, t-t:1, h-t:2, t-h:3
        redge_mask = redge_index[0] == rel
        rel_inverse = rel+n_rel if rel < n_rel else rel-n_rel
        R = torch.zeros((num_rels, 4)).float()
        R[redge_index[1][redge_mask], redge_type[redge_mask]] = 1
        
        Rhh = R[:, :1].to_sparse_coo()
        Rtt = R[:, 1:2].to_sparse_coo()
        Rht = R[:, 2:3].to_sparse_coo()
        Rth = R[:, 3:].to_sparse_coo()
        
        Ahh = custom_similarity(Eh, Rhh, [rel, rel_inverse]) # for head  
        Att = custom_similarity(Et, Rtt, [rel, rel_inverse]) # for tail
        Aht = custom_similarity(Et, Rht, [rel, rel_inverse]) # for head
        Ath = custom_similarity(Eh, Rth, [rel, rel_inverse]) # for tail
        
        Ah = (Ahh + Aht).to_dense()
        At = (Att + Ath).to_dense() # [N_e]
        
        Ch = Eh.to_dense()[:, rel] # the true entity that in the rel triple
        Ct = Et.to_dense()[:, rel]
        
        k_h = Ch.sum().int().item()
        k_t = Ct.sum().int().item()
        _, topkH = torch.topk(Ah.squeeze(), k = k_h)
        _, topkT = torch.topk(At.squeeze(), k = k_t)
        print("rel", rel, graph.relation_types[rel])
        print("topN", k_h, k_t, graph.relation_types[rel], Ch[topkH].mean(), Ct[topkT].mean())
        
        k_h = min(20,k_h)
        k_t = min(20,k_t)
        _, topkH = torch.topk(Ah.squeeze(), k = k_h)
        _, topkT = torch.topk(At.squeeze(), k = k_t)
        print("top20", k_h, k_t, graph.relation_types[rel], Ch[topkH].mean(), Ct[topkT].mean())
        
        ratio_value = 0.7 # 基本能达到足够的精度 可以利用这两个集合创建三元组测试一下， 直接创建全集或执行模型预测流程，但这个过程本身有天然拓扑缺陷，无法与语义relation embedding抗衡
        hh_max = Ahh.max().item() * ratio_value
        ht_max = Aht.max().item() * ratio_value
        tt_max = Att.max().item() * ratio_value
        th_max = Ath.max().item() * ratio_value
        print(hh_max, ht_max, tt_max, th_max)
        maskH = (Ah >= hh_max+ht_max) | (Ahh >=hh_max) | (Aht >=ht_max)
        maskT = (At >= tt_max+th_max) | (Att >=tt_max) | (Ath >=th_max)
        print("mask", maskH.sum().item(), maskT.sum().item(), graph.relation_types[rel], Ch[maskH].mean(), Ct[maskT].mean())
        print("\n")
        # after select the target head set and tail set, how to generate the mapping between them, which seems important but no support from relation graph.
        if rel > 5: break
    

def build_relation_graph(graph):

    # expect the graph is already with inverse edges
    
    # inv_rel_vocab
    num_nodes, num_rels = graph.num_nodes, graph.num_relations
    #####################
    rel_mask = graph.edge_type != graph.num_relations # remove self-loop first
    #####################
    edge_index, edge_type = graph.edge_index.t()[rel_mask].t(), graph.edge_type[rel_mask]

    device = edge_index.device

    Eh = torch.vstack([edge_index[0], edge_type]).T.unique(dim=0)  # (num_edges, 2)
    
    Dh = scatter_add(torch.ones_like(Eh[:, 1]), Eh[:, 0])

    EhT = torch.sparse_coo_tensor(
        torch.flip(Eh, dims=[1]).T, 
        torch.ones(Eh.shape[0], device=device) / Dh[Eh[:, 0]], 
        (num_rels, num_nodes)
    )
    Eh = torch.sparse_coo_tensor(
        Eh.T, 
        torch.ones(Eh.shape[0], device=device), 
        (num_nodes, num_rels)
    )
    Et = torch.vstack([edge_index[1], edge_type]).T.unique(dim=0)  # (num_edges, 2)

    Dt = scatter_add(torch.ones_like(Et[:, 1]), Et[:, 0])
    assert not (Dt[Et[:, 0]] == 0).any()

    EtT = torch.sparse_coo_tensor(
        torch.flip(Et, dims=[1]).T, 
        torch.ones(Et.shape[0], device=device) / Dt[Et[:, 0]], 
        (num_rels, num_nodes)
    )
    Et = torch.sparse_coo_tensor(
        Et.T, 
        torch.ones(Et.shape[0], device=device), 
        (num_nodes, num_rels)
    )

    Ahh = torch.sparse.mm(EhT, Eh).coalesce()
    Att = torch.sparse.mm(EtT, Et).coalesce()
    Aht = torch.sparse.mm(EhT, Et).coalesce()
    Ath = torch.sparse.mm(EtT, Eh).coalesce()
    hh_edges = torch.cat([Ahh.indices().T, torch.zeros(Ahh.indices().T.shape[0], 1, dtype=torch.long).fill_(0).to(device)], dim=1)  # head to head
    tt_edges = torch.cat([Att.indices().T, torch.zeros(Att.indices().T.shape[0], 1, dtype=torch.long).fill_(1).to(device)], dim=1)  # tail to tail
    ht_edges = torch.cat([Aht.indices().T, torch.zeros(Aht.indices().T.shape[0], 1, dtype=torch.long).fill_(2).to(device)], dim=1)  # head to tail
    th_edges = torch.cat([Ath.indices().T, torch.zeros(Ath.indices().T.shape[0], 1, dtype=torch.long).fill_(3).to(device)], dim=1)  # tail to head
    
    # remove edges of same relation
    # hh_edges = hh_edges[~(hh_edges[:, 0] == hh_edges[:, 1])]
    # tt_edges = tt_edges[~(tt_edges[:, 0] == tt_edges[:, 1])]

    # print(hh_edges.shape, tt_edges.shape, ht_edges.shape, th_edges.shape)
    # print("Ahh range", Ahh.values().max(), Ahh.values().min(), torch.quantile(Ahh.values(), 0.2))
    # print("Att range", Att.values().max(), Att.values().min(), torch.quantile(Att.values(), 0.2))
    # print("Aht range", Aht.values().max(), Aht.values().min(), torch.quantile(Aht.values(), 0.2))
    # print("Ath range", Ath.values().max(), Ath.values().min(), torch.quantile(Ath.values(), 0.2))
    
    all_edge_index = torch.cat([hh_edges[:, [0, 1]].T, tt_edges[:, [0, 1]].T, ht_edges[:, [0, 1]].T, th_edges[:, [0, 1]].T], dim=1)
    all_edge_type = torch.cat([hh_edges[:, 2], tt_edges[:, 2], ht_edges[:, 2], th_edges[:, 2]], dim=0) # origin
    all_edge_weight = torch.cat([Ahh.values().unsqueeze(1), Att.values().unsqueeze(1), Aht.values().unsqueeze(1), Ath.values().unsqueeze(1)], dim=0) # origin
    
    # all_edge_type = torch.cat([ht_edges[:, 2], hh_edges[:, 2], th_edges[:, 2], tt_edges[:, 2]], dim=0) # exchangeType
    # all_edge_type = torch.zeros_like(all_edge_type)  # 0type
    # all_edge_index, all_edge_type = all_edge_index[:,:2], all_edge_type[:2] # 2edge
    # all_edge_index = all_edge_index[:,torch.randperm(all_edge_index.shape[1])] # shuffle
    # all_edge_index = torch.randint(high=all_edge_index.max(), size=all_edge_index.shape) # random
    # cutting
    # indexes = torch.randperm(all_edge_index.shape[1])[:int(len(all_edge_type)*0.5)]
    # all_edge_index = all_edge_index[:,indexes] 
    # all_edge_type = all_edge_type[indexes]
    
    # print("all_relation_edges", all_edge_index.shape, all_edge_type.shape)
    
    rel_graph = Data(
        edge_index=all_edge_index, 
        edge_type=all_edge_type,
        edge_weight = all_edge_weight,
        num_nodes=num_rels, 
        num_relations=4
    )

    graph.relation_graph = rel_graph
    return graph


def build_relation_graph_origin(graph):
    # expect the graph is already with inverse edges

    edge_index, edge_type = graph.edge_index, graph.edge_type
    num_nodes, num_rels = graph.num_nodes, graph.num_relations
    device = edge_index.device

    Eh = torch.vstack([edge_index[0], edge_type]).T.unique(dim=0)  # (num_edges, 2)
    Dh = scatter_add(torch.ones_like(Eh[:, 1]), Eh[:, 0])

    EhT = torch.sparse_coo_tensor(
        torch.flip(Eh, dims=[1]).T, 
        torch.ones(Eh.shape[0], device=device) / Dh[Eh[:, 0]], 
        (num_rels, num_nodes)
    )
    Eh = torch.sparse_coo_tensor(
        Eh.T, 
        torch.ones(Eh.shape[0], device=device), 
        (num_nodes, num_rels)
    )
    Et = torch.vstack([edge_index[1], edge_type]).T.unique(dim=0)  # (num_edges, 2)

    Dt = scatter_add(torch.ones_like(Et[:, 1]), Et[:, 0])
    assert not (Dt[Et[:, 0]] == 0).any()

    EtT = torch.sparse_coo_tensor(
        torch.flip(Et, dims=[1]).T, 
        torch.ones(Et.shape[0], device=device) / Dt[Et[:, 0]], 
        (num_rels, num_nodes)
    )
    Et = torch.sparse_coo_tensor(
        Et.T, 
        torch.ones(Et.shape[0], device=device), 
        (num_nodes, num_rels)
    )

    Ahh = torch.sparse.mm(EhT, Eh).coalesce()
    Att = torch.sparse.mm(EtT, Et).coalesce()
    Aht = torch.sparse.mm(EhT, Et).coalesce()
    Ath = torch.sparse.mm(EtT, Eh).coalesce()

    hh_edges = torch.cat([Ahh.indices().T, torch.zeros(Ahh.indices().T.shape[0], 1, dtype=torch.long).fill_(0)], dim=1)  # head to head
    tt_edges = torch.cat([Att.indices().T, torch.zeros(Att.indices().T.shape[0], 1, dtype=torch.long).fill_(1)], dim=1)  # tail to tail
    ht_edges = torch.cat([Aht.indices().T, torch.zeros(Aht.indices().T.shape[0], 1, dtype=torch.long).fill_(2)], dim=1)  # head to tail
    th_edges = torch.cat([Ath.indices().T, torch.zeros(Ath.indices().T.shape[0], 1, dtype=torch.long).fill_(3)], dim=1)  # tail to head
    
    rel_graph = Data(
        edge_index=torch.cat([hh_edges[:, [0, 1]].T, tt_edges[:, [0, 1]].T, ht_edges[:, [0, 1]].T, th_edges[:, [0, 1]].T], dim=1), 
        edge_type=torch.cat([hh_edges[:, 2], tt_edges[:, 2], ht_edges[:, 2], th_edges[:, 2]], dim=0),
        num_nodes=num_rels, 
        num_relations=4
    )

    graph.relation_graph = rel_graph
    return graph
