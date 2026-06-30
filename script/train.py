import os
import sys
import copy
import math
import pprint
import random
import argparse
from itertools import islice
from functools import partial

import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ultra import tasks, util
from ultra.models import SCR, GraphEncoding
from tqdm import tqdm
from sklearn.metrics import f1_score
import wordninja


separator = ">" * 30
line = "-" * 30
# os.environ["WORLD_SIZE"] = '3'
os.environ['NCCL_DEBUG'] = 'WARN'
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
os.environ["TORCH_USE_CUDA_DSA"] = '1'

default_dataset_type = {
    "LP_Amazon": "linkprediction",
    "LP_DDI": "linkprediction",
    "LP_Collab": "linkprediction",
    "LP_ML1m": "linkprediction",
    "LP_ML10m": "linkprediction",
    "LP_DDIwF": "linkprediction",
    #
    "NC_CORA": "nodeclassification",
    "NC_Citeseer": "nodeclassification",
    "NC_Pubmed": "nodeclassification",
    "NC_CORAwF": "nodeclassification",
    "NC_CiteseerwF": "nodeclassification",  
    "NC_PubmedwF": "nodeclassification",
    #
    "NF_CORA": "nodeclassification",
    "NF_Citeseer": "nodeclassification",
    "NF_Pubmed": "nodeclassification",
    # 
    'NF_Reddit': "nodeclassification",
    'NF_Computers': "nodeclassification",
    'NF_Photo': "nodeclassification",
    'NF_WikiCS': "nodeclassification",
    'NF_Actor': "nodeclassification",
    'NF_Wisconsin': "nodeclassification",
    'NF_Texas': "nodeclassification",
    'NF_Ogbnarxiv': "nodeclassification",
    # 
    'GF_COLLAB': "nodeclassification",
    'GF_IMDB': "nodeclassification",
    'GF_DD': "nodeclassification",
    'GF_PROTEINS': "nodeclassification",
    'GF_BZR': "nodeclassification",
    'GF_COX2': "nodeclassification",
    'GF_ENZYMES': "nodeclassification",
    'GF_MUTAG': "nodeclassification",
    #
    "NF2_CORA": "nodeclassification",
    "NF2_FCora": "nodeclassification",
    "NF2_Citeseer": "nodeclassification",
    "NF2_DBLP": "nodeclassification", 
    "NF2_Pubmed": "nodeclassification",
    "NF2_Arxiv": "nodeclassification",
    "NF2_Wiki": "nodeclassification",
    "NF2_WkCS": "nodeclassification",
    "NF2_Reddit": "nodeclassification",
    "NF2_Product": "nodeclassification",
    "NF2_AmzComp": "nodeclassification",
    "NF2_AmzPhoto": "nodeclassification",
    "NF2_BlogCatalog": "nodeclassification",
    "NF2_LastFMAsia": "nodeclassification",
    "NF2_Deezer": "nodeclassification",
    "NF2_CoCS": "nodeclassification",
    "NF2_CoPhysics": "nodeclassification",
    "NF2_Cornell": "nodeclassification",
    "NF2_Texas": "nodeclassification",
    "NF2_Wisconsin": "nodeclassification",
    "NF2_AirBrazil": "nodeclassification",
    "NF2_AirUS": "nodeclassification",
    "NF2_AirEU": "nodeclassification",
    "NF2_Chameleon": "nodeclassification",
    "NF2_Actor": "nodeclassification",
    "NF2_Squirrel": "nodeclassification",
    "NF2_Roman": "nodeclassification",
    "NF2_AmzRatings": "nodeclassification",
    "NF2_Minesweeper": "nodeclassification",
    "NF2_Tolokers": "nodeclassification",
    "NF2_Questions": "nodeclassification",
}

graph2text_dict = {"FB15k237": "FB", 
                   "WN18RR": "WN", 
                   "CoDExMedium": "WK",
                   "FB15k237Inductive:v1": "FB", 
                   "NELLInductive:v1": "Self",  
                   "WN18RRInductive:v1": "WN",
                   "FB15k237Inductive:v2": "FB", 
                   "NELLInductive:v2": "Self",  
                   "WN18RRInductive:v2": "WN",
                   "FB15k237Inductive:v3": "FB", 
                   "NELLInductive:v3": "Self",  
                   "WN18RRInductive:v3": "WN",
                   "FB15k237Inductive:v4": "FB", 
                   "NELLInductive:v4": "Self",  
                   "WN18RRInductive:v4": "WN",
                   "FBIngram:25": "FB", 
                   "FBIngram:50": "FB", 
                   "FBIngram:75": "FB",
                   "FBIngram:100": "FB",
                   "WKIngram:25": "WK", 
                   "WKIngram:50": "WK", 
                   "WKIngram:75": "WK",
                   "WKIngram:100": "WK",
                   "NLIngram:25": "Self", 
                   "NLIngram:50": "Self", 
                   "NLIngram:75": "Self",
                   "NLIngram:100": "Self",
                   "NLIngram:25": "Self", 
                   "NLIngram:50": "Self", 
                   "NLIngram:75": "Self",
                   "NLIngram:100": "Self",}

def multigraph_collator(batch, train_graphs, noeg_ratio = None):
    num_graphs = len(train_graphs)
    probs = torch.tensor([graph. edge_index.shape[1] for graph in train_graphs]).float() # unbalanced
    
    # probs = torch.ones_like(probs) # add for even training
    
    probs /= probs.sum()
    # print("probs", probs) 

    graph_id = torch.multinomial(probs, 1, replacement=False).item()

    graph = train_graphs[graph_id]
    graph_name = pretrain_graph_names[graph_id] 
    bs = len(batch)
    
    # this is the cause of CUDA error?
    # edge_mask= torch.randint(low=0, high=graph.target_edge_index.shape[1], size=(bs,))
    edge_mask = torch.randperm(graph.target_edge_index.shape[1])[:bs]
    torch.cuda.synchronize()
    # print(edge_mask.shape, graph.target_edge_index.shape, bs)
    batch = torch.cat([graph.target_edge_index[:, edge_mask], graph.target_edge_type[edge_mask].unsqueeze(0)]).t()

    batch_type = 0 
    return graph, batch, graph_name, batch_type
    
# here we assume that train_data and valid_data are tuples of datasets, add an additional dataset dict eval_graph_data for additioinal evaluation
def train_and_validate(cfg, model, train_data, valid_data, filtered_data=None, batch_per_epoch=None, eval_data_dict=None):
    if cfg.train.num_epoch == 0:
        return

    world_size = util.get_world_size()
    rank = util.get_rank()
    
    pretrain_graph_names = cfg.dataset["graphs"]
    
    train_triplets = torch.cat([
        torch.cat([g.target_edge_index, g.target_edge_type.unsqueeze(0)]).t()
        for g in train_data
    ])
    
    # 对node tasks只训练特定关系的预测
    # mask = train_triplets[:, 2] == 1
    # train_triplets = train_triplets[mask]

    cls = cfg.optimizer.pop("class")
    optimizer = getattr(optim, cls)(model.parameters(), **cfg.optimizer)
    num_params = sum(p.numel() for p in model.parameters())
    logger.warning(line)
    logger.warning(f"Number of parameters: {num_params}")

    if world_size > 1:
        parallel_model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True, device_ids=[device])
    else:
        parallel_model = model
    
    sampler = torch_data.DistributedSampler(train_triplets, world_size, rank)
    noeg_ratio = 0.5 if cfg.train["fewshot_train"] < 1 else 0 #0.5
    train_loader = torch_data.DataLoader(train_triplets, cfg.train.batch_size, sampler=sampler, collate_fn=partial(multigraph_collator, train_graphs=train_data, noeg_ratio=noeg_ratio))
    batch_per_epoch = batch_per_epoch or len(train_loader)
        
    # test(cfg, model, eval_graph_data['valid_data'], filtered_data=eval_graph_data["valid_filter"], graph_names = eval_graph_data["graphs"])
    step = math.ceil(cfg.train.num_epoch / 10)
    best_result = float("-inf")
    best_epoch = -1
    batch_id = 0 
    for i in range(0, cfg.train.num_epoch, step):
        parallel_model.train()

        # total_feature_types = ["none","bert","top"]
        # type_randint = random.randint(0,len(total_feature_types)-1)
        # feature_type = total_feature_types[type_randint]
        feature_type = "bert"
        new_train_data = []
        for j, single_train_graph in enumerate(train_data):
            graph_name = pretrain_graph_names[j]
            new_train_graph = graph_encoder.build_semantic_graph(single_train_graph, graph_name, feature_type, mode="train")
            new_train_data.append(new_train_graph)
        train_loader = torch_data.DataLoader(train_triplets, cfg.train.batch_size, sampler=sampler, 
                                             collate_fn=partial(multigraph_collator, train_graphs=new_train_data, noeg_ratio=noeg_ratio))
    
        for epoch in range(i, min(cfg.train.num_epoch, i + step)):
            if util.get_rank() == 0:
                logger.warning(separator)
                logger.warning("Epoch %d begin: %s" % (epoch, feature_type))
            
            # check_gpu_status()
            # for graph_id in range(len(train_data)):
            #     print(graph_id, train_data[graph_id].target_edge_index[:,:10].sum())
            # continue
            losses = []
            sampler.set_epoch(epoch)
            for batch in islice(train_loader, batch_per_epoch):
                # now at each step we sample a new graph and edges from it
                train_graph, batch, graph_name, batch_type = batch
                
                total_feature_types = ["none","bert","top"]
                type_randint = random.randint(0,len(total_feature_types)-1)
                feature_type = total_feature_types[type_randint]
                train_graph = graph_encoder.build_semantic_graph(train_graph, graph_name, feature_type, mode="train")

                batch = tasks.negative_sampling(train_graph, batch, cfg.task.num_negative,
                                                strict=cfg.task.strict_negative)
                
                if "MOE" in cfg.model["entity_model"]["class"]:
                    pred, moe_loss = parallel_model(train_graph, batch)
                    moe_loss *= 0.2
                else:
                    pred = parallel_model(train_graph, batch)
                    moe_loss = 0
                
                target = torch.zeros_like(pred)
                target[:, 0] = 1
                loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
                neg_weight = torch.ones_like(pred)
                if cfg.task.adversarial_temperature > 0:
                    with torch.no_grad():
                        neg_weight[:, 1:] = F.softmax(pred[:, 1:] / cfg.task.adversarial_temperature, dim=-1)
                else:
                    neg_weight[:, 1:] = 1 / cfg.task.num_negative
                loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)

                loss = loss.mean() + moe_loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
              
                if util.get_rank() == 0 and batch_id % cfg.train.log_interval == 0:
                    logger.warning(separator)
                    logger.warning("binary cross entropy: %g" % (loss))

                losses.append(loss.item())
                batch_id += 1

            if util.get_rank() == 0 and len(losses)>0:
                avg_loss = sum(losses) / len(losses)
                logger.warning(separator)
                logger.warning("Epoch %d end" % epoch)
                logger.warning(line)
                logger.warning("average binary cross entropy: %g" % avg_loss)

        epoch = min(cfg.train.num_epoch, i + step)
        if rank == 0:
            logger.warning("Save checkpoint to model_epoch_%d.pth" % epoch)
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(state, "model_epoch_%d.pth" % epoch)

        try:
            if rank == 0:
                logger.warning(separator)
                logger.warning("Evaluate on valid")
            result = test(cfg, model, valid_data, filtered_data=filtered_data, graph_names=pretrain_graph_names)
            # torch.cuda.empty_cache()

            # --- transfer-aware checkpoint selection (optional, gated by cfg.train.transfer_ckpt) ---
            # Default (transfer_ckpt=false): best epoch = KG-valid MRR only (original behavior).
            # When enabled: best epoch = (1-w)*KG_valid_MRR + w*transfer_proxy, where the proxy is the
            # mean of (inductive-KG MRR + NC/GC Accuracy) over cfg.train.transfer_datasets. This avoids
            # keeping a KG-best checkpoint that happens to be worst on GC/NC transfer tasks.
            sel_metric = float(result)
            if bool(cfg.train.get("transfer_ckpt", False)) and eval_data_dict is not None:
                start_ep = int(cfg.train.get("transfer_start_epoch", 0))
                interval = max(1, int(cfg.train.get("transfer_interval", 1)))
                if epoch >= start_ep and (epoch - start_ep) % interval == 0:
                    try:
                        t_datasets = list(cfg.train.get("transfer_datasets", []))
                        t_score = eval_transfer_proxy(cfg, model, eval_data_dict, t_datasets)
                        if t_score is not None:
                            w = float(cfg.train.get("transfer_weight", 0.5))
                            sel_metric = (1 - w) * float(result) + w * t_score
                            if rank == 0:
                                logger.warning("transfer proxy=%.4f  select metric=%.4f (w=%.2f)" % (t_score, sel_metric, w))
                    except Exception as te:
                        if rank == 0:
                            logger.warning("transfer proxy skipped this epoch: %s" % str(te))

            if sel_metric > best_result:
                best_result = sel_metric
                best_epoch = epoch
            # torch.cuda.empty_cache()
        except Exception as e:
            print(e)
            logger.warning(str(e))
        
    if rank == 0:
        logger.warning("Load checkpoint from model_epoch_%d.pth" % best_epoch)
    state = torch.load("model_epoch_%d.pth" % best_epoch, map_location=device)
    model.load_state_dict(state["model"])
    util.synchronize()

def strict_negative_mask(data, batch, mode="train"):
    # this function makes sure that for a given (h, r) batch we will NOT sample true tails as random negatives
    # similarly, for a given (t, r) we will NOT sample existing true heads as random negatives

    pos_h_index, pos_t_index, pos_r_index = batch.t()
    if mode == "target":
        data_edge_index = data.target_edge_index
        data_edge_type = data.target_edge_type
    else:
        data_edge_index = data.edge_index
        data_edge_type = data.edge_type

    # part I: sample hard negative tails
    # edge index of all (head, relation) edges from the underlying graph
    edge_index = torch.stack([data_edge_index[0], data_edge_type])
    # edge index of current batch (head, relation) for which we will sample negatives
    query_index = torch.stack([pos_h_index, pos_r_index])
    # search for all true tails for the given (h, r) batch
    edge_id, num_t_truth = tasks.edge_match(edge_index, query_index)
    # build an index from the found edges
    t_truth_index = data_edge_index[1, edge_id]
    sample_id = torch.arange(len(num_t_truth), device=batch.device).repeat_interleave(num_t_truth)
    t_mask = torch.ones(len(num_t_truth), data.num_nodes, dtype=torch.bool, device=batch.device)
    # assign 0s to the mask with the found true tails
    t_mask[sample_id, t_truth_index] = 0
    return t_mask

@torch.no_grad()
def test_lp(cfg, model, test_data, filtered_data=None, graph_names=None, with_feature="none", mode="test"):
    world_size = util.get_world_size()
    rank = util.get_rank()
    all_metrics = []
    for test_graph, filters, graph_name in zip(test_data, filtered_data, graph_names):
        metric_str = "\t %s: " % (graph_name)
        fact_triplets = torch.cat([test_graph.edge_index, test_graph.edge_type.unsqueeze(0)]).t()
        
        # for single-relation link prediction (like user-item interactions)
        unique_items = torch.unique(test_graph.target_edge_index[0])
        test_triplets = torch.stack([unique_items, unique_items, torch.zeros_like(unique_items)], dim=1)  # the t column is meaningless
    
        sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)
        test_loader = torch_data.DataLoader(test_triplets, cfg.train.batch_size, sampler=sampler)

        test_graph = graph_encoder.build_semantic_graph(test_graph, graph_name, with_feature, mode)

        model.eval()
        topk_list = [20, 40] # , 200, 400, 2000
        all_recall = {k:[] for k in topk_list}
        all_ndcg = {k:[] for k in topk_list}
        for batch in test_loader:
            t_batch, h_batch = tasks.all_negative(test_graph, batch)
            
            if "MOE" in cfg.model["entity_model"]["class"]:
                t_pred, _ = model(test_graph, t_batch)
            else:
                t_pred = model(test_graph, t_batch)
    
            test_t_mask = strict_negative_mask(test_graph, batch, mode="target")
            train_t_mask = strict_negative_mask(filters, batch)
            filter_mask = (train_t_mask == 0) & (~test_t_mask == 0) # 需要排除掉训练集中的项。
            target_mask = (test_t_mask == 0)
            # print(filter_mask.sum(), target_mask.sum())
            
            t_pred[filter_mask] = t_pred.min()
            # target_mask = filter_mask | target_mask
            
            for topk in topk_list:
                t_topk_values, t_topk_predids = torch.topk(t_pred, k=topk, dim=1)
                for i in range(len(batch)):
                    gounds = max(target_mask[i].sum().item(), 1)
                    maxDcg = torch.sum(torch.reciprocal(torch.log2(torch.arange(min(gounds, topk)) + 2)))
                    dcg, hits = 0, 0
                    # print(i, t_topk_values[i].mean(), t_pred[i][target_mask[i]].mean())
                    for j in torch.arange(topk):
                        if target_mask[i][t_topk_predids[i][j]] == 1:
                            dcg += torch.reciprocal(torch.log2(j + 2))
                            hits += 1
                    # print(i, hits, gounds, torch.nonzero(target_mask[i]).view(-1), t_topk_predids[i])
                    # print(hits, gounds)
                    recall = hits / gounds
                    ndcg = dcg / maxDcg  
                    all_recall[topk].append(recall)
                    all_ndcg[topk].append(ndcg)
            
        if world_size > 1:
            dist.all_reduce(all_recall, op=dist.ReduceOp.SUM)
            dist.all_reduce(all_ndcg, op=dist.ReduceOp.SUM)
        
        metrics = {}
        metrics['Graph'] = graph_name
        for topk in topk_list:
            metrics["recall@%d" % topk] = torch.FloatTensor(all_recall[topk]).mean()
            metrics["ndcg@%d" % topk] = torch.FloatTensor(all_ndcg[topk]).mean()
            metric_str += "\tRecall@%d: %.3f NDCG@%d: %.3f" % (topk, metrics["recall@%d" % topk], topk, metrics["ndcg@%d" % topk])
        if rank == 0:
            logger.warning(metric_str)
        torch.cuda.empty_cache()   # release per-graph memory + fragmentation (GF_DD ~336k edges is huge)
        all_metrics.append(metrics)
    return all_metrics

@torch.no_grad()
def test_nc_bak(cfg, model, test_data, filtered_data=None, graph_names=None, node_features=None, with_feature="none", mode="test"):
    world_size = util.get_world_size()
    rank = util.get_rank()
    # test_data is a tuple of validation/test datasets
    # process sequentially
    if util.get_rank() == 0:
        logger.warning("NC Evaluation With Feature Types: " + " ".join(with_feature))
    all_metrics = []
    for test_graph, filters, graph_name, node_feature in zip(test_data, filtered_data, graph_names, node_features):
        metric_str = "\t %s: " % (graph_name)
        print(graph_name, test_graph.num_relations, test_graph.edge_type.max().item(), test_graph.edge_type.shape)
        # print(graph_name, test_graph.num_relations, test_graph.edge_type.max(), test_graph.edge_type.min())
        # print(len(test_graph.relation_graph.edge_index.unique()), test_graph.relation_graph.num_nodes)
        # print(test_graph.num_nodes, test_graph.num_relations, node_feature.shape)
        # print((node_feature.sum(0)==0).sum(), (node_feature.sum(-1)==0).sum(), node_feature[(node_feature.sum(-1)!=0)][:20])
        
        test_triplets = torch.cat([test_graph.target_edge_index, test_graph.target_edge_type.unsqueeze(0)]).t()
        sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)
        # Larger batch for eval: the per-query forward is heavy and the training batch_size (8)
        # underutilizes the GPU. Tunable via cfg.train.test_batch_size (default 32).
        test_batch_size = cfg.train.test_batch_size if "test_batch_size" in cfg.train else 32
        test_loader = torch_data.DataLoader(test_triplets, test_batch_size, sampler=sampler)
    
        # label_ids = torch.unique(test_graph.edge_index.T[test_graph.edge_type==1][:,1])
        label_ids = torch.unique(test_graph.edge_index.T[test_graph.edge_type==torch.unique(test_graph.target_edge_type)][:,1])
        # target_label_ids = torch.unique(test_data.target_edge_index.T[test_data.target_edge_type==1][:,1])
        # print(label_ids, target_label_ids)

        test_graph = graph_encoder.build_semantic_graph(test_graph, graph_name, with_feature, mode)

        model.eval()
        all_preds, all_labels = [], []  # for explicit tail-only evaluation needed for 5 datasets
        ep_acc, ep_tot = 0, 0
        for batch in test_loader:
            t_batch, h_batch = tasks.label_negative(test_graph, batch, label_ids)
            if "MOE" in cfg.model["entity_model"]["class"]:
                t_pred, _ = model(test_graph, t_batch, node_feature)
            else:
                t_pred = model(test_graph, t_batch, node_feature)
            pred_ids = t_pred.argmax(dim=1)
            pred_ls = label_ids[pred_ids]
            labels = batch[:,1]
            # print(pred_ls, labels, label_ids)
            if False:
                topk_num = 2
                _, topk_pred_ids = t_pred.topk(dim=1, k=topk_num)
                cand_ids = label_ids[topk_pred_ids]
                re_batch = torch.stack([batch[:,0:1].repeat(1,topk_num), cand_ids, batch[:,2:].repeat(1,topk_num)], dim=2).view(-1,3).to(t_batch.device)
                tre_batch, hre_batch = tasks.all_negative(test_graph, re_batch)
                if "MOE" in cfg.model["entity_model"]["class"]:
                    h_pred, _ = model(test_graph, hre_batch, node_feature)
                else:
                    h_pred = model(test_graph, hre_batch, node_feature)
                
                sorted_indices = torch.argsort(h_pred, dim=1, descending=True)
                label_positions = (sorted_indices == batch[:,0].repeat(topk_num).unsqueeze(1)).nonzero(as_tuple=False)
                ranks = label_positions[:, 1].view(-1, topk_num) 
                pred_ids = ranks.argmin(dim=1)
                pred_ls = cand_ids[torch.arange(len(batch)), pred_ids]
            # print(pred_ls, cand_ids.view(-1), ranks)
        
            hit = (labels == pred_ls).float().sum().item()
            ep_acc += hit
            ep_tot += labels.shape[0]
            all_preds.append(pred_ls)
            all_labels.append(labels)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        if world_size > 1:
            dist.all_reduce(all_preds, op=dist.ReduceOp.SUM)
            dist.all_reduce(all_labels, op=dist.ReduceOp.SUM)
        
        metrics = {}
        metrics['Graph'] = graph_name
        metrics['Acc'] = torch.Tensor([ep_acc / ep_tot])
        f1_value = f1_score(all_labels.cpu().numpy(), all_preds.cpu().numpy(), average='macro')
        metrics['F1'] = torch.Tensor([f1_value])
        metric_str += "\t%s\t%.3f\t%s\t%.3f" % ('Acc', metrics['Acc'], 'F1', metrics['F1'])
        if rank == 0:
            logger.warning(metric_str)
        torch.cuda.empty_cache()   # release per-graph memory + fragmentation (GF_DD ~336k edges is huge)
        all_metrics.append(metrics)

    return all_metrics

@torch.no_grad()
def test_nc(cfg, model, test_data, filtered_data=None, graph_names=None, node_features=None, with_feature="none", mode="test", k=None, p=None):
    world_size = util.get_world_size()
    rank = util.get_rank()
    # test_data is a tuple of validation/test datasets
    # process sequentially
    if util.get_rank() == 0:
        logger.warning("NC Evaluation With Feature Types: " + with_feature)
    all_metrics = []
    for test_graph, filters, graph_name, node_feature in zip(test_data, filtered_data, graph_names, node_features):
        metric_str = "\t %s: " % (graph_name)
        # print(graph_name, test_graph.num_relations, test_graph.edge_type.max().item(), test_graph.edge_type.shape)
        # print(len(test_graph.relation_graph.edge_index.unique()), test_graph.relation_graph.num_nodes)
        # print(test_graph.num_nodes, test_graph.num_relations, node_feature.shape)
        # print((node_feature.sum(0)==0).sum(), (node_feature.sum(-1)==0).sum(), node_feature[(node_feature.sum(-1)!=0)][:20])
        
        test_triplets = torch.cat([test_graph.target_edge_index, test_graph.target_edge_type.unsqueeze(0)]).t()
        sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)
        # Large GC graphs (PROTEINS 44K, DD 336K, COLLAB 377K nodes) can OOM in
        # message_and_aggregate even at batch_size=8 when GPU has training fragmentation.
        # Auto-reduce batch for graphs exceeding a node-count threshold.
        _nc_bs = cfg.train.batch_size
        if "large_graph_batch_size" in cfg.train and test_graph.num_nodes > cfg.train.get("large_graph_threshold", 30000):
            _nc_bs = cfg.train.large_graph_batch_size
        test_loader = torch_data.DataLoader(test_triplets, _nc_bs, sampler=sampler)
    
        # label_ids = torch.unique(test_graph.edge_index.T[test_graph.edge_type==1][:,1])
        
        label_ids = torch.unique(test_graph.edge_index.T[test_graph.edge_type==torch.unique(test_graph.target_edge_type)][:,1])
        # print(test_graph.target_edge_type.unique(), label_ids)
        # target_label_ids = torch.unique(test_data.target_edge_index.T[test_data.target_edge_type==1][:,1])
        # print(graph_name, label_ids, len(test_triplets), torch.unique(test_graph.edge_type))
        # label_count = torch.Tensor([sum(test_graph.target_edge_index[1]==l) for l in label_ids])
        # print(label_count/len(test_triplets))

        if False:
            target_mask = torch.zeros(test_graph.num_nodes)
            label_relation_type = torch.unique(test_graph.target_edge_type)
            label_ids = torch.unique(test_graph.edge_index.T[test_graph.edge_type==label_relation_type][:,1])
            labeled_ents = test_graph.edge_index[0][(test_graph.edge_type==label_relation_type)].unique()
            target_mask[labeled_ents] = 1   
            target_mask[label_ids] = -1 # label -1, labeled node 1, test node 0
            # print(target_mask.shape, (target_mask==-1).sum(), (target_mask==1).sum(), (target_mask==0).sum())
        else:
            target_mask = None
        test_graph = graph_encoder.build_semantic_graph(test_graph, graph_name, with_feature, mode, 
                                                        node_feature=node_feature, target_mask=target_mask, ifload=False, k=k,p=p) # 
        # print(test_graph.ent_embeds.shape, test_graph.ent_embeds.mean(0), test_graph.ent_embeds[:2], test_graph.ent_embeds[label_ids])
        model.eval()
        all_preds, all_labels = [], []  # for explicit tail-only evaluation needed for 5 datasets
        ep_acc, ep_tot = 0, 0
        for batch in test_loader:
            t_batch, h_batch = tasks.label_negative(test_graph, batch, label_ids)
            if "MOE" in cfg.model["entity_model"]["class"]:
                t_pred, _ = model(test_graph, t_batch, k=k,p=p) # only for graph tasks
            else:
                t_pred = model(test_graph, t_batch, k=k,p=p)
            pred_ids = t_pred.argmax(dim=1)
            pred_ls = label_ids[pred_ids]
            labels = batch[:,1]

            if False:
                topk_num = 2
                _, topk_pred_ids = t_pred.topk(dim=1, k=topk_num)
                cand_ids = label_ids[topk_pred_ids]
                re_batch = torch.stack([batch[:,0:1].repeat(1,topk_num), cand_ids, batch[:,2:].repeat(1,topk_num)], dim=2).view(-1,3).to(t_batch.device)
                tre_batch, hre_batch = tasks.all_negative(test_graph, re_batch)
                if "MOE" in cfg.model["entity_model"]["class"]:
                    h_pred, _ = model(test_graph, hre_batch, node_feature)
                else:
                    h_pred = model(test_graph, hre_batch, node_feature)
                
                sorted_indices = torch.argsort(h_pred, dim=1, descending=True)
                label_positions = (sorted_indices == batch[:,0].repeat(topk_num).unsqueeze(1)).nonzero(as_tuple=False)
                ranks = label_positions[:, 1].view(-1, topk_num) 
                pred_ids = ranks.argmin(dim=1)
                pred_ls = cand_ids[torch.arange(len(batch)), pred_ids]
            # print(pred_ls, cand_ids.view(-1), ranks)
        
            hit = (labels == pred_ls).float().sum().item()
            ep_acc += hit
            ep_tot += labels.shape[0]
            all_preds.append(pred_ls)
            all_labels.append(labels)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        if world_size > 1:
            dist.all_reduce(all_preds, op=dist.ReduceOp.SUM)
            dist.all_reduce(all_labels, op=dist.ReduceOp.SUM)
        
        metrics = {}
        metrics['Graph'] = graph_name
        metrics['Acc'] = torch.Tensor([ep_acc / ep_tot])
        f1_value = f1_score(all_labels.cpu().numpy(), all_preds.cpu().numpy(), average='macro')
        metrics['F1'] = torch.Tensor([f1_value])
        metric_str += "\t%s\t%.4f\t%s\t%.4f" % ('Acc', metrics['Acc'], 'F1', metrics['F1'])
        if rank == 0:
            logger.warning(metric_str)
        torch.cuda.empty_cache()   # release per-graph memory + fragmentation (GF_DD ~336k edges is huge)
        all_metrics.append(metrics)
    return all_metrics

@torch.no_grad()
def test_nc_fewshot(cfg, model, test_data, filtered_data=None, graph_names=None, node_features=None, with_feature="none", mode="test", k=None, p=None, label_prop=None):
    world_size = util.get_world_size()
    rank = util.get_rank()
    # test_data is a tuple of validation/test datasets
    # process sequentially
    if util.get_rank() == 0:
        logger.warning("NC Evaluation With Feature Types: " + with_feature)
    all_metrics = []
    for test_graph, filters, graph_name, node_feature in zip(test_data, filtered_data, graph_names, node_features):
        metric_str = "\t %s: " % (graph_name)
        # print(graph_name, test_graph.num_relations, test_graph.edge_type.max().item(), test_graph.edge_type.shape)
        # print(len(test_graph.relation_graph.edge_index.unique()), test_graph.relation_graph.num_nodes)
        # print(test_graph.num_nodes, test_graph.num_relations, node_feature.shape)
        # print((node_feature.sum(0)==0).sum(), (node_feature.sum(-1)==0).sum(), node_feature[(node_feature.sum(-1)!=0)][:20])
        
        test_triplets = torch.cat([test_graph.target_edge_index, test_graph.target_edge_type.unsqueeze(0)]).t()
        sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)
        batch_size = cfg.train.batch_size
        test_loader = torch_data.DataLoader(test_triplets, batch_size, sampler=sampler)
    
        # label_ids = torch.unique(test_graph.edge_index.T[test_graph.edge_type==1][:,1])
        
        label_ids = torch.unique(test_graph.edge_index.T[test_graph.edge_type==torch.unique(test_graph.target_edge_type)][:,1])
        # print(test_graph.target_edge_type.unique(), label_ids)
        # target_label_ids = torch.unique(test_data.target_edge_index.T[test_data.target_edge_type==1][:,1])
        # print(graph_name, label_ids, len(test_triplets), torch.unique(test_graph.edge_type))
        # label_count = torch.Tensor([sum(test_graph.target_edge_index[1]==l) for l in label_ids])
        # print(label_count/len(test_triplets))

        if False:
            target_mask = torch.zeros(test_graph.num_nodes)
            label_relation_type = torch.unique(test_graph.target_edge_type)
            label_ids = torch.unique(test_graph.edge_index.T[test_graph.edge_type==label_relation_type][:,1])
            labeled_ents = test_graph.edge_index[0][(test_graph.edge_type==label_relation_type)].unique()
            target_mask[labeled_ents] = 1   
            target_mask[label_ids] = -1 # label -1, labeled node 1, test node 0
            # print(target_mask.shape, (target_mask==-1).sum(), (target_mask==1).sum(), (target_mask==0).sum())
        else:
            target_mask = None
        test_graph = graph_encoder.build_semantic_graph(test_graph, graph_name, with_feature, mode, 
                                                        node_feature=node_feature, target_mask=target_mask, ifload=False, k=k,p=p) # 
        # print(test_graph.ent_embeds.shape, test_graph.ent_embeds.mean(0), test_graph.ent_embeds[:2], test_graph.ent_embeds[label_ids])

        # remove the labeled edges from the test graph
        test_graph = copy.deepcopy(test_graph)
        label_relation_type = torch.unique(test_graph.target_edge_type)
        label_revrel_type = label_relation_type + int(test_graph.num_relations//2)
        label_edge_mask = test_graph.edge_type == label_relation_type
        label_edges = test_graph.edge_index.T[label_edge_mask]

        edge_mask = label_edge_mask.clone() | (test_graph.edge_type == label_revrel_type)
        rest_edge_index = test_graph.edge_index.T[~edge_mask]
        rest_edge_type = test_graph.edge_type[~edge_mask]
        
        remove_prop = 0.5 if label_prop is None else label_prop
        if remove_prop == 0:
            selected_indices = None
        elif remove_prop < 1 and remove_prop > 0: # remove with proportion
            label_indices = torch.randperm(len(label_edges))
            selected_indices = label_indices[:int((1-remove_prop)*len(label_indices))]
            # print(len(label_indices), selected_indices.shape)
        else:
            train_labels = test_graph.edge_index.T[label_edge_mask][:,1] 
            shot_num = int(remove_prop) if remove_prop > 0 else -int(remove_prop)
            # get k-shot label indexes
            unique_labels = torch.unique(train_labels)  # Get unique labels
            label_edge_select = torch.zeros_like(train_labels, dtype=torch.long)  # Initialize a 0-1 mask
            selected_indices = torch.IntTensor([]).to(label_edge_mask.device)
            for label in unique_labels:
                label_indices = torch.where(train_labels == label)[0]  # Get indices for the current label
                if remove_prop < 0: # shuffle for -1, -3, -5
                    label_indices = label_indices[torch.randperm(len(label_indices))] # Shuffle the indices
                selected_indices = torch.cat([selected_indices, label_indices[:shot_num]])  # Add the remaining k samples to the selected indices
            # print(selected_indices, type(selected_indices))
        edge_mask = label_edge_mask.clone()
        if selected_indices is not None:
            label_edges = label_edges[selected_indices]
            rest_edge_index = torch.cat([rest_edge_index, label_edges], dim=0)
            rest_edge_type = torch.cat([rest_edge_type, torch.ones(len(label_edges)).to(label_edges.device)*label_relation_type])

            rest_edge_index = torch.cat([rest_edge_index, torch.stack([label_edges[:,1], label_edges[:,0]], dim=1)], dim=0)
            rest_edge_type = torch.cat([rest_edge_type, torch.ones(len(label_edges)).to(label_edges.device)*label_revrel_type])

        # print(remove_prop, label_edge_mask.sum(), (~edge_mask).sum())
        remain_label_num = len(label_edges)
        test_graph.edge_index = rest_edge_index.T.long()
        test_graph.edge_type = rest_edge_type.long()

        model.eval()
        all_preds, all_labels = [], []  # for explicit tail-only evaluation needed for 5 datasets
        ep_acc, ep_tot = 0, 0
        for batch in test_loader:
            t_batch, h_batch = tasks.label_negative(test_graph, batch, label_ids)

            # for each query modify the test_graph
            # query_entity = batch[0][0]
            # specific_graph = copy.deepcopy(test_graph)
            # label_relation_type = torch.unique(test_graph.target_edge_type)

            if "MOE" in cfg.model["entity_model"]["class"]:
                t_pred, _ = model(test_graph, t_batch, k=k,p=p) # only for graph tasks
            else:
                t_pred = model(test_graph, t_batch, k=k,p=p)
            pred_ids = t_pred.argmax(dim=1)
            pred_ls = label_ids[pred_ids]
            labels = batch[:,1]

            if False:
                topk_num = 2
                _, topk_pred_ids = t_pred.topk(dim=1, k=topk_num)
                cand_ids = label_ids[topk_pred_ids]
                re_batch = torch.stack([batch[:,0:1].repeat(1,topk_num), cand_ids, batch[:,2:].repeat(1,topk_num)], dim=2).view(-1,3).to(t_batch.device)
                tre_batch, hre_batch = tasks.all_negative(test_graph, re_batch)
                if "MOE" in cfg.model["entity_model"]["class"]:
                    h_pred, _ = model(test_graph, hre_batch, node_feature)
                else:
                    h_pred = model(test_graph, hre_batch, node_feature)
                
                sorted_indices = torch.argsort(h_pred, dim=1, descending=True)
                label_positions = (sorted_indices == batch[:,0].repeat(topk_num).unsqueeze(1)).nonzero(as_tuple=False)
                ranks = label_positions[:, 1].view(-1, topk_num) 
                pred_ids = ranks.argmin(dim=1)
                pred_ls = cand_ids[torch.arange(len(batch)), pred_ids]
            # print(pred_ls, cand_ids.view(-1), ranks)
        
            hit = (labels == pred_ls).float().sum().item()
            ep_acc += hit
            ep_tot += labels.shape[0]
            all_preds.append(pred_ls)
            all_labels.append(labels)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        if world_size > 1:
            dist.all_reduce(all_preds, op=dist.ReduceOp.SUM)
            dist.all_reduce(all_labels, op=dist.ReduceOp.SUM)
        
        metrics = {}
        metrics['Graph'] = graph_name
        metrics['Acc'] = torch.Tensor([ep_acc / ep_tot])
        f1_value = f1_score(all_labels.cpu().numpy(), all_preds.cpu().numpy(), average='macro')
        metrics['F1'] = torch.Tensor([f1_value])
        metric_str += "Label:%s %s %.4f %s %.4f %s %d" % (remove_prop, 'Acc', metrics['Acc'], 'F1', metrics['F1'], 'Labels', remain_label_num)
        if rank == 0:
            logger.warning(metric_str)
        torch.cuda.empty_cache()
        all_metrics.append(metrics)
    return all_metrics



def load_generaleval_graph(cfg, ds_list, eval_graph_data, mode="transductive", data_type="graph"):
    for graph in ds_list:
        ds, version = graph.split(":") if ":" in graph else (graph, None)
        ds_task_type = default_dataset_type[ds]
        # get dynamic arguments defined in the config file
        vars = util.detect_variables(args.config)
        parser = argparse.ArgumentParser()
        for var in vars:
            parser.add_argument("--%s" % var)
        vars = parser.parse_known_args()[0] # unparsed
        vars = {k: util.literal_eval(v) for k, v in vars._get_kwargs()}
        epochs, batch_per_epoch = 0, 'null'
        vars['epochs'] = epochs
        vars['bpe'] = batch_per_epoch
        vars['dataset'] = ds
        if version is not None:
            vars['version'] = version
        cfg = util.load_config(args.config, context=vars)
        cfg.dataset['class'] = ds
        cfg.dataset.pop('graphs')
        # util.load_config() above re-reads the YAML, resetting root to the relative
        # ./kg-datasets/ which is wrong after create_working_directory() chdir'd the CWD.
        cfg.dataset.root = _abs_dataset_root

        dataset = util.build_dataset(cfg)
        device = util.get_device(cfg)
        train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]
        
        feature_path = os.path.join(train_data.data_path, "node_feats.pt")
        if os.path.exists(feature_path):
            node_features = torch.load(feature_path)
            print(graph, train_data.data_path, node_features.shape)
        else:
            node_features = None
        
        valid_data = tasks.build_relation_graph(valid_data)
        valid_data = tasks.build_relation_types(valid_data)
        test_data = tasks.build_relation_graph(test_data)
        test_data = tasks.build_relation_types(test_data)
        if cfg.model.entity_model["self_loop"]:
            valid_data = tasks.build_selfloop_edges(valid_data)
            test_data = tasks.build_selfloop_edges(test_data)
        
        if "fast_test" in cfg.train and mode == "transductive":
            num_val_edges = cfg.train.fast_test
            short_valid = copy.deepcopy(valid_data)
            mask = torch.randperm(short_valid.target_edge_index.shape[1])[:num_val_edges]
            short_valid.target_edge_index = short_valid.target_edge_index[:, mask]
            short_valid.target_edge_type = short_valid.target_edge_type[mask]
            valid_data = short_valid
        
        filtered_data = Data(edge_index=dataset._data.target_edge_index, edge_type=dataset._data.target_edge_type, num_nodes=dataset[0].num_nodes)
        val_filtered_data = test_filtered_data = filtered_data

        valid_data = valid_data.to(device)
        test_data = test_data.to(device)
        valid_data.original_edge_index = copy.deepcopy(valid_data.edge_index)
        valid_data.original_edge_type = copy.deepcopy(valid_data.edge_type)
        valid_data.original_num_relations = valid_data.num_relations
        test_data.original_edge_index = copy.deepcopy(test_data.edge_index)
        test_data.original_edge_type = copy.deepcopy(test_data.edge_type)
        test_data.original_num_relations = test_data.num_relations

        val_filtered_data = val_filtered_data.to(device)
        test_filtered_data = test_filtered_data.to(device)

        eval_graph_data["graphs"].append(graph)
        eval_graph_data["valid_data"].append(valid_data)
        eval_graph_data["test_data"].append(test_data)
        eval_graph_data["valid_filter"].append(val_filtered_data)
        eval_graph_data["test_filter"].append(test_filtered_data)
        eval_graph_data["node_feature"].append(node_features)
        eval_graph_data["data_type"].append(data_type)
    return eval_graph_data

@torch.no_grad()
def test(cfg, model, test_data, filtered_data=None, graph_names=None, with_feature="none", drop_ratio=0, mode="test"):
    world_size = util.get_world_size()
    rank = util.get_rank()
    # test_data is a tuple of validation/test datasets
    # process sequentially
    if util.get_rank() == 0:
        logger.warning("KG Evaluation With Feature Types: %s Drop Ratio: %.2f" % (with_feature, drop_ratio))
    all_metrics = []
    for test_graph, filters, graph_name in zip(test_data, filtered_data, graph_names):
        # print(graph_name, test_graph.num_relations, test_graph.edge_type.max().item(), test_graph.edge_type.shape())
        # print(len(test_graph.relation_graph.edge_index.unique()), len(test_graph.inv_rel_vocab), test_graph.relation_graph.num_nodes)
        metric_str = "\t %s: " % (graph_name)
        test_triplets = torch.cat([test_graph.target_edge_index, test_graph.target_edge_type.unsqueeze(0)]).t()
        sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)
        # Larger batch for eval: the per-query forward is heavy and the training batch_size (8)
        # underutilizes the GPU. Tunable via cfg.train.test_batch_size (default 32).
        test_batch_size = cfg.train.test_batch_size if "test_batch_size" in cfg.train else 32
        test_loader = torch_data.DataLoader(test_triplets, test_batch_size, sampler=sampler)

        if drop_ratio > 0: test_graph = tasks.drop_edges(copy.deepcopy(test_graph), drop_ratio)
        test_graph = graph_encoder.build_semantic_graph(test_graph, graph_name, with_feature, mode, ifload=(drop_ratio == 0))
        # print(with_feature, test_graph.ex_relation_graph.edge_type.shape)

        model.eval()
        rankings = []
        num_negatives = []
        for batch in test_loader:
            t_batch, h_batch = tasks.all_negative(test_graph, batch)
            # print(graph_name, batch.shape)
            if "MOE" in cfg.model["entity_model"]["class"]:
                t_pred, _ = model(test_graph, t_batch)
                h_pred, _ = model(test_graph, h_batch)
            else:
                t_pred = model(test_graph, t_batch)
                h_pred = model(test_graph, h_batch)

            if filtered_data is None:
                t_mask, h_mask = tasks.strict_negative_mask(test_graph, batch)
            else:
                t_mask, h_mask = tasks.strict_negative_mask(filters, batch)
            pos_h_index, pos_t_index, pos_r_index = batch.t()
            t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
            h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask)
            num_t_negative = t_mask.sum(dim=-1)
            num_h_negative = h_mask.sum(dim=-1)

            rankings += [t_ranking, h_ranking]
            num_negatives += [num_t_negative, num_h_negative]

        ranking = torch.cat(rankings)
        num_negative = torch.cat(num_negatives)
        all_size = torch.zeros(world_size, dtype=torch.long, device=device)
        all_size[rank] = len(ranking)
        if world_size > 1:
            dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
        cum_size = all_size.cumsum(0)
        all_ranking = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
        all_ranking[cum_size[rank] - all_size[rank]: cum_size[rank]] = ranking
        all_num_negative = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
        all_num_negative[cum_size[rank] - all_size[rank]: cum_size[rank]] = num_negative
        if world_size > 1:
            dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
            dist.all_reduce(all_num_negative, op=dist.ReduceOp.SUM)

        if rank == 0:
            for metric in cfg.task.metric:
                if metric == "mr":
                    score = all_ranking.float().mean()
                elif metric == "mrr":
                    score = (1 / all_ranking.float()).mean()
                elif metric.startswith("hits@"):
                    values = metric[5:].split("_")
                    threshold = int(values[0])
                    if len(values) > 1:
                        num_sample = int(values[1])
                        # unbiased estimation
                        fp_rate = (all_ranking - 1).float() / all_num_negative
                        score = 0
                        for i in range(threshold):
                            # choose i false positive from num_sample - 1 negatives
                            num_comb = math.factorial(num_sample - 1) / \
                                    math.factorial(i) / math.factorial(num_sample - i - 1)
                            score += num_comb * (fp_rate ** i) * ((1 - fp_rate) ** (num_sample - i - 1))
                        score = score.mean()
                    else:
                        score = (all_ranking <= threshold).float().mean()
                metric_str += "\t%s\t%.3f" % (metric, score)
        mrr = (1 / all_ranking.float()).mean()

        all_metrics.append(mrr)
        if rank == 0:
            # logger.warning(separator)
            logger.warning(metric_str)
        # torch.cuda.empty_cache()
        # torch.cuda.synchronize()

    avg_metric = sum(all_metrics) / max(1,len(all_metrics))
    return avg_metric

def load_eval_graph(cfg, ds_list, eval_graph_data, mode="transductive"):
    for graph in ds_list:
        ds, version = graph.split(":") if ":" in graph else (graph, None)
        # get dynamic arguments defined in the config file
        vars = util.detect_variables(args.config)
        parser = argparse.ArgumentParser()
        for var in vars:
            parser.add_argument("--%s" % var)
        vars = parser.parse_known_args()[0] # unparsed
        vars = {k: util.literal_eval(v) for k, v in vars._get_kwargs()}
        epochs, batch_per_epoch = 0, 'null'
        vars['epochs'] = epochs
        vars['bpe'] = batch_per_epoch
        vars['dataset'] = ds
        if version is not None:
            vars['version'] = version
        cfg = util.load_config(args.config, context=vars)
        cfg.dataset['class'] = ds
        if ds not in ["FB15k237", "WN18RR", "CoDExMedium", "CoDExSmall"]:
            cfg.dataset['version'] = version
        cfg.dataset.pop('graphs')
        # util.load_config() above re-reads the YAML, resetting root to the relative
        # ./kg-datasets/ which is wrong after create_working_directory() chdir'd the CWD.
        # Restore the absolute root saved before the chdir — same as eval.py does.
        cfg.dataset.root = _abs_dataset_root

        # root_dir = os.path.expanduser(cfg.output_dir) # resetting the path to avoid inf nesting
        # os.chdir(root_dir)
        # working_dir = util.create_working_directory(cfg)

        # logger = util.get_root_logger()
        # if util.get_rank() == 0:
        #     logger.warning("Config file: %s" % args.config)
        #     logger.warning(pprint.pformat(cfg))
        task_name = cfg.task["name"]
        dataset = util.build_dataset(cfg)
        device = util.get_device(cfg)
        train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]

        valid_data = tasks.build_relation_graph(valid_data)
        valid_data = tasks.build_relation_types(valid_data)
        test_data = tasks.build_relation_graph(test_data)
        test_data = tasks.build_relation_types(test_data)
        if cfg.model.entity_model["self_loop"]:
            valid_data = tasks.build_selfloop_edges(valid_data)
            test_data = tasks.build_selfloop_edges(test_data)
        
        if "fast_test" in cfg.train and mode == "transductive":
            num_val_edges = cfg.train.fast_test
            short_valid = copy.deepcopy(valid_data)
            mask = torch.randperm(short_valid.target_edge_index.shape[1])[:num_val_edges]
            short_valid.target_edge_index = short_valid.target_edge_index[:, mask]
            short_valid.target_edge_type = short_valid.target_edge_type[mask]
            valid_data = short_valid
        
        if mode == "inductive":
            # filtering for inductive datasets
            # Grail, MTDEA, HM datasets have validation sets based off the training graph
            # ILPC, Ingram have validation sets from the inference graph
            # filtering dataset should contain all true edges (base graph + (valid) + test) 
            if "ILPC" in cfg.dataset['class'] or "Ingram" in cfg.dataset['class']:
                # add inference, valid, test as the validation and test filtering graphs
                full_inference_edges = torch.cat([valid_data.edge_index, valid_data.target_edge_index, test_data.target_edge_index], dim=1)
                full_inference_etypes = torch.cat([valid_data.edge_type, valid_data.target_edge_type, test_data.target_edge_type])
                test_filtered_data = Data(edge_index=full_inference_edges, edge_type=full_inference_etypes, num_nodes=test_data.num_nodes)
                val_filtered_data = test_filtered_data
            else:
                # test filtering graph: inference edges + test edges
                full_inference_edges = torch.cat([test_data.edge_index, test_data.target_edge_index], dim=1)
                full_inference_etypes = torch.cat([test_data.edge_type, test_data.target_edge_type])
                test_filtered_data = Data(edge_index=full_inference_edges, edge_type=full_inference_etypes, num_nodes=test_data.num_nodes)
    
                # validation filtering graph: train edges + validation edges
                val_filtered_data = Data(
                    edge_index=torch.cat([train_data.edge_index, valid_data.target_edge_index], dim=1),
                    edge_type=torch.cat([train_data.edge_type, valid_data.target_edge_type])
                )
        else:
            # for transductive setting, use the whole graph for filtered ranking
            filtered_data = Data(edge_index=dataset._data.target_edge_index, edge_type=dataset._data.target_edge_type, num_nodes=dataset[0].num_nodes)
            val_filtered_data = test_filtered_data = filtered_data
                    
        valid_data = valid_data.to(device)
        test_data = test_data.to(device)
        valid_data.original_edge_index = copy.deepcopy(valid_data.edge_index)
        valid_data.original_edge_type = copy.deepcopy(valid_data.edge_type)
        valid_data.original_num_relations = valid_data.num_relations
        test_data.original_edge_index = copy.deepcopy(test_data.edge_index)
        test_data.original_edge_type = copy.deepcopy(test_data.edge_type)
        test_data.original_num_relations = test_data.num_relations

        val_filtered_data = val_filtered_data.to(device)
        test_filtered_data = test_filtered_data.to(device)
        # eval_graph_data[graph] = {"dataset": ds, "version": version, "valid_data": valid_data, "test_data": test_data, "valid_filter":val_filtered_data, "test_filter":test_filtered_data}
        eval_graph_data["graphs"].append(graph)
        eval_graph_data["valid_data"].append(valid_data)
        eval_graph_data["test_data"].append(test_data)
        eval_graph_data["valid_filter"].append(val_filtered_data)
        eval_graph_data["test_filter"].append(test_filtered_data)
    return eval_graph_data

def eval_transfer_proxy(cfg, model, eval_data_dict, datasets, mode="test"):
    """Lightweight transfer metric on a small dataset subset, used for checkpoint selection.

    Returns a normalized score in [0, 1] = mean of (inductive-KG MRR + NC/GC Accuracy)
    over the requested `datasets`. Returns None if none of the datasets are available.

    Why: the default best-checkpoint rule only looks at KG-valid MRR, which is blind to
    GC/NC transfer. Evaluating a 2-3 dataset proxy (e.g. NELL:v1 + MUTAG + Wisconsin) is
    cheap (~tens of seconds) yet lets GC/NC influence which epoch is kept.
    """
    if not datasets:
        return None
    hetelink = eval_data_dict["eval_hetelink_data"]
    nodegraph = eval_data_dict["eval_nodegraph_data"]
    score_terms = []

    # inductive KG subset: test() returns avg MRR (clean float in [0, 1])
    kg_idx = [i for i, g in enumerate(hetelink["graphs"]) if g in datasets]
    if kg_idx:
        sub_data = [hetelink[mode + "_data"][i] for i in kg_idx]
        sub_filter = [hetelink[mode + "_filter"][i] for i in kg_idx]
        sub_graphs = [hetelink["graphs"][i] for i in kg_idx]
        kg_mrr = test(cfg, model, sub_data, filtered_data=sub_filter, graph_names=sub_graphs,
                      with_feature="bert", mode=mode)
        score_terms.append(float(kg_mrr))

    # NC/GC subset: test_nc() returns [{'Graph': name, 'Acc': tensor, 'F1': tensor}, ...].
    # Evaluate each dataset with its tuned (k, p). Without this, GC graphs fall through
    # SCR.forward's `if k is not None` guard into the flat path and return a degenerate,
    # epoch-invariant Acc (e.g. MUTAG stuck at exactly 0.6467) -> no selection signal.
    # (k, p) mirror the eval.py GC_BEST_KP / NC tuning so the proxy reflects real quality.
    PROXY_KP = {
        "GF_MUTAG": (10, 0.3), "GF_DD": (3, 0.3),
        "GF_IMDB": (5, 0.3), "GF_PROTEINS": (5, 0.3),
        "GF_ENZYMES": (5, 0.3), "GF_BZR": (5, 0.3),
        "GF_COLLAB": (1, 0.3), "GF_COX2": (None, None),   # COX2 = flat mode
        "NF_Pubmed": (1, 0.8), "NF_Wisconsin": (1, 0.5), "NF_Texas": (1, 0.3),
    }
    accs = []
    for i in (idx for idx, g in enumerate(nodegraph["graphs"]) if g in datasets):
        gname = nodegraph["graphs"][i]
        k, p = PROXY_KP.get(gname, (None, None))
        nc_out = test_nc(cfg, model, [nodegraph[mode + "_data"][i]],
                         filtered_data=[nodegraph[mode + "_filter"][i]], graph_names=[gname],
                         node_features=[nodegraph["node_feature"][i]], with_feature="bert",
                         mode=mode, k=k, p=p)
        accs.extend(float(m["Acc"]) for m in nc_out if "Acc" in m)
    if accs:
        score_terms.append(sum(accs) / len(accs))

    if not score_terms:
        return None
    return sum(score_terms) / len(score_terms)


def global_test(cfg, model, eval_data_dict, mode="test", k=None, p=None):
    eval_hetelink_data = eval_data_dict["eval_hetelink_data"]
    eval_nodegraph_data = eval_data_dict["eval_nodegraph_data"]
    eval_homolink_data = eval_data_dict["eval_homolink_data"]
    all_metrics = []
    # test(cfg, model, eval_hetelink_data[mode+'_data'], filtered_data=eval_hetelink_data[mode+"_filter"], graph_names = eval_hetelink_data["graphs"], with_feature="none", mode=mode)
    for feature in ["bert"]: # graph_encoder.semantic_types()
        test(cfg, model, eval_hetelink_data[mode+'_data'], filtered_data=eval_hetelink_data[mode+"_filter"], graph_names = eval_hetelink_data["graphs"], with_feature=feature, mode=mode)
        all_metrics = test_nc(cfg, model, eval_nodegraph_data[mode+'_data'], filtered_data=eval_nodegraph_data[mode+"_filter"], graph_names = eval_nodegraph_data["graphs"], 
                              node_features = eval_nodegraph_data["node_feature"], with_feature=feature, k=k, p=p)
        
        # for label_prop in [0.9, 0.8, 0.7, 1, 3, 5, 10, -1, -3, -5, -10]: # 
        #     test_nc_fewshot(cfg, model, eval_nodegraph_data[mode+'_data'], filtered_data=eval_nodegraph_data[mode+"_filter"], graph_names = eval_nodegraph_data["graphs"] 
        #                        node_features = eval_nodegraph_data["node_feature"], with_feature=feature, k=k, p=p, label_prop=label_prop)
        
    # for k,p in [(1,0.5), (3,0.5), (5,0.5), (10,0.5), (20,0.5)]:
    #     print("k=%d, p=%.2f" % (k,p))
    #     test_nc(cfg, model, eval_nodegraph_data[mode+'_data'], filtered_data=eval_nodegraph_data[mode+"_filter"], graph_names = eval_nodegraph_data["graphs"], 
    #                           node_features = eval_nodegraph_data["node_feature"], with_feature=feature, k=k, p=p)
    
    # test_lp(cfg, model, eval_homolink_data[mode+'_data'], filtered_data=eval_homolink_data[mode+"_filter"], graph_names = eval_homolink_data["graphs"], with_feature="bert")
    # for drop_ratio in [0.1, 0.3, 0.5]:
    #     test(cfg, model, eval_hetelink_data[mode+'_data'], filtered_data=eval_hetelink_data[mode+"_filter"], graph_names = eval_hetelink_data["graphs"], with_feature="none", drop_ratio=drop_ratio, mode=mode)
    #     test(cfg, model, eval_hetelink_data[mode+'_data'], filtered_data=eval_hetelink_data[mode+"_filter"], graph_names = eval_hetelink_data["graphs"], with_feature="bert", drop_ratio=drop_ratio, mode=mode)
    return all_metrics

def process_output(total_output_list):
    graph_names = []
    graph_metrics = {}
    for single_output in total_output_list:
        graph_name = single_output['Graph']
        if graph_name not in graph_metrics:
            graph_metrics[graph_name] = {}
        for k, v in single_output.items():
            if k not in ['Graph']:
                if k not in graph_metrics[graph_name]:  
                    graph_metrics[graph_name][k] = []
                graph_metrics[graph_name][k].append(v.item())
    for graph_name in graph_metrics:
        print(graph_name)
        for k, v in graph_metrics[graph_name].items():
            v = torch.tensor(v)
            metric_str = "\t%s\t%.4f\t%s\t%.4f" % (k, v.mean().item(), "std",v.std().item())
            print(metric_str)
    return graph_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file",
                        default=os.path.join(os.path.dirname(__file__), "..", "config", "train.yaml"))
    parser.add_argument("-g", "--gpu", help="GPU device id", default=0, type=int)
    parser.add_argument("--ckpt", help="checkpoint path (default: ./ckpts/scr_3g.pth)",
                        default=None)
    parser.add_argument("--no-freeze", action="store_true",
                        help="do not freeze relation_model and entity_model.layers (full fine-tuning)")
    parser.add_argument("-s", "--seed", help="random seed", type=int, default=1024)

    args, unparsed = parser.parse_known_args()
    print("gpu:", args.gpu)

    # Set random seed
    import numpy as np
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    torch.cuda.set_device(args.gpu)

    # Resolve config path to absolute before any chdir
    args.config = os.path.abspath(args.config)

    # Resolve project root: walk up from config file until we find a directory
    # containing both 'config/' and 'script/'
    _project_dir = os.path.dirname(os.path.dirname(args.config))
    while not os.path.isdir(os.path.join(_project_dir, "script")):
        parent = os.path.dirname(_project_dir)
        if parent == _project_dir:
            break
        _project_dir = parent

    cfg = util.load_config(args.config)

    # Resolve the dataset root to an absolute path BEFORE create_working_directory()
    # changes the CWD to ./output/.../. Otherwise, a relative root such as ./kg-datasets/
    # would be resolved relative to the new (wrong) working directory: PyG would not find
    # the data, fall into download(), and crash (e.g. 'FB15k237' object has no attribute 'urls').
    # Mirrors the same fix already applied in script/eval.py.
    if "root" in cfg.dataset:
        cfg.dataset.root = os.path.abspath(cfg.dataset.root)
    _abs_dataset_root = cfg.dataset.root
    # Same chdir issue: resolve checkpoint to absolute before CWD changes.
    if "checkpoint" in cfg:
        cfg.checkpoint = os.path.abspath(cfg.checkpoint)

    working_dir = util.create_working_directory(cfg)

    # Override checkpoint from CLI if specified
    if args.ckpt is not None:
        cfg.checkpoint = args.ckpt
    # Default checkpoint: ./ckpts/scr_3g.pth (already set in YAML)

    logger = util.get_root_logger()
    if util.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))

    task_name = cfg.task["name"]
    pretrain_graph_names = cfg.dataset["graphs"]
    dataset = util.build_dataset(cfg)
    device = util.get_device(cfg)

    train_data, valid_data, test_data = dataset._data[0], dataset._data[1], dataset._data[2]
    pretrain_graph_names = cfg.dataset["graphs"]
    # Auto-detect resource path relative to project (use project_dir already resolved)
    resource_path = os.path.join(_project_dir, "resource") + os.sep
    logger.warning("Resource path: %s" % resource_path)
    graph_encoder = GraphEncoding(cfg.model["entity_model"]["feat_dim"], resource_path, graph2text_dict, k=3, p=0.8)
    
    # preprocessing train datasets
    for i, train_graph in enumerate(train_data):
        train_graph = graph_encoder.build_semantic_graph(train_graph, pretrain_graph_names[i], semantic_type="bert", k=10, p=0.5)
        train_data[i] = train_graph
     
    eval_hetelink_data = {"graphs":[], "valid_data":[], "test_data":[], "valid_filter":[], "test_filter":[], }
    eval_hetelink_data = load_eval_graph(cfg, cfg.moreeval["evalgraphs"], eval_hetelink_data, mode="transductive")
    eval_hetelink_data = load_eval_graph(cfg, cfg.moreeval["ind_evalgraphs"], eval_hetelink_data, mode="inductive")
    
    eval_nodegraph_data = {"graphs":[], "valid_data":[], "test_data":[], "valid_filter":[], "test_filter":[], "node_feature":[], "data_type":[]}
    eval_nodegraph_data = load_generaleval_graph(cfg, cfg.moreeval["nodeevalgraphs"], eval_nodegraph_data, data_type="node")
    eval_nodegraph_data = load_generaleval_graph(cfg, cfg.moreeval["graphevalgraphs"], eval_nodegraph_data, data_type="graph")

    eval_homolink_data = {"graphs":[], "valid_data":[], "test_data":[], "valid_filter":[], "test_filter":[], }
    eval_homolink_data = load_eval_graph(cfg, cfg.moreeval["homolinkgraphs"], eval_homolink_data, mode="transductive")
    
    eval_data_dict = {"eval_hetelink_data":eval_hetelink_data, "eval_nodegraph_data":eval_nodegraph_data, "eval_homolink_data":eval_homolink_data}
    
    if "fast_test" in cfg.train:
        num_val_edges = cfg.train.fast_test
        if util.get_rank() == 0:
            logger.warning(f"Fast evaluation on {num_val_edges} samples in validation")
        short_valid = [copy.deepcopy(vd) for vd in valid_data]
        for graph in short_valid:
            mask = torch.randperm(graph.target_edge_index.shape[1])[:num_val_edges]
            graph.target_edge_index = graph.target_edge_index[:, mask]
            graph.target_edge_type = graph.target_edge_type[mask]
        
        short_valid = [sv.to(device) for sv in short_valid]

    train_data = [td.to(device) for td in train_data]
    valid_data = [vd.to(device) for vd in valid_data]
    test_data = [tst.to(device) for tst in test_data]

    model = SCR(
        rel_model_cfg=cfg.model.relation_model,
        entity_model_cfg=cfg.model.entity_model,
    )
    
    if cfg.checkpoint == "none": cfg.checkpoint = None

    if "checkpoint" in cfg and cfg.checkpoint is not None:
        print("feat_type", model.entity_model.feat_type, cfg.checkpoint)
        state = torch.load(cfg.checkpoint, map_location="cpu")
        model_state_dict = model.state_dict()
        filtered_state_dict = {k: v for k, v in state["model"].items() if k in model_state_dict and v.size() == model_state_dict[k].size()}
        model.load_state_dict(filtered_state_dict, strict=False)
    model = model.to(device)

    assert task_name == "MultiGraphPretraining", "Only the MultiGraphPretraining task is allowed for this script"
    
    for name, param in model.named_parameters():
        # print(name)
        if not args.no_freeze:
            if "relation_model" in name:
                param.requires_grad = False
            if "entity_model.layers" in name:
                param.requires_grad = False
        else:
            param.requires_grad = True

    total_output_list = []

    # for transductive setting, use the whole graph for filtered ranking
    filtered_data = [
        Data(
            edge_index=torch.cat([trg.edge_index, valg.target_edge_index, testg.target_edge_index], dim=1),  # trg.target_edge_index
            edge_type=torch.cat([trg.edge_type, valg.target_edge_type, testg.target_edge_type,]), # trg.target_edge_type
            num_nodes=trg.num_nodes).to(device)
        for trg, valg, testg in zip(train_data, valid_data, test_data)
    ]
    
    train_and_validate(cfg, model, train_data, valid_data if "fast_test" not in cfg.train else short_valid, filtered_data=filtered_data, 
                       batch_per_epoch=cfg.train.batch_per_epoch, eval_data_dict = eval_data_dict)
                       
    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on test")
    test(cfg, model, test_data, filtered_data=filtered_data, graph_names = pretrain_graph_names)
    torch.cuda.empty_cache()   # release training-loop fragmentation before the heavy transfer eval (avoids CUDA OOM on GF_DD)
    global_test(cfg, model, eval_data_dict, mode="test")
    
    
    