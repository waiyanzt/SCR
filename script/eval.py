"""
SCR — Evaluation Script
Paper: Towards Graph Foundation Models: Training on Knowledge Graphs Enables Transferability to General Graphs
Model: SCR (Semantic Conditional Reasoner)

This script evaluates a pretrained SCR model on KG link prediction, node classification,
and graph classification tasks with zero-shot transfer.

Hardcoded optimal k/p per dataset (from exhaustive sweep experiments):
  - KG tasks: default parameters (no tuning needed)
  - NC tasks: most use defaults; PubMed/Wisconsin/Texas need k=1
  - GC tasks: most use defaults; MUTAG needs k=10, DD needs k=3
  - GC adaptive routing: COX2 uses flat mode (entity-level only, no meta-graph)
"""

import os
import sys
import csv
import copy
import time
import math
import pprint
import random
import argparse

import torch
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ultra import tasks, util
from ultra.models import SCR, GraphEncoding
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
import wordninja
import numpy as np


separator = ">" * 30
line = "-" * 30
os.environ['NCCL_DEBUG'] = 'WARN'
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
os.environ["TORCH_USE_CUDA_DSA"] = '1'
disable_tqdm = not sys.stdout.isatty()

# Node Classification: optimal k and p
# "default" means use GraphEncoding defaults (k=3, p=0.8)
NC_BEST_KP = {
    "NF_CORA":     (None, None), 
    "NF_Citeseer": (None, None), 
    "NF_Pubmed":   (1, 0.8),    
    "NF_Wisconsin":(1, 0.5),   
    "NF_Texas":    (1, 0.3),    
    "NF_Actor":    (None, None),  
}

# Graph Classification: optimal k and p for meta-graph (build_global)
# (None, None) means use build_global defaults (k=5, p=0.3).
# IMPORTANT: These Nones are resolved to (5, 0.3) in test_gc before calling
# the model, because SCR.forward uses "if k is not None" to enter GC mode.
GC_BEST_KP = {
    "GF_IMDB":     (None, None),
    "GF_COLLAB":   (1, None),  
    "GF_PROTEINS": (None, None), 
    "GF_MUTAG":    (10, None),
    "GF_ENZYMES":  (None, None), 
    "GF_COX2":     (None, None),
    "GF_BZR":      (None, None), 
    "GF_DD":       (3, None),
}

# Adaptive routing: datasets where flat model (SCR) outperforms meta model (SCR GC mode).
# These datasets skip build_global and use entity-level prediction directly.
GC_FLAT_DATASETS = {
    "GF_COX2", 
}


default_dataset_type = {
    "NC_CORA": "nodeclassification", "NC_Citeseer": "nodeclassification",
    "NC_Pubmed": "nodeclassification", "NC_CORAwF": "nodeclassification",
    "NC_CiteseerwF": "nodeclassification", "NC_PubmedwF": "nodeclassification",
    "NF_CORA": "nodeclassification", "NF_Citeseer": "nodeclassification",
    "NF_Pubmed": "nodeclassification",
    'NF_Reddit': "nodeclassification", 'NF_Computers': "nodeclassification",
    'NF_Photo': "nodeclassification", 'NF_WikiCS': "nodeclassification",
    'NF_Actor': "nodeclassification", 'NF_Wisconsin': "nodeclassification",
    'NF_Texas': "nodeclassification", 'NF_Ogbnarxiv': "nodeclassification",
    'GF_COLLAB': "nodeclassification", 'GF_IMDB': "nodeclassification",
    'GF_DD': "nodeclassification", 'GF_PROTEINS': "nodeclassification",
    'GF_BZR': "nodeclassification", 'GF_COX2': "nodeclassification",
    'GF_ENZYMES': "nodeclassification", 'GF_MUTAG': "nodeclassification",
}

graph2text_dict = {
    "FB15k237": "FB", "WN18RR": "WN", "CoDExMedium": "WK", "NELL995": "Self",
    "FB15k237Inductive:v1": "FB", "NELLInductive:v1": "Self", "WN18RRInductive:v1": "WN",
    "FB15k237Inductive:v2": "FB", "NELLInductive:v2": "Self", "WN18RRInductive:v2": "WN",
    "FB15k237Inductive:v3": "FB", "NELLInductive:v3": "Self", "WN18RRInductive:v3": "WN",
    "FB15k237Inductive:v4": "FB", "NELLInductive:v4": "Self", "WN18RRInductive:v4": "WN",
    "FBIngram:25": "FB", "FBIngram:50": "FB", "FBIngram:75": "FB", "FBIngram:100": "FB",
    "WKIngram:25": "WK", "WKIngram:50": "WK", "WKIngram:75": "WK", "WKIngram:100": "WK",
    "NLIngram:25": "Self", "NLIngram:50": "Self", "NLIngram:75": "Self", "NLIngram:100": "Self",
    "NLIngram:0": "Self",
}


def build_multilabels(test_triplets, label_rel_type=1):
    edge_mask = test_triplets[:, -1] == label_rel_type
    input_data = test_triplets[edge_mask]
    node_ids = input_data[:, 0].unique()
    label_ids = input_data[:, 1].unique()
    label_ids_sorted, _ = torch.sort(label_ids)
    node_id_to_index = {nid.item(): idx for idx, nid in enumerate(node_ids)}
    label_id_to_index = {lid.item(): idx for idx, lid in enumerate(label_ids_sorted)}
    node2triple_dict = {nid.item(): [] for idx, nid in enumerate(node_ids)}
    n = len(node_ids)
    m = len(label_ids_sorted)
    label_matrix = torch.zeros((n, m), dtype=torch.float)
    for row in input_data:
        node_id, label_id, value = row.tolist()
        i = node_id_to_index[node_id]
        j = label_id_to_index[label_id]
        label_matrix[i, j] = 1
        node2triple_dict[node_id].append(row.tolist())
    return label_matrix, node_ids, label_ids_sorted


@torch.no_grad()
def test_nc(cfg, model, test_data, filtered_data=None, graph_names=None, node_features=None, with_feature="none", mode="test", key_ratio=1, k=None, p=None):
    world_size = util.get_world_size()
    rank = util.get_rank()
    if util.get_rank() == 0:
        logger.warning("NC Evaluation With Feature Types: " + with_feature)
    all_metrics = []
    for test_graph, filters, graph_name, node_feature in zip(test_data, filtered_data, graph_names, node_features):
        metric_str = "\t %s: " % (graph_name)
        print(graph_name, test_graph.num_nodes, test_graph.num_relations, node_feature.shape if node_feature is not None else "None", test_graph.edge_type.shape)

        test_triplets = torch.cat([test_graph.target_edge_index, test_graph.target_edge_type.unsqueeze(0)]).t()
        target_edge_type = torch.unique(test_graph.target_edge_type).item()
        label_ids = torch.unique(test_graph.edge_index.T[test_graph.edge_type == target_edge_type][:, 1])
        print(label_ids)

        if "GF_LRGB_PEPFUN" in graph_name:
            label_matrix, node_ids, label_ids2 = build_multilabels(test_triplets, label_rel_type=target_edge_type)
            test_triplets = torch.stack([node_ids, torch.ones_like(node_ids) * label_ids2[0], torch.ones_like(node_ids) * target_edge_type], dim=1)
            print("rebuild test_triplets", test_triplets.shape)
            node_id_to_index = {nid.item(): idx for idx, nid in enumerate(node_ids)}

        sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)
        # Large GC graphs (PROTEINS 44K, DD 336K, COLLAB 377K nodes) can OOM in
        # message_and_aggregate. Auto-reduce batch for graphs exceeding a threshold.
        _eval_bs = cfg.train.batch_size
        if "large_graph_batch_size" in cfg.train and test_graph.num_nodes > cfg.train.get("large_graph_threshold", 30000):
            _eval_bs = cfg.train.large_graph_batch_size
        test_loader = torch_data.DataLoader(test_triplets, _eval_bs, sampler=sampler)

        print("build semantic graph, k, p", k, p)
        test_graph = graph_encoder.build_semantic_graph(test_graph, graph_name, with_feature, mode, node_feature=node_feature, k=k, p=p)
        if not hasattr(model, "global_mlp"):
            sem_edge_mask = test_graph.edge_type == 4
            test_graph.edge_type = test_graph.edge_type[sem_edge_mask]
            test_graph.edge_index = test_graph.edge_index[:, sem_edge_mask]

        print("edge_type", torch.unique(test_graph.edge_type), "target_edge_type", target_edge_type)

        model.eval()
        all_preds, all_labels = [], []
        ep_acc, ep_tot = 0, 0
        for batch in tqdm(test_loader, disable=disable_tqdm):
            t_batch, h_batch = tasks.label_negative(test_graph, batch, label_ids)
            if "MOE" in cfg.model["entity_model"]["class"]:
                t_pred, _ = model(test_graph, t_batch)
            else:
                t_pred = model(test_graph, t_batch)

            if "GF_LRGB_PEPFUN" in graph_name:
                batch_node_index = [node_id_to_index[nid.item()] for nid in batch[:, 0]]
                batch_labels = label_matrix[batch_node_index]
                pred_ids = t_pred.argmax(dim=1)
                pred_ls = label_ids[pred_ids]
                hit = np.array([batch_labels[i][l] == 1 for i, l in enumerate(pred_ids)]).sum()
                ep_acc += float(hit.item())
                ep_tot += pred_ids.shape[0]
                all_preds.append(t_pred)
                all_labels.append(batch_labels)
            else:
                pred_ids = t_pred.argmax(dim=1)
                pred_ls = label_ids[pred_ids]
                labels = batch[:, 1]
                hit = (labels == pred_ls).float().sum().item()
                ep_acc += hit
                ep_tot += labels.shape[0]
                all_preds.append(pred_ls)
                all_labels.append(labels)

            if len(all_labels) > 1000:
                print("end early")
                break

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        if world_size > 1:
            dist.all_reduce(all_preds, op=dist.ReduceOp.SUM)
            dist.all_reduce(all_labels, op=dist.ReduceOp.SUM)

        metrics = {}
        metrics['Graph'] = graph_name
        metrics['Acc'] = torch.Tensor([ep_acc / ep_tot])
        metric_str += "\t%s\t%.3f" % ('Acc', metrics['Acc'])

        if "GF_LRGB_PEPFUN" in graph_name:
            ap_value = average_precision_score(all_labels.cpu().numpy(), all_preds.cpu().numpy(), average='macro')
            metrics['AP'] = ap_value
            metric_str += "\t%s\t%.3f" % ('AP', metrics['AP'])
        else:
            f1_value = f1_score(all_labels.cpu().numpy(), all_preds.cpu().numpy(), average='macro')
            metrics['F1'] = torch.Tensor([f1_value])
            metric_str += "\t%s\t%.3f" % ('F1', metrics['F1'])

        if rank == 0:
            logger.warning(metric_str)
        all_metrics.append(metrics)

    return all_metrics


@torch.no_grad()
def test_gc(cfg, model, test_data, filtered_data=None, graph_names=None, node_features=None, with_feature="none", mode="test", k=None, p=None, flat_mode=False):
    """Graph Classification evaluation using SCR's graph-level reasoning.

    Args:
        flat_mode: If True, use entity-level prediction (no build_global).
                   Used for datasets where flat model outperforms meta model (e.g., GF_COX2).
    """
    world_size = util.get_world_size()
    rank = util.get_rank()
    mode_str = "flat (entity-level)" if flat_mode else f"meta(k={k}, p={p})"
    if util.get_rank() == 0:
        logger.warning("GC Evaluation (graph-level) [%s] With Feature Types: %s" % (mode_str, with_feature))
    all_metrics = []
    for test_graph, filters, graph_name, node_feature in zip(test_data, filtered_data, graph_names, node_features):
        metric_str = "\t %s: " % (graph_name)
        print(graph_name, test_graph.num_nodes, test_graph.num_relations,
              node_feature.shape if node_feature is not None else "None", test_graph.edge_type.shape)

        test_triplets = torch.cat([test_graph.target_edge_index, test_graph.target_edge_type.unsqueeze(0)]).t()
        sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)
        # Large GC graphs (PROTEINS 44K, DD 336K, COLLAB 377K nodes) can OOM in
        # message_and_aggregate. Auto-reduce batch for graphs exceeding a threshold.
        _eval_bs = cfg.train.batch_size
        if "large_graph_batch_size" in cfg.train and test_graph.num_nodes > cfg.train.get("large_graph_threshold", 30000):
            _eval_bs = cfg.train.large_graph_batch_size
        test_loader = torch_data.DataLoader(test_triplets, _eval_bs, sampler=sampler)

        label_ids = torch.unique(test_graph.edge_index.T[test_graph.edge_type == torch.unique(test_graph.target_edge_type)][:, 1])
        print("label_ids:", label_ids)

        # Build semantic graph for individual dataset (use defaults k=3, p=0.8)
        test_graph = graph_encoder.build_semantic_graph(test_graph, graph_name, with_feature, mode,
                                                         node_feature=node_feature, target_mask=None, ifload=False)

        print("edge_type after build_semantic_graph:", torch.unique(test_graph.edge_type))

        model.eval()
        # Reset meta-graph cache only for meta mode (GC)
        if not flat_mode and hasattr(model, 'second_graph'):
            model.second_graph = None
            model.highlevel_embeds = None
            model.unique_elements = None

        all_preds, all_labels = [], []
        ep_acc, ep_tot = 0, 0
        for batch in tqdm(test_loader, disable=disable_tqdm):
            t_batch, h_batch = tasks.label_negative(test_graph, batch, label_ids)
            if flat_mode:
                t_pred = model(test_graph, t_batch)
            else:
                # CRITICAL: SCR.forward uses "if k is not None" to enter GC meta-graph path.
                # When GC_BEST_KP returns (None, None) meaning "use defaults", we must
                # substitute build_global's defaults (k=5, p=0.3) so the guard triggers.
                # Otherwise k=None causes fallback to NC/flat path (the regression bug).
                _gc_k = k if k is not None else 5
                _gc_p = p if p is not None else 0.3
                t_pred = model(test_graph, t_batch, k=_gc_k, p=_gc_p)
            if isinstance(t_pred, tuple):
                t_pred = t_pred[0]
            pred_ids = t_pred.argmax(dim=1)
            pred_ls = label_ids[pred_ids]
            labels = batch[:, 1]

            hit = (labels == pred_ls).float().sum().item()
            ep_acc += hit
            ep_tot += labels.shape[0]
            all_preds.append(pred_ls)
            all_labels.append(labels)

            if len(all_labels) > 1000:
                print("end early")
                break

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
        all_metrics.append(metrics)

    return all_metrics


def load_generaleval_graph(cfg, ds_list, eval_graph_data, mode="transductive", data_type="graph"):
    for graph in ds_list:
        ds, version = graph.split(":") if ":" in graph else (graph, None)
        vars = util.detect_variables(args.config)
        parser = argparse.ArgumentParser()
        for var in vars:
            parser.add_argument("--%s" % var)
        vars = parser.parse_known_args()[0]
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
        # Use the absolute root saved before chdir (not os.path.abspath which
        # would resolve relative to the wrong CWD after os.chdir(output/))
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
def test(cfg, model, test_data, filtered_data=None, graph_names=None, with_feature="none", drop_ratio=0, k=None, p=None, mode="test"):
    world_size = util.get_world_size()
    rank = util.get_rank()
    if util.get_rank() == 0:
        logger.warning("KG Evaluation With Feature Types: %s Drop Ratio: %.2f" % (with_feature, drop_ratio))
    all_metrics = []
    for test_graph, filters, graph_name in zip(test_data, filtered_data, graph_names):
        metric_str = "\t %s: " % (graph_name)
        test_triplets = torch.cat([test_graph.target_edge_index, test_graph.target_edge_type.unsqueeze(0)]).t()
        sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)
        # Large GC graphs (PROTEINS 44K, DD 336K, COLLAB 377K nodes) can OOM in
        # message_and_aggregate. Auto-reduce batch for graphs exceeding a threshold.
        _eval_bs = cfg.train.batch_size
        if "large_graph_batch_size" in cfg.train and test_graph.num_nodes > cfg.train.get("large_graph_threshold", 30000):
            _eval_bs = cfg.train.large_graph_batch_size
        test_loader = torch_data.DataLoader(test_triplets, _eval_bs, sampler=sampler)

        if drop_ratio > 0:
            test_graph = tasks.drop_edges(copy.deepcopy(test_graph), drop_ratio)
        test_graph = graph_encoder.build_semantic_graph(test_graph, graph_name, with_feature, mode, ifload=(drop_ratio == 0), k=k, p=p)

        model.eval()
        rankings = []
        num_negatives = []
        for batch in test_loader:
            t_batch, h_batch = tasks.all_negative(test_graph, batch)
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

        metrics = {}
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
                    fp_rate = (all_ranking - 1).float() / all_num_negative
                    score = 0
                    for i in range(threshold):
                        num_comb = math.factorial(num_sample - 1) / \
                                   math.factorial(i) / math.factorial(num_sample - i - 1)
                        score += num_comb * (fp_rate ** i) * ((1 - fp_rate) ** (num_sample - i - 1))
                    score = score.mean()
                else:
                    score = (all_ranking <= threshold).float().mean()
            metric_str += "\t%s\t%.3f" % (metric, score)
            metrics[metric] = score
        mrr = (1 / all_ranking.float()).mean()

        all_metrics.append(mrr)
        if rank == 0:
            logger.warning(metric_str)

        metrics = {k: v.item() for k, v in metrics.items()}
        metrics['semtype'] = with_feature
        metrics['dataset'] = graph_name
        with open(results_file, "a", newline='') as csv_file:
            fieldnames = ['dataset', 'semtype'] + list(metrics.keys())[:-1]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=',')
            if csv_file.tell() == 0:
                writer.writeheader()
            writer.writerow(metrics)

    avg_metric = sum(all_metrics) / max(1, len(all_metrics))
    return avg_metric


def load_eval_graph(cfg, ds_list, eval_graph_data, mode="transductive"):
    for graph in ds_list:
        ds, version = graph.split(":") if ":" in graph else (graph, None)
        vars = util.detect_variables(args.config)
        parser = argparse.ArgumentParser()
        for var in vars:
            parser.add_argument("--%s" % var)
        vars = parser.parse_known_args()[0]
        vars = {k: util.literal_eval(v) for k, v in vars._get_kwargs()}
        epochs, batch_per_epoch = 0, 'null'
        vars['epochs'] = epochs
        vars['bpe'] = batch_per_epoch
        vars['dataset'] = ds
        if version is not None:
            vars['version'] = version
        cfg = util.load_config(args.config, context=vars)
        cfg.dataset['class'] = ds
        if version is not None and ds not in ["FB15k237", "WN18RR", "CoDExMedium", "CoDExSmall"]:
            cfg.dataset['version'] = version
        cfg.dataset.pop('graphs')
        cfg.dataset.root = _abs_dataset_root

        task_name = cfg.task["name"]
        dataset = util.build_dataset(cfg)
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
            if "ILPC" in cfg.dataset['class'] or "Ingram" in cfg.dataset['class']:
                full_inference_edges = torch.cat([valid_data.edge_index, valid_data.target_edge_index, test_data.target_edge_index], dim=1)
                full_inference_etypes = torch.cat([valid_data.edge_type, valid_data.target_edge_type, test_data.target_edge_type])
                test_filtered_data = Data(edge_index=full_inference_edges, edge_type=full_inference_etypes, num_nodes=test_data.num_nodes)
                val_filtered_data = test_filtered_data
            else:
                full_inference_edges = torch.cat([test_data.edge_index, test_data.target_edge_index], dim=1)
                full_inference_etypes = torch.cat([test_data.edge_type, test_data.target_edge_type])
                test_filtered_data = Data(edge_index=full_inference_edges, edge_type=full_inference_etypes, num_nodes=test_data.num_nodes)
                val_filtered_data = Data(
                    edge_index=torch.cat([train_data.edge_index, valid_data.target_edge_index], dim=1),
                    edge_type=torch.cat([train_data.edge_type, valid_data.target_edge_type])
                )
        else:
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
    return eval_graph_data


def _split_nodegraph_data(eval_nodegraph_data, mode="test"):
    """Split eval_nodegraph_data into NC datasets (data_type='node') and GC datasets (data_type='graph')."""
    nc_indices = [i for i, dt in enumerate(eval_nodegraph_data["data_type"]) if dt == "node"]
    gc_indices = [i for i, dt in enumerate(eval_nodegraph_data["data_type"]) if dt == "graph"]

    def _slice_data(data_dict, indices):
        if not indices:
            return None
        return {key: [data_dict[key][i] for i in indices] for key in data_dict}

    nc_data = _slice_data(eval_nodegraph_data, nc_indices)
    gc_data = _slice_data(eval_nodegraph_data, gc_indices)
    return nc_data, gc_data


def _get_best_nc_kp(graph_name):
    """Get the hardcoded best k/p for a NC dataset. Returns (k, p) — None means use defaults."""
    return NC_BEST_KP.get(graph_name, (None, None))


def _get_best_gc_kp(graph_name):
    """Get the hardcoded best k/p for a GC dataset's meta-graph. Returns (k, p) — None means use defaults."""
    return GC_BEST_KP.get(graph_name, (None, None))


def global_test(cfg, model, eval_data_dict, gc_model=None, mode="test"):
    eval_hetelink_data = eval_data_dict["eval_hetelink_data"]
    eval_nodegraph_data = eval_data_dict["eval_nodegraph_data"]
    eval_homolink_data = eval_data_dict["eval_homolink_data"]

    for feature in ["bert"]:
        # KG link prediction evaluation
        test(cfg, model, eval_hetelink_data[mode + '_data'], filtered_data=eval_hetelink_data[mode + "_filter"],
             graph_names=eval_hetelink_data["graphs"], with_feature=feature, mode=mode)

        # Split NC and GC datasets
        nc_data, gc_data = _split_nodegraph_data(eval_nodegraph_data, mode=mode)

        # Node Classification — each dataset gets its own hardcoded best k/p
        if nc_data is not None and len(nc_data["graphs"]) > 0:
            # Group datasets by (k, p) to batch them efficiently
            kp_groups = {}
            for idx, graph_name in enumerate(nc_data["graphs"]):
                k, p = _get_best_nc_kp(graph_name)
                key = (k, p)
                if key not in kp_groups:
                    kp_groups[key] = []
                kp_groups[key].append(idx)

            for (k, p), indices in kp_groups.items():
                sub_data = {key: [nc_data[key][i] for i in indices] for key in nc_data}
                test_nc(cfg, model, sub_data[mode + '_data'], filtered_data=sub_data[mode + "_filter"],
                        graph_names=sub_data["graphs"], node_features=sub_data["node_feature"],
                        with_feature="bert", k=k, p=p)

        # Graph Classification — each dataset gets its own hardcoded best k/p for meta-graph
        if gc_data is not None and len(gc_data["graphs"]) > 0:
            if gc_model is not None:
                # Evaluate each GC dataset with its own best k/p
                for idx, graph_name in enumerate(gc_data["graphs"]):
                    gc_k, gc_p = _get_best_gc_kp(graph_name)
                    single_data = {key: [gc_data[key][idx]] for key in gc_data}

                    # Adaptive routing: use flat model for datasets where it outperforms meta
                    if graph_name in GC_FLAT_DATASETS:
                        logger.warning(f"GC: {graph_name} using FLAT mode (entity-level, no meta-graph)")
                        test_gc(cfg, model, single_data[mode + '_data'],
                                filtered_data=single_data[mode + "_filter"],
                                graph_names=single_data["graphs"],
                                node_features=single_data["node_feature"],
                                with_feature="bert", flat_mode=True)
                    else:
                        # Resolve None to build_global defaults for logging clarity
                        _log_k = gc_k if gc_k is not None else 5
                        _log_p = gc_p if gc_p is not None else 0.3
                        logger.warning(f"GC: {graph_name} using META mode k={_log_k}, p={_log_p}")
                        test_gc(cfg, gc_model, single_data[mode + '_data'],
                                filtered_data=single_data[mode + "_filter"],
                                graph_names=single_data["graphs"],
                                node_features=single_data["node_feature"],
                                with_feature="bert", k=gc_k, p=gc_p)
            else:
                logger.warning("WARNING: No model provided for GC, falling back to test_nc")
                test_nc(cfg, model, gc_data[mode + '_data'], filtered_data=gc_data[mode + "_filter"],
                        graph_names=gc_data["graphs"], node_features=gc_data["node_feature"],
                        with_feature="bert")


if __name__ == "__main__":
    seeds = [1024, 42, 1337, 512, 256]
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file",
                        default=os.path.join(os.path.dirname(__file__), "..", "config", "eval_all.yaml"))
    parser.add_argument("-g", "--gpu", help="GPU device id", default=0, type=int)
    parser.add_argument("--ckpt", help="checkpoint to evaluate (overrides cfg.checkpoint, e.g. a trained model_epoch_8.pth)", default=None)

    args, unparsed = parser.parse_known_args()
    print("gpu:", args.gpu)
    torch.cuda.set_device(args.gpu)

    # Resolve config path to absolute before any chdir
    args.config = os.path.abspath(args.config)

    # Resolve project root
    _project_dir = os.path.dirname(os.path.dirname(args.config))
    while not os.path.isdir(os.path.join(_project_dir, "script")):
        parent = os.path.dirname(_project_dir)
        if parent == _project_dir:
            break
        _project_dir = parent
    _script_dir = os.path.join(_project_dir, "script")

    path = os.path.dirname(os.path.expanduser(args.config))
    results_file = os.path.join(path, f"../results/ultra_results_{time.strftime('%Y-%m-%d-%H-%M-%S')}.csv")
    seed = seeds[0] if 0 < len(seeds) else random.randint(0, 10000)

    # Semantic mode is fixed to "end"
    sem_mode = "end"

    cfg = util.load_config(args.config)

    # Allow overriding the checkpoint from the CLI (e.g. a freshly trained model_epoch_8.pth)
    if args.ckpt is not None:
        cfg.checkpoint = args.ckpt

    # Resolve all relative paths to absolute before any chdir
    if "checkpoint" in cfg and cfg.checkpoint is not None:
        cfg.checkpoint = os.path.abspath(cfg.checkpoint)
    if "root" in cfg.dataset:
        cfg.dataset.root = os.path.abspath(cfg.dataset.root)

    # Save absolute dataset root for reuse after chdir
    # (load_eval_graph / load_generaleval_graph re-load config and would
    #  resolve relative paths against the wrong CWD)
    _abs_dataset_root = cfg.dataset.root

    # Fix gpus: parse string "[0]" → list [0]
    if isinstance(cfg.train.gpus, str):
        cfg.train.gpus = eval(cfg.train.gpus)

    root_dir = os.path.expanduser(cfg.output_dir)
    os.makedirs(root_dir, exist_ok=True)
    os.chdir(root_dir)

    working_dir = util.create_working_directory(cfg)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    logger = util.get_root_logger()
    if util.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))

    print(cfg.train.gpus)
    task_name = cfg.task["name"]
    device = util.get_device(cfg)

    # Auto-detect resource path
    resource_path = os.path.join(_project_dir, "resource") + os.sep
    logger.warning("Resource path: %s" % resource_path)
    graph_encoder = GraphEncoding(cfg.model["entity_model"]["feat_dim"], resource_path, graph2text_dict, k=3, p=0.8)

    eval_hetelink_data = {"graphs": [], "valid_data": [], "test_data": [], "valid_filter": [], "test_filter": []}
    eval_hetelink_data = load_eval_graph(cfg, cfg.moreeval["evalgraphs"], eval_hetelink_data, mode="transductive")
    eval_hetelink_data = load_eval_graph(cfg, cfg.moreeval["ind_evalgraphs"], eval_hetelink_data, mode="inductive")

    eval_nodegraph_data = {"graphs": [], "valid_data": [], "test_data": [], "valid_filter": [], "test_filter": [], "node_feature": [], "data_type": []}
    eval_nodegraph_data = load_generaleval_graph(cfg, cfg.moreeval["nodeevalgraphs"], eval_nodegraph_data, data_type="node")
    eval_nodegraph_data = load_generaleval_graph(cfg, cfg.moreeval["graphevalgraphs"], eval_nodegraph_data, data_type="graph")

    eval_homolink_data = {"graphs": [], "valid_data": [], "test_data": [], "valid_filter": [], "test_filter": []}
    eval_homolink_data = load_eval_graph(cfg, cfg.moreeval["homolinkgraphs"], eval_homolink_data, mode="transductive")

    eval_data_dict = {"eval_hetelink_data": eval_hetelink_data, "eval_nodegraph_data": eval_nodegraph_data, "eval_homolink_data": eval_homolink_data}

    cfg.model['entity_model']['feat_type'] = sem_mode
    model = SCR(rel_model_cfg=cfg.model.relation_model, entity_model_cfg=cfg.model.entity_model)

    if "checkpoint" in cfg and cfg.checkpoint is not None:
        print("feat_type", model.entity_model.feat_type, sem_mode, cfg.checkpoint)
        state = torch.load(cfg.checkpoint, map_location="cpu")
        model.load_state_dict(state["model"], strict=False)
    model = model.to(device)

    # Run evaluation with hardcoded per-dataset k/p
    # GC evaluation uses the same SCR model (build_global is built-in)
    global_test(cfg, model, eval_data_dict, gc_model=model, mode="test")
