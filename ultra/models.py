import torch
import os, copy
import faiss
from torch import nn
from . import tasks, layers
from ultra.base_nbfnet import BaseNBFNet
import torch_geometric.utils as pygutils
from torch_geometric.data import Data
from torch_scatter import scatter
import numpy as np
from scipy.sparse import csr_matrix
import wordninja
import logging
from ultra.tasks import build_relation_graph, build_relation_types
from transformers import AutoTokenizer, AutoModel

class FeedForwardLayer(nn.Module):
    def __init__(self, in_feat, out_feat, bias=True, act=None):
        super(FeedForwardLayer, self).__init__()
        self.linear = nn.Linear(in_feat, out_feat, bias=bias)#, dtype=t.bfloat16)
        if act == 'identity' or act is None:
            self.act = None
        elif act == 'leaky':
            self.act = nn.LeakyReLU(negative_slope=0.5)
        elif act == 'relu':
            self.act = nn.ReLU()
        elif act == 'relu6':
            self.act = nn.ReLU6()
        else:
            raise Exception('Error')
    
    def forward(self, embeds):
        if self.act is None:
            return self.linear(embeds)
        return self.act(self.linear(embeds))
    
class MLP(nn.Module):
    def __init__(self, hidden_dims, layer_num):
        super(MLP, self).__init__()
        self.layer_num = layer_num
        self.dense_layers = nn.Sequential(*[FeedForwardLayer(hidden_dims[i], hidden_dims[i+1], bias=True, act="relu") for i in range(layer_num)])
        self.layer_norms = nn.Sequential(*[nn.LayerNorm(hidden_dims[i+1], elementwise_affine=True) for i in range(layer_num)])
        self.dropout = nn.Dropout(p=0.1)
    
    def forward(self, embeds):
        for i in range(self.layer_num):
            embeds = self.layer_norms[i](self.dropout(self.dense_layers[i](embeds)) + embeds)
        return embeds

class TextEncoding(nn.Module): 
    def __init__(self, model_name=None):
        super(TextEncoding, self).__init__()
        self.mlp = None
        if model_name is None:
            model_name = "sentence-transformers/bert-base-nli-mean-tokens"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sentenceBert = AutoModel.from_pretrained(model_name)
        self.sem_dim = 768
    
    def load_sentence_from_file(self, file_path, inv_rel_vocab):
        relation2id_dict = {k: v for k, v in inv_rel_vocab}
        rel_text_list = [0] * len(relation2id_dict)
        with open(file_path + ".txt", "r") as f:
            for line in f.readlines():
                rel, text = line.replace("\n", "").split("\t")
                if rel not in relation2id_dict: continue
                rel_text_list[relation2id_dict[rel]] = text
         # print("rel_text_list:", len(rel_text_list), rel_text_list)
        return rel_text_list

    def build_semantic_embeddings(self, sentence_list):
        with torch.no_grad():
            sem_embedding_list = []
            for sentence in sentence_list:
                sentence = sentence.strip().replace("/"," / ")
                encoded_input = self.tokenizer(sentence, padding=True, truncation=True, max_length=128, return_tensors='pt')
                model_output = self.sentenceBert(**encoded_input)
                embeddings = model_output.last_hidden_state.mean(dim=1)
                sem_embedding_list.append(embeddings)
        if len(sem_embedding_list) == 0:
            return None
        else:
            return torch.stack(sem_embedding_list, dim=0).squeeze(1).float()

    def load_item_text(self, name, path, mode="entity"):
        item2text_dict = {}
        with open(path + name +"-"+mode+"2text.txt", "r", encoding="utf-8") as f:
            for line in f.readlines():
                items = line.replace("\n", "").split("\t") 
                item = items[0]
                text = " ".join(items[1:])
                item2text_dict[item] = text
        return item2text_dict 
    
    def generate_sem_embeddings(self, graph_data, graph_name, graph2text_dict, data_path = "../resource/", mode="entity"):
        if False: #os.path.exists(data_path + graph_name + mode + "_embeds.pth"):
            total_embeds = torch.load(data_path + graph_name + mode + "_embeds.pth")
        else:
            text_list, id_list = [], []
            item_type = "entity" if "entity" in mode else "relation"
            inv_item_vocab = graph_data.inv_ent_vocab if "entity" in mode else graph_data.inv_rel_vocab
            if graph2text_dict[graph_name] != "Self":
                item2text_dict = self.load_item_text(graph2text_dict[graph_name], data_path, item_type)
                for item_name, item_id in inv_item_vocab:
                    if item_name in item2text_dict:
                        text_list.append(item2text_dict[item_name])
                        id_list.append(item_id)
                        if "relation" in mode:
                            text_list.append(item2text_dict[item_name] + "_reverse")
                            id_list.append(item_id + len(inv_item_vocab))
            else:
                for item_name, item_id in inv_item_vocab:
                    item_text = " ".join(wordninja.split(item_name))
                    text_list.append(item_text)
                    id_list.append(item_id)
            print(len(inv_item_vocab), len(text_list))
            sem_embeddings = self.build_semantic_embeddings(text_list)
            if sem_embeddings is None: return None
            sem_index = torch.LongTensor(id_list)
            item_length = len(inv_item_vocab) if "entity" in mode else len(inv_item_vocab) * 2
            total_embeds = torch.zeros(item_length, sem_embeddings.size(1))
            total_embeds[sem_index] = sem_embeddings
            torch.save(total_embeds, data_path + graph_name + mode + "_embeds.pth")
        return total_embeds

class GraphEncoding(nn.Module): 
    def __init__(self, hidden_dim, resource_path, graph2text_dict, k=10, p=0.5):
        super(GraphEncoding, self).__init__()
        self.hidden_dim = hidden_dim
        self.resource_path = resource_path
        self.graph2text_dict = graph2text_dict
        self.k = k # top k number
        self.p = p # threshold for topk edge
        self.semantic_model_dict = {"bert": "sentence-transformers/bert-base-nli-mean-tokens",
                                    "sbbert": "sentence-transformers/all-mpnet-base-v2",
                                    "minilm": "sentence-transformers/paraphrase-MiniLM-L6-v2",
                                    "distilbert": "sentence-transformers/distiluse-base-multilingual-cased-v1"}
        self.glove_embedding_path = "glove.2024.wikigiga.50d/wiki_giga_2024_50_MFT20_vectors_seed_123_alpha_0.75_eta_0.075_combined.txt"

    
    def semantic_types(self):
        return list(self.semantic_model_dict.keys())+["top"]

    def svd_proj(self, adj, latdim):
        q = latdim
        niter = 2
        cache_proj = True
        if latdim > adj.shape[0] or latdim > adj.shape[1]:
            dim = min(adj.shape[0], adj.shape[1])
            svd_u, s, svd_v = torch.svd_lowrank(adj, q=dim, niter=niter)
            svd_u = torch.concat([svd_u, torch.zeros([svd_u.shape[0], latdim - dim]).to(adj.device)], dim=1)
            svd_v = torch.concat([svd_v, torch.zeros([svd_v.shape[0], latdim - dim]).to(adj.device)], dim=1)
            s = torch.concat([s, torch.zeros(latdim - dim).to(adj.device)])
        else:
            svd_u, s, svd_v = torch.svd_lowrank(adj, q=q, niter=niter)
        svd_u = svd_u @ torch.diag(torch.sqrt(s))
        svd_v = svd_v @ torch.diag(torch.sqrt(s))
        if adj.shape[0] != adj.shape[1]:
            projection = torch.concat([svd_u, svd_v], dim=0)
        else:
            projection = svd_u + svd_v
        if not cache_proj:
            projection = projection.cpu()
        return projection

    def svd_proj_2(self, feats, latdim, niter=2):
        cache_proj = True
        if latdim > feats.shape[0] or latdim > feats.shape[1]:
            dim = min(feats.shape[0], feats.shape[1])
            decom_feats, s, decom_featdim = torch.svd_lowrank(feats, q=dim, niter=niter)
            decom_feats = torch.concat([decom_feats, torch.zeros([decom_feats.shape[0], latdim-dim]).to(feats.device)], dim=1)
            s = torch.concat([s, torch.zeros(latdim - dim).to(feats.device)])
        else:
            decom_feats, s, decom_featdim = torch.svd_lowrank(feats, q=latdim, niter=niter)
        decom_feats = decom_feats @ torch.diag(torch.sqrt(s))
        if not cache_proj:
            decom_feats = decom_feats.cpu()
        # print(feats.shape, decom_feats.shape, latdim)
        return decom_feats

    def build_kg_adj(self, graph):
        encoding_values = torch.linspace(0, 1, (graph.num_relations.item() // 2)+1)
        encoding_values = torch.cat([encoding_values, encoding_values*-1],dim=0).to(graph.edge_type.device)
        adj = torch.zeros((graph.num_nodes, graph.num_nodes)).float().to(graph.edge_type.device)
        encoded_edge_type = encoding_values[graph.edge_type]
        adj.index_put_((graph.edge_index[0], graph.edge_index[1]), encoded_edge_type, accumulate=True)
        # adj[graph.edge_index[0], graph.edge_index[1]] = encoded_edge_type
        return adj
    
    def preprocess_semantics(self, single_graph, data_name, node_feature = None, ifload=True, semantic_type="bert"):
        n_rel = single_graph.num_relations + 1
        n_ent = single_graph.num_nodes
        device = single_graph.edge_index.device
        if data_name not in self.graph2text_dict: 
            semantic_type = "top"
        embedfile_path= self.resource_path+data_name+'_'+semantic_type+'_entembeds.pth'
        if os.path.exists(embedfile_path) and ifload:
            ent_embeds = torch.load(embedfile_path)
        else:
            if semantic_type in self.semantic_model_dict and data_name in self.graph2text_dict:
                model_name = self.semantic_model_dict[semantic_type]
                text_encoder = TextEncoding(model_name)
                ent_embeds = text_encoder.generate_sem_embeddings(single_graph, data_name, self.graph2text_dict, data_path=self.resource_path, mode="entity")
                ent_embeds[ent_embeds.sum(1)==0] = torch.mean(ent_embeds, dim=0)
            elif semantic_type == "glove50d":
                glove_path = self.resource_path+self.glove_embedding_path
                text_encoder = TextEncoding(None)
                embed_dict, embed_dim = text_encoder.load_glove_embeddings(glove_path)
                ent_embeds = text_encoder.get_glove_embeddings(single_graph, data_name, self.graph2text_dict, embed_dict, embed_dim, data_path=self.resource_path, mode="entity")
                ent_embeds[ent_embeds.sum(1)==0] = torch.mean(ent_embeds, dim=0)
            elif node_feature is not None: # semantic_type != "top"
                # print(semantic_type, node_feature.shape)
                ent_embeds = torch.Tensor(node_feature).to(device)  
                ent_embeds[ent_embeds.sum(1)==0] = torch.mean(ent_embeds, dim=0)
            else:
                node_rel_matrix = torch.zeros((n_ent, n_rel), dtype=torch.int32).to(device)
                index_tensor = torch.stack([single_graph.edge_index[0], single_graph.edge_type], dim=1).long()
                node_rel_matrix.index_put_((index_tensor[:,0], index_tensor[:,1]), torch.ones(len(index_tensor), dtype=torch.int32).to(device), accumulate=True)
                ent_embeds = node_rel_matrix.float()
            torch.save(ent_embeds, embedfile_path)
        ent_embeds = ent_embeds.to(device)
        return ent_embeds

    def build_faiss(self, graph, node_embeds, target_mask=None, k=10, p=0.5):
        np_node_embeds = node_embeds.cpu().numpy()
        norm = np.linalg.norm(np_node_embeds, axis=1, keepdims=True)
        norm[norm == 0] = 1
        np_node_embeds = np_node_embeds / norm
        # target_mask: for graph task, label -1, train node 1, test node 0
        
        if target_mask is not None:
            index_embeds = np.zeros_like(np_node_embeds)
            index_mask = target_mask.cpu().numpy() == 1
            index_embeds[index_mask] = np_node_embeds[index_mask]
        else:
            index_embeds = np_node_embeds
        index = faiss.IndexFlatIP(index_embeds.shape[1]) 
        index.add(index_embeds) 
        distances, indices = index.search(np_node_embeds, k*10)
        # 增加对1阶邻居的过滤，验证distance范围
        pairs = []
        topk_indices = [[] for i in range(graph.num_nodes)]
        for i, neighbors in enumerate(indices):
            if target_mask is not None and target_mask[i] != 0: continue
            filtered_neighbors = [] 
            for j, nid in enumerate(neighbors):
                if len(filtered_neighbors) >= k: break
                if target_mask is not None and target_mask[nid] != 1: continue
                if i != nid and nid != -1 and distances[i][j] > p:
                    pairs.append((i, nid))
                    filtered_neighbors.append(nid)
            # if target_mask is not None:
            #     print(i, target_mask[i], len(filtered_neighbors), [target_mask[k] for k in filtered_neighbors])
            topk_indices[i] = filtered_neighbors 

        if len(pairs) == 0: return None, None, topk_indices
        triples_tensor = torch.tensor(pairs).long()
        triples_tensor = torch.stack([triples_tensor[:,0], torch.zeros_like(triples_tensor[:,0]), triples_tensor[:,1]], dim=1)
        triples_tensor2 = torch.stack([triples_tensor[:,1], torch.ones_like(triples_tensor[:,0]), triples_tensor[:,0]], dim=1)
        triples_tensor = torch.cat([triples_tensor, triples_tensor2], dim=0).to(node_embeds.device)
        return node_embeds, triples_tensor, topk_indices
    
    def core_process(self, graph, feats, target_mask, p, k):
        if feats.shape[-1] < self.hidden_dim:
            ent_embeds = torch.zeros((feats.shape[0], self.hidden_dim))
            ent_embeds[:,:feats.shape[-1]] = feats
        if feats.shape[-1] != self.hidden_dim:
            ent_embeds = self.svd_proj_2(feats, self.hidden_dim)[:graph.num_nodes]
        else:
            ent_embeds = feats[:graph.num_nodes]
        # ent_embeds = self.svd_proj_4(feats, self.hidden_dim)[:graph.num_nodes] # 每次都重算太慢了
        node_embeds, triples_tensor, topk_indices = self.build_faiss(graph, ent_embeds, target_mask, k=k, p=p)
        triples_tensor.to(graph.edge_index.device)
        ex_graph = copy.deepcopy(graph)
        ex_graph.edge_index = torch.cat([ex_graph.edge_index, triples_tensor[:,(0,2)].T], dim=1)
        ex_graph.edge_type = torch.cat([ex_graph.edge_type, triples_tensor[:,1]+ex_graph.num_relations], dim=0)
        ex_graph.num_relations += 2
        ex_graph = build_relation_graph(ex_graph)
        return ex_graph, node_embeds, topk_indices

    def build_semantic_graph(self, single_graph, data_name, semantic_type="bert", mode="train", node_feature = None, target_mask=None, ifload=True, k=None, p=None):
        if k is None: k = self.k
        if p is None: p = self.p
        device = single_graph.edge_type.device
        if semantic_type=="none": # resume the original graph
            single_graph.ent_embeds = None
            single_graph.topk_indices = None
            single_graph.ex_relation_graph = single_graph.relation_graph
            return single_graph
        graphfile_path = self.resource_path+data_name+'_'+mode+'_'+semantic_type+"_"+str(k)+"_"+str(p)+'_graph_data.pth'
        # ifload = False
        if ifload and os.path.exists(graphfile_path): # 稀疏结果对应的预算图不加载
            single_graph = torch.load(graphfile_path, weights_only=False)
        else:
            # print("start semantic_graph", semantic_type)
            ent_embeds = self.preprocess_semantics(single_graph, data_name, semantic_type=semantic_type, ifload=ifload, node_feature=node_feature)

            ex_graph, node_embeds, topk_indices = self.core_process(single_graph, ent_embeds, target_mask, p, k)

            single_graph.ent_embeds = node_embeds
            single_graph.topk_indices = topk_indices
            single_graph.ex_edge_index = ex_graph.edge_index
            single_graph.ex_edge_type = ex_graph.edge_type
            single_graph.ex_num_relations = ex_graph.num_relations  
            single_graph.ex_relation_graph = copy.deepcopy(ex_graph.relation_graph)
            if ifload: torch.save(single_graph, graphfile_path)
            del ex_graph
        single_graph = single_graph.to(device)
        return single_graph


class SCR(nn.Module):
    def __init__(self, rel_model_cfg, entity_model_cfg):
        super(SCR, self).__init__()
        self.with_feats = entity_model_cfg["with_feats"]
        self.rel_model_class = rel_model_cfg["class"]
        self.ent_model_class = entity_model_cfg["class"]
        self.feat_dim = entity_model_cfg['feat_dim']
        if 'ablation_type' in entity_model_cfg:
            self.ablation_type = entity_model_cfg['ablation_type']
        else:
            self.ablation_type = "none"

        if self.rel_model_class == "NBFNet":
            self.relation_model = RelNBFNet(**rel_model_cfg)
        elif self.rel_model_class == "TGFNet":
            self.relation_model = RelTGFNet(**rel_model_cfg)
        elif self.rel_model_class == "OGMNet":
            self.relation_model = RelOGMNet(**rel_model_cfg)

        if self.ent_model_class == "IndNBFNet": # FirstFusion Features
            self.entity_model = EntityNBFNet(**entity_model_cfg)
        elif self.ent_model_class == "MOENet":
            self.entity_model = EntityMOENet(**entity_model_cfg)
        elif self.ent_model_class == "GRPNet":
            self.entity_model = EntityMOENet(**entity_model_cfg)
        self.global_mlp = MLP([self.feat_dim, self.feat_dim, self.feat_dim], 2)
        self.entity_model2 = EntityNBFNet(**entity_model_cfg)

        # GC (graph classification) meta-graph components
        self.graph_encoder_gc = GraphEncoding(self.feat_dim, "", {}, k=3, p=0.8)
        self.second_graph = None
        self.highlevel_embeds = None
        self.unique_elements = None

    def build_global(self, data, k=None, p=None):
        """Build meta-graph for graph-level classification."""
        if k is None: k=5
        if p is None: p=0.3
        globalrel_embeds = self.relation_model(data.ex_relation_graph, query=None)
        globalent_embeds= self.entity_model.global_forward(data, globalrel_embeds, None).squeeze(0)
        globalent_embeds = self.global_mlp(globalent_embeds)

        target_relation_id = data.target_edge_type.unique()
        label_ids = torch.unique(data.edge_index.T[data.edge_type==torch.unique(data.target_edge_type)][:,1])
        edge_mask = (data.edge_type == target_relation_id) | (data.edge_type == target_relation_id+data.num_relations//2)
        highlevel_edge_index = data.edge_index[:, edge_mask]
        highlevel_edge_type = data.edge_type[edge_mask]

        # build highlevel_graphs
        highlevel_triples = torch.cat([highlevel_edge_index,data.target_edge_index], dim=1)
        unique_elements, inverse_indices = torch.unique(highlevel_triples, return_inverse=True)
        reindexed_index = inverse_indices.view(highlevel_triples.size())
        reindexed_edge_index = reindexed_index[:, :highlevel_edge_index.size(1)]
        reindexed_target_index = reindexed_index[:, highlevel_edge_index.size(1):]
        reindexed_edge_type = (highlevel_edge_type != target_relation_id).long()
        highlevel_embeds = globalent_embeds[unique_elements]
        reindexed_label_ids = torch.unique(reindexed_edge_index.T[reindexed_edge_type==0][:,1])

        highlevel_graph = Data(edge_index=reindexed_edge_index,
                                edge_type=reindexed_edge_type,
                                edge_weight=None,
                                num_nodes=unique_elements.size(0),
                                num_relations=2)

        target_mask = torch.zeros(highlevel_graph.num_nodes)
        labeled_ents = reindexed_edge_index[0][(reindexed_edge_type == 0)].unique()
        target_mask[labeled_ents] = 1
        target_mask[reindexed_label_ids] = -1
        highlevel_graph = build_relation_graph(highlevel_graph)

        second_graph = self.graph_encoder_gc.build_semantic_graph(highlevel_graph, "self", semantic_type="bert", mode="test", target_mask=target_mask,
                                                            node_feature=highlevel_embeds, ifload=False, k=k, p=p)
        second_graph.num_nodes = data.num_nodes
        second_graph.edge_index = second_graph.ex_edge_index
        second_graph.edge_type = second_graph.ex_edge_type
        second_graph.num_relations = second_graph.ex_num_relations

        return second_graph, highlevel_embeds, unique_elements

    def forward(self, data, batch, k=None, p=None):
        if k is not None:
            # GC mode: graph-level classification via meta-graph reasoning
            if self.second_graph is None or self.second_graph.num_nodes != data.num_nodes:
                self.second_graph, self.highlevel_embeds, self.unique_elements = self.build_global(data, k=k, p=p)
                highlevel_data = copy.deepcopy(self.second_graph)
                highlevel_data.num_nodes = self.unique_elements.size(0)
            target_relation_id = data.target_edge_type.unique()
            highlevel_data = copy.deepcopy(self.second_graph)
            highlevel_data.num_nodes = self.unique_elements.size(0)
            new_id_vector = torch.searchsorted(self.unique_elements, batch.view(-1))
            reindex_batch = new_id_vector.view(batch.shape)
            reindex_batch[:, :, 2] = 0
            highlevel_embeds = self.highlevel_embeds
            query_rels = torch.zeros_like(batch[:, 0, 2])

            globalent_embeds = None
            if self.ablation_type != "global":
                globalent_embeds = highlevel_embeds
            relation_graph = highlevel_data.ex_relation_graph if self.ablation_type != "relgraph" else highlevel_data.relation_graph
            if self.ablation_type == "boundary":
                highlevel_data.topk_indices = None

            relation_representations = self.relation_model(relation_graph, query=query_rels)

            score = self.entity_model(highlevel_data, relation_representations, reindex_batch, node_embeds = globalent_embeds)
            if "MOE" in self.ent_model_class:
                return score, self.entity_model.moe_loss
            else:
                return score

        # KG / NC mode: node-level link prediction and classification
        globalent_embeds = None
        if self.with_feats and self.ablation_type != "global":
            if hasattr(data, "ent_embeds"):
                globalrel_embeds = self.relation_model(data.ex_relation_graph, query=None)
                globalent_embeds= self.entity_model.global_forward(data, globalrel_embeds, data.ent_embeds).squeeze(0)
                globalent_embeds = self.global_mlp(globalent_embeds)

        query_rels = batch[:, 0, 2]

        relation_graph = data.ex_relation_graph if self.ablation_type != "relgraph" else data.relation_graph
        relation_representations = self.relation_model(relation_graph, query=query_rels)

        if self.ablation_type == "boundary":
            data.topk_indices = None

        score = self.entity_model(data, relation_representations, batch, node_embeds = globalent_embeds)
        if "MOE" in self.ent_model_class:
            return score, self.entity_model.moe_loss
        else:
            return score

# NBFNet to work on the graph of relations with 4 fundamental interactions
# Doesn't have the final projection MLP from hidden dim -> 1, returns all node representations 
# of shape [bs, num_rel, hidden]
class RelNBFNet(BaseNBFNet):

    def __init__(self, input_dim, hidden_dims, num_relation=4, **kwargs):
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)

        self.with_feats = kwargs["with_feats"] if "with_feats" in kwargs else False
        self.feat_dim = kwargs["feat_dim"] if "feat_dim" in kwargs else 0
        self.feat_type = kwargs["feat_type"] if "feat_type" in kwargs else "none"
        
        if self.with_feats and self.feat_type == "middle":
            input_dim = input_dim + self.feat_dim
            hidden_dims = [hidden_dim + self.feat_dim for hidden_dim in hidden_dims]
            self.dims = [dim + self.feat_dim for dim in self.dims]

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False)
                )

        if self.concat_hidden:
            feature_dim = sum(hidden_dims) + input_dim
            self.mlp = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, input_dim)
            )

    def bellmanford(self, data, h_index, separate_grad=False, rel_embeds=None):
        if h_index is not None:
            batch_size = len(h_index)
            # initialize initial nodes (relations of interest in the batcj) with all ones
            query = torch.ones(h_index.shape[0], self.dims[0], device=h_index.device, dtype=torch.float)
            index = h_index.unsqueeze(-1).expand_as(query)
            # initial (boundary) condition - initialize all node states as zeros
            boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=h_index.device)
            #boundary = torch.zeros(data.num_nodes, *query.shape, device=h_index.device)
            # Indicator function: by the scatter operation we put ones as init features of source (index) nodes
            boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))
            edge_weight = torch.ones(data.num_edges, device=h_index.device)
        else: # For Global Computation
            batch_size = 1
            device = self.layers[0].linear.weight.device
            query = torch.ones(1, self.dims[0], device=device, dtype=torch.float)
            boundary = torch.zeros(1, data.num_nodes, self.dims[0], device=device)
            edge_weight = torch.ones(data.num_edges, device=device)
        

        if rel_embeds is not None and self.feat_type == "middle":
            rel_embeds = rel_embeds.unsqueeze(0).expand(batch_size, -1, -1)
            boundary[:, :, -rel_embeds.shape[-1]:] = rel_embeds

        if rel_embeds is not None and self.feat_type == "front":
            rel_embeds = rel_embeds.unsqueeze(0).expand(batch_size, -1, -1)
            boundary = rel_embeds

        hiddens = []
        edge_weights = []
        layer_input = boundary
        size = (data.num_nodes, data.num_nodes)

        for layer in self.layers:
            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            assert query.device == layer_input.device
            hidden = layer(layer_input, query, boundary, data.edge_index, data.edge_type, size, edge_weight)
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        if rel_embeds is not None and self.feat_type == "end":
            rel_embeds = rel_embeds.unsqueeze(0).expand(batch_size, -1, -1)
            # hidden = hiddens[-1] + rel_embeds
            node_query = rel_embeds
        else: 
            # original query (relation type) embeddings
            node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1) # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
            output = self.mlp(output)
        else:
            output = hiddens[-1]

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, rel_graph, query, rel_embeds=None):

        # message passing and updated node representations (that are in fact relations)
        output = self.bellmanford(rel_graph, h_index=query)["node_feature"]  # (batch_size, num_nodes, hidden_dim）
        
        return output

class EntityNBFNet(BaseNBFNet):

    def __init__(self, input_dim, hidden_dims, num_relation=1, **kwargs):

        # dummy num_relation = 1 as we won't use it in the NBFNet layer
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)
        self.with_feats = kwargs["with_feats"] if "with_feats" in kwargs else False
        self.feat_dim = kwargs["feat_dim"] if "feat_dim" in kwargs else 0
        self.feat_type = kwargs["feat_type"] if "feat_type" in kwargs else "None"

        # if self.with_feats and self.feat_type == "middle":
        #     input_dim = input_dim + self.feat_dim
        #     hidden_dims = [hidden_dim + self.feat_dim for hidden_dim in hidden_dims]
        #     self.dims = [dim + self.feat_dim for dim in self.dims]

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False, project_relations=True)
            )

        feature_dim = (sum(hidden_dims) if self.concat_hidden else hidden_dims[-1]) + input_dim
        self.mlp = nn.Sequential()
        mlp = []
        for i in range(self.num_mlp_layers - 1):
            mlp.append(nn.Linear(feature_dim, feature_dim))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(feature_dim, 1))
        self.mlp = nn.Sequential(*mlp)
        self.dropout = nn.Dropout(0.1)
        #self.assis_vector = nn.Parameter(torch.ones(3, self.dims[0])) * 0.001
        self.assis_vector = nn.Parameter(torch.ones(self.dims[0]))
        self.assis_vector2 = nn.Parameter(torch.ones(self.dims[0]))
        self.assis_vector3 = nn.Parameter(torch.ones(self.dims[0]))

    def bellmanford_bak(self, data, h_index, r_index, node_embeds=None, separate_grad=False):
        batch_size = len(r_index)
        
        # initialize queries (relation types of the given triples)
        query = self.query[torch.arange(batch_size, device=r_index.device), r_index]
        index = h_index.unsqueeze(-1).expand_as(query)

        # initial (boundary) condition - initialize all node states as zeros
        boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=h_index.device)
        # by the scatter operation we put query (relation) embeddings as init features of source (index) nodes
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))
        
        if node_embeds is not None and self.feat_type == "middle":
            node_embeds = node_embeds.unsqueeze(0).expand(batch_size, -1, -1)
            boundary[:, :, -node_embeds.shape[-1]:] = node_embeds

        if node_embeds is not None and self.feat_type == "front":
            node_embeds = node_embeds.unsqueeze(0).expand(batch_size, -1, -1)
            boundary = node_embeds
        
        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=h_index.device)

        hiddens = []
        edge_weights = []
        layer_input = boundary

        for i, layer in enumerate(self.layers):

            # for visualization
            if separate_grad:
                edge_weight = edge_weight.clone().requires_grad_()

            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            # print(layer.relation.shape, data.edge_type.max(), data.edge_type.unique().shape)
            hidden = layer(layer_input, query, boundary, data.edge_index, data.edge_type, size, edge_weight, r_index) # add additional r_index

            # print(i, layer_input.shape, (layer_input.sum(2)==0).sum(1), hidden.shape)

            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden
        
        if node_embeds is not None and self.feat_type == "end":
            node_embeds = node_embeds.unsqueeze(0).expand(batch_size, -1, -1)
            # hidden = hiddens[-1] + node_embeds
            node_query = node_embeds
        else:
            # original query (relation type) embeddings
            node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1) # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output = torch.cat([hidden, node_query], dim=-1)

        # assert torch.isnan(output).any() == False
        output[torch.isnan(output)] = 0

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def bellmanford(self, data, h_index, r_index, node_embeds=None, feature_boundary=False, separate_grad=False):
        if h_index is None:
            batch_size = 1
            query = torch.ones_like(self.query[0,0]).unsqueeze(0)
            device = self.layers[0].linear.weight.device
            if node_embeds is not None:
                boundary = node_embeds.unsqueeze(0)
            else:
                boundary = torch.zeros(1, data.num_nodes, self.dims[0], device=device)
            extended_query = query
            edge_weight = torch.ones(data.num_edges, device=device)
        else:
            batch_size = len(r_index)
            # initialize queries (relation types of the given triples)
            query = self.query[torch.arange(batch_size, device=r_index.device), r_index]
            index = h_index.unsqueeze(-1).expand_as(query)
            # initial (boundary) condition - initialize all node states as zeros
            boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=h_index.device)
            # by the scatter operation we put query (relation) embeddings as init features of source (index) nodes
            boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))
            
            if node_embeds is not None and feature_boundary:
                node_embeds = node_embeds.float()
                boundary = node_embeds.unsqueeze(0).expand_as(boundary)
                
                # boundary.scatter_add_(1, index.unsqueeze(1), node_embeds[h_index].unsqueeze(1))  # INIT3

            extended_query = query
            edge_weight = torch.ones(data.num_edges, device=h_index.device)

            ##########################################
            boundary2 = torch.zeros_like(boundary)
            if hasattr(data, "topk_indices") and not feature_boundary:
                topk_indices = getattr(data, "topk_indices")
                if topk_indices is not None:
                    for i, single_index in enumerate(h_index):
                        for j in topk_indices[single_index]:
                            boundary[i, j] = self.assis_vector.to(boundary) 

            # boundary = torch.cat([boundary, boundary2, boundary+boundary2], dim=0) # original, topksole, hybrid
            # assis_query1 = self.assis_vector2.unsqueeze(0).expand_as(query).to(boundary) 
            # assis_query2 = self.assis_vector3.unsqueeze(0).expand_as(query).to(boundary) 
            # extended_query = torch.cat([query, assis_query1, assis_query2], dim=0)
            # for i, layer in enumerate(self.layers):
            #     layer.relation = layer.relation.repeat(3, 1, 1)  #   torch.cat([layer.relation, layer.relation, layer.relation], dim=0)
            # r_index = torch.cat([r_index, r_index, r_index], dim=0)
            ########################################## 

            # if node_embeds is not None and self.feat_type == "middle":
            #     node_embeds = node_embeds.unsqueeze(0).expand(batch_size, -1, -1)
            #     boundary[:, :, -node_embeds.shape[-1]:] = node_embeds
            
            if node_embeds is not None and self.feat_type == "front":
                node_embeds = node_embeds.unsqueeze(0).expand(batch_size, -1, -1)
                boundary = node_embeds
            if node_embeds is not None and self.feat_type == "middle":
               node_embeds = node_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        
        size = (data.num_nodes, data.num_nodes)
        hiddens = []
        edge_weights = [] 
        layer_input = boundary

        for i, layer in enumerate(self.layers):
            # for visualization
            if separate_grad:
                edge_weight = edge_weight.clone().requires_grad_()
            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            hidden = layer(layer_input, extended_query, boundary, data.edge_index, data.edge_type, size, edge_weight, r_index) # add additional r_index
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            if node_embeds is not None and self.feat_type == "middle":
                hidden = hidden + node_embeds
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        ##########################################
        # hidden_split = hidden.view(batch_size, 3, hidden.size(1), hidden.size(2))
        # hidden_split = self.dropout(hidden_split) 
        # hidden = hidden_split.sum(1)
        ###########################################
        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1) # (batch_size, num_nodes, input_dim)
        if node_embeds is not None and self.feat_type == "end":
            node_embeds = node_embeds.unsqueeze(0).expand(batch_size, -1, -1)
            hidden = hiddens[-1] + node_embeds
            # node_query = node_embeds

        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output = torch.cat([hidden, node_query], dim=-1)
        # assert torch.isnan(output).any() == False
        output[torch.isnan(output)] = 0

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }
    
    def global_forward(self, data, relation_representations, node_embeds=None):
        self.query = relation_representations
        for layer in self.layers:
            layer.relation = relation_representations
        output = self.bellmanford(data, None, None, node_embeds)  # (num_nodes, batch_size, feature_dim）
        feature = output["node_feature"][:,:,:self.dims[-1]]
        return feature

    def forward(self, data, relation_representations, batch, feature_boundary=False, node_embeds=None):
        h_index, t_index, r_index = batch.unbind(-1)
        # print(h_index.max(), t_index.max(), r_index.max(), relation_representations.shape)
        
        # initial query representations are those from the relation graph
        self.query = relation_representations

        # initialize relations in each NBFNet layer (with uinque projection internally)
        for layer in self.layers:
            layer.relation = relation_representations

        if self.training:
            # Edge dropout in the training mode
            # here we want to remove immediate edges (head, relation, tail) from the edge_index and edge_types
            # to make NBFNet iteration learn non-trivial paths
            data = self.remove_easy_edges(data, h_index, t_index, r_index)

        shape = h_index.shape
        # turn all triples in a batch into a tail prediction mode
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index, num_direct_rel=data.num_relations // 2)
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()
        # print(h_index, t_index, r_index)

        # message passing and updated node representations
        output = self.bellmanford(data, h_index[:, 0], r_index[:, 0], node_embeds, feature_boundary)  # (num_nodes, batch_size, feature_dim）
        feature = output["node_feature"]
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        # extract representations of tail entities from the updated node states
        feature = feature.gather(1, index)  # (batch_size, num_negative + 1, feature_dim)

        # probability logit for each tail node in the batch
        # (batch_size, num_negative + 1, dim) -> (batch_size, num_negative + 1)
        score = self.mlp(feature).squeeze(-1)
        return score.view(shape)
    
    # for graph task
    def encoding(self, data, relation_representations, batch, feature_boundary=False, node_embeds=None):
        h_index, t_index, r_index = batch.unbind(-1)
        # print(h_index.max(), t_index.max(), r_index.max(), relation_representations.shape)
        
        # initial query representations are those from the relation graph
        self.query = relation_representations

        # initialize relations in each NBFNet layer (with uinque projection internally)
        for layer in self.layers:
            layer.relation = relation_representations

        if self.training:
            # Edge dropout in the training mode
            # here we want to remove immediate edges (head, relation, tail) from the edge_index and edge_types
            # to make NBFNet iteration learn non-trivial paths
            data = self.remove_easy_edges(data, h_index, t_index, r_index)

        shape = h_index.shape
        # turn all triples in a batch into a tail prediction mode
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index, num_direct_rel=data.num_relations // 2)
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()
        # print(h_index, t_index, r_index)

        # message passing and updated node representations
        output = self.bellmanford(data, h_index[:, 0], r_index[:, 0], node_embeds, feature_boundary)  # (num_nodes, batch_size, feature_dim）
        feature = output["node_feature"]
        feature = feature[:,:,:self.dims[-1]]
        return feature
    
    # for preedge output
    def forward_output(self, data, relation_representations, batch, feature_boundary=False, node_embeds=None):
        h_index, t_index, r_index = batch.unbind(-1)
        # print(h_index.max(), t_index.max(), r_index.max(), relation_representations.shape)
        
        # initial query representations are those from the relation graph
        self.query = relation_representations

        # initialize relations in each NBFNet layer (with uinque projection internally)
        for layer in self.layers:
            layer.relation = relation_representations

        if self.training:
            # Edge dropout in the training mode
            # here we want to remove immediate edges (head, relation, tail) from the edge_index and edge_types
            # to make NBFNet iteration learn non-trivial paths
            data = self.remove_easy_edges(data, h_index, t_index, r_index)

        shape = h_index.shape
        # turn all triples in a batch into a tail prediction mode
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index, num_direct_rel=data.num_relations // 2)
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()
        # print(h_index, t_index, r_index)

        # message passing and updated node representations
        output = self.bellmanford(data, h_index[:, 0], r_index[:, 0], node_embeds, feature_boundary=False)  # (num_nodes, batch_size, feature_dim）
        feature = output["node_feature"]
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        # extract representations of tail entities from the updated node states
        feature = feature.gather(1, index)  # (batch_size, num_negative + 1, feature_dim)

        # probability logit for each tail node in the batch
        # (batch_size, num_negative + 1, dim) -> (batch_size, num_negative + 1)
        score = self.mlp(feature).squeeze(-1)
        return score.view(shape), output["node_feature"]
