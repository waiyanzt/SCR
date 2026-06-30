import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree
from typing import Tuple

from .moe import MoE
from torch_scatter import scatter_mean, scatter_add, scatter_max, scatter_softmax


class MultiConvMoE(MoE):
    def __init__(self, input_size, output_size, num_experts, experts_conv, experts_bn=None, noisy_gating=True, k=4, coef=1, num_experts_1hop=None, layer_type="ent_conv_fnn"):
        super(MultiConvMoE, self).__init__(input_size, output_size, num_experts, experts_conv, experts_bn, noisy_gating, k, coef, num_experts_1hop)
        self.layer_type = layer_type
        
    def forward(self, kwargs):
        if self.layer_type == "rel_layer":
            return self.forward_rel_layer(**kwargs)
        elif self.layer_type == "ent_conv_fnn":
            return self.forward_ent_conv_fnn(**kwargs)
    
    def forward_rel_layer(self, layer_input, query, boundary, edge_index, edge_type, size, edge_weight):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        gates, load = self.noisy_top_k_gating(query, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        #
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= self.loss_coef

        expert_outputs = []
        for i in range(self.num_experts):
            expert_i_output = self.experts_conv[i](layer_input, query, boundary, edge_index, edge_type, size, edge_weight)
            if self.experts_bn is not None:
                expert_i_output = self.experts_bn[i](expert_i_output)
            expert_outputs.append(expert_i_output)
        expert_outputs = torch.stack(expert_outputs, dim=1) # shape=[num_nodes, num_experts, d_feature]
       
        # gates: shape=[num_nodes, num_experts]
        y = gates.unsqueeze(dim=-1).unsqueeze(dim=-1) * expert_outputs
        y = y.mean(dim=1)
        return y, loss
        
    def forward_ent_conv_fnn(self, layer_input, query):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        
        gates, load = self.noisy_top_k_gating(query, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        #
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= self.loss_coef
        
        if False:
            expert_outputs = torch.zeros([gates.shape[0], layer_input.shape[-2], self.output_size]).to(layer_input.device)
            for i in range(self.num_experts):
                gate_index = torch.nonzero(gates[:,i]).squeeze(1)
                expert_i_output = self.experts_conv[i](layer_input[gate_index])
                expert_i_output = self.experts_bn[i](expert_i_output)
                expert_outputs[gate_index] += expert_i_output * gates[:,i][gate_index].unsqueeze(dim=-1).unsqueeze(dim=-1)
            y = expert_outputs / self.k
        
        if True:
            expert_outputs = []
            for i in range(self.num_experts):
                expert_i_output = self.experts_conv[i](layer_input)
                expert_i_output = self.experts_bn[i](expert_i_output)
                expert_outputs.append(expert_i_output)
            expert_outputs = torch.stack(expert_outputs, dim=1) # shape=[num_nodes, num_experts, d_feature]
           
            # gates: shape=[num_nodes, num_experts]
            y = gates.unsqueeze(dim=-1).unsqueeze(dim=-1) * expert_outputs
            y = y.mean(dim=1)

        return y, loss

class GeneralizedRelationalConv_bak(MessagePassing):

    eps = 1e-6

    message2mul = {
        "transe": "add",
        "distmult": "mul",
    }

    # TODO for compile() - doesn't work currently
    # propagate_type = {"edge_index": torch.LongTensor, "size": Tuple[int, int]}

    def __init__(self, input_dim, output_dim, num_relation, query_input_dim, message_func="distmult",
                 aggregate_func="pna", layer_norm=False, activation="relu", has_readout=None, dependent=False, project_relations=False):
        super(GeneralizedRelationalConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.query_input_dim = query_input_dim
        self.message_func = message_func
        self.aggregate_func = aggregate_func
        self.dependent = dependent
        self.project_relations = project_relations
        self.readout_type = has_readout

        if self.readout_type is not None:
            if self.readout_type == "query":
                self.outward_readout = nn.Linear(input_dim, output_dim)
                self.inward_readout = nn.Linear(input_dim, output_dim)
            else:
                if self.readout_type == "mean":
                    self.readout = MeanReadout()
                elif self.readout_type == "sum":
                    self.readout = SumReadout()
                else:
                    raise NotImplementedError
            self.transform_readout = nn.Linear(input_dim, output_dim)

        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        if self.aggregate_func == "pna":
            self.linear = nn.Linear(input_dim * 13, output_dim)
        elif self.aggregate_func == "spna":
            self.linear = nn.Linear(input_dim * 7, output_dim)
        else:
            self.linear = nn.Linear(input_dim * 2, output_dim)  # bias = False

        if dependent:
            # obtain relation embeddings as a projection of the query relation
            self.relation_linear = nn.Linear(query_input_dim, num_relation * input_dim)
        else:
            if not self.project_relations:
                # relation embeddings as an independent embedding matrix per each layer
                self.relation = nn.Embedding(num_relation, input_dim)
            else:
                # will be initialized after the pass over relation graph
                self.relation = None
                self.relation_projection = nn.Sequential(
                    nn.Linear(input_dim, input_dim),
                    nn.ReLU(),
                    nn.Linear(input_dim, input_dim)
                )

    def forward(self, input, query, boundary, edge_index, edge_type, size, edge_weight=None):
        batch_size = len(query)
        # print(self.num_relation, edge_type.max().item()+1)
        # self.num_relation = max(self.num_relation, edge_type.max().item()+1)
        
        if self.dependent:
            # layer-specific relation features as a projection of input "query" (relation) embeddings
            relation = self.relation_linear(query).view(batch_size, self.num_relation, self.input_dim)
        else:
            if not self.project_relations:
                # layer-specific relation features as a special embedding matrix unique to each layer
                relation = self.relation.weight.expand(batch_size, -1, -1)
            else:
                # NEW and only change: 
                # projecting relation features to unique features for this layer, then resizing for the current batch
                relation = self.relation_projection(self.relation)
        if edge_weight is None:
            edge_weight = torch.ones(len(edge_type), device=input.device)
        
        # note that we send the initial boundary condition (node states at layer0) to the message passing
        # correspond to Eq.6 on p5 in https://arxiv.org/pdf/2106.06935.pdf
        output = self.propagate(input=input, relation=relation, boundary=boundary, edge_index=edge_index,
                                edge_type=edge_type, size=size, edge_weight=edge_weight)
        return output

    def propagate(self, edge_index, size=None, **kwargs):
        if kwargs["edge_weight"].requires_grad or self.message_func == "rotate":
            # the rspmm cuda kernel only works for TransE and DistMult message functions
            # otherwise we invoke separate message & aggregate functions
            return super(GeneralizedRelationalConv, self).propagate(edge_index, size, **kwargs)

        for hook in self._propagate_forward_pre_hooks.values():
            res = hook(self, (edge_index, size, kwargs))
            if res is not None:
                edge_index, size, kwargs = res

        # in newer PyG, 
        # __check_input__ -> _check_input()
        # __collect__ -> _collect()
        # __fused_user_args__ -> _fuser_user_args
        size = self._check_input(edge_index, size)
        coll_dict = self._collect(self._fused_user_args, edge_index, size, kwargs)

        msg_aggr_kwargs = self.inspector.distribute("message_and_aggregate", coll_dict)
        for hook in self._message_and_aggregate_forward_pre_hooks.values():
            res = hook(self, (edge_index, msg_aggr_kwargs))
            if res is not None:
                edge_index, msg_aggr_kwargs = res
        out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)
        for hook in self._message_and_aggregate_forward_hooks.values():
            res = hook(self, (edge_index, msg_aggr_kwargs), out)
            if res is not None:
                out = res

        update_kwargs = self.inspector.distribute("update", coll_dict)
        out = self.update(out, **update_kwargs)

        for hook in self._propagate_forward_hooks.values():
            res = hook(self, (edge_index, size, kwargs), out)
            if res is not None:
                out = res

        return out

    def message(self, input_j, relation, boundary, edge_type):
        # print(relation.shape, relation.max(), self.node_dim, edge_type.min(), edge_type.max(), edge_type.shape)
        edge_type = torch.clamp(edge_type, 0, relation.shape[self.node_dim]-1)
        relation_j = relation.index_select(self.node_dim, edge_type)

        if self.message_func == "transe":
            message = input_j + relation_j
        elif self.message_func == "distmult":
            message = input_j * relation_j
        elif self.message_func == "rotate":
            x_j_re, x_j_im = input_j.chunk(2, dim=-1)
            r_j_re, r_j_im = relation_j.chunk(2, dim=-1)
            message_re = x_j_re * r_j_re - x_j_im * r_j_im
            message_im = x_j_re * r_j_im + x_j_im * r_j_re
            message = torch.cat([message_re, message_im], dim=-1)
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)
        # print(message.shape, boundary.shape, print(torch.cuda.memory_summary()))
        # augment messages with the boundary condition
        message = torch.cat([message, boundary], dim=self.node_dim)  # (num_edges + num_nodes, batch_size, input_dim)
        return message

    def aggregate(self, input, edge_weight, index, dim_size):
        # augment aggregation index with self-loops for the boundary condition
        dim_size = dim_size.item() if torch.is_tensor(dim_size) else dim_size
        index = torch.cat([index, torch.arange(dim_size, device=input.device)]) # (num_edges + num_nodes,)
        edge_weight = torch.cat([edge_weight, torch.ones(dim_size, device=input.device)])
        shape = [1] * input.ndim
        shape[self.node_dim] = -1
        edge_weight = edge_weight.view(shape)

        if self.aggregate_func == "pna":
            mean = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="mean")
            sq_mean = scatter(input ** 2 * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="mean")
            max = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="max")
            min = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="min")
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2)
            degree_out = degree(index, dim_size).unsqueeze(0).unsqueeze(-1)
            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            output = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        elif self.aggregate_func == "spna":
            mean = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="mean")
            sq_mean = scatter(input ** 2 * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="mean")
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2)
            degree_out = degree(index, dim_size).unsqueeze(0).unsqueeze(-1)
            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            output = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        else:
            output = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size,
                             reduce=self.aggregate_func)
        return output

    def message_and_aggregate(self, edge_index, input, relation, boundary, edge_type, edge_weight, index, dim_size):
        # fused computation of message and aggregate steps with the custom rspmm cuda kernel
        # speed up computation by several times
        # reduce memory complexity from O(|E|d) to O(|V|d), so we can apply it to larger graphs
        # print(relation.shape, relation.max(), self.node_dim, edge_type.min(), edge_type.max(), edge_type.shape)
        from .rspmm import generalized_rspmm

        batch_size, num_node = input.shape[:2]
        input = input.transpose(0, 1).flatten(1)
        relation = relation.transpose(0, 1).flatten(1)
        boundary = boundary.transpose(0, 1).flatten(1)
        degree_out = degree(index, dim_size).unsqueeze(-1) + 1

        if self.message_func in self.message2mul:
            mul = self.message2mul[self.message_func]
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)
        # print(input.shape, relation.shape, index.shape)
        torch.cuda.synchronize()
        if self.aggregate_func == "sum":
            update = generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum="add", mul=mul)
            torch.cuda.synchronize()
            update = update.contiguous()
            update = update + boundary
        elif self.aggregate_func == "mean":
            update = generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum="add", mul=mul)
            update = (update + boundary) / degree_out
        elif self.aggregate_func == "max":
            update = generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum="max", mul=mul)
            update = torch.max(update, boundary)
        elif self.aggregate_func == "pna":
            # we use PNA with 4 aggregators (mean / max / min / std)
            # and 3 scalars (identity / log degree / reciprocal of log degree)
            sum = generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum="add", mul=mul)
            sq_sum = generalized_rspmm(edge_index, edge_type, edge_weight, relation ** 2, input ** 2, sum="add",
                                       mul=mul)
            max = generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum="max", mul=mul)
            min = generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum="min", mul=mul)
            mean = (sum + boundary) / degree_out
            sq_mean = (sq_sum + boundary ** 2) / degree_out
            max = torch.max(max, boundary)
            min = torch.min(min, boundary) # (node, batch_size * input_dim)
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2) # (node, batch_size * input_dim * 4)
            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1) # (node, 3)
            update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2) # (node, batch_size * input_dim * 4 * 3)
        elif self.aggregate_func == "spna":
            # we use PNA with 2 aggregators (mean / std)
            # and 3 scalars (identity / log degree / reciprocal of log degree)
            sum = generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum="add", mul=mul)
            sq_sum = generalized_rspmm(edge_index, edge_type, edge_weight, relation ** 2, input ** 2, sum="add",
                                       mul=mul)
            mean = (sum + boundary) / degree_out
            sq_mean = (sq_sum + boundary ** 2) / degree_out
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2) # (node, batch_size * input_dim * 4)
            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1) # (node, 3)
            update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2) # (node, batch_size * input_dim * 4 * 3)
        else:
            raise ValueError("Unknown aggregation function `%s`" % self.aggregate_func)
        update = update.view(num_node, batch_size, -1).transpose(0, 1)
        return update

    def update(self, update, input):
        if self.readout_type is not None:
            if self.readout_type == "query":
                ## Here we will aggregate all the rep with the specific query
                # edge_list = graph.edge_list
                # tail_read_out, head_read_out = self.calculate_read_out(
                #     edge_list, input, r_index, self.readout_type
                # )
                # output = self.linear(torch.cat([input, update], dim=-1))
                # tail_readout_update = self.outward_readout(tail_read_out)
                # output += tail_readout_update
                # head_readout_update = self.inward_readout(head_read_out)
                # output += head_readout_update
                output = self.linear(torch.cat([input, update], dim=-1))
            else:
                if self.readout_type == "mean":
                    readout_update = input.mean(dim=1)
                elif self.readout_type == "sum":
                    readout_update = input.sum(dim=1)
                readout_update = readout_update.repeat((1, input.size(1), 1))
                output = self.linear(torch.cat([input, update], dim=-1))
                output += self.transform_readout(readout_update)
        else:
            output = self.linear(torch.cat([input, update], dim=-1))
            # output = input + update
        if self.layer_norm:
            # output_mask = (output.sum(-1)!=0)
            # output[output_mask] = self.layer_norm(output[output_mask]) 
            output = self.layer_norm(output) # 输入的全0向量输出不为0
        if self.activation:
            output = self.activation(output)
        return output
    
    def calculate_read_out(self, edge_list, input, r_index, readout_type):
        num_node, batch_size, dimension = input.shape

        # Create a mask to identify the relevant edge types in r_index
        r_index_expanded = r_index.unsqueeze(0).expand(edge_list.size(0), -1)
        edge_type_mask = edge_list[:, 2].unsqueeze(1) == r_index_expanded

        # Now create two node masks, one for tail and one for head; TODO: rename and swap the order
        tail_read_out_val = torch.zeros(1, batch_size, dimension, device=self.device)

        head_read_out_val = torch.zeros(1, batch_size, dimension, device=self.device)

        # TODO: Need to be further optimized by using torch masked sum
        for i in range(batch_size):
            # Find tail and head indices for each edge type
            tail_node_mask = torch.zeros(num_node, dtype=torch.bool, device=self.device)
            head_node_mask = torch.zeros(num_node, dtype=torch.bool, device=self.device)

            # tail_indices: all the source nodes with relation type q
            tail_indices = edge_list[edge_type_mask[:, i], 0]

            # head_indices: all the target node with relation type q
            head_indices = edge_list[edge_type_mask[:, i], 1]

            # Update the tail and head node masks
            tail_node_mask[tail_indices] = True
            head_node_mask[head_indices] = True

            # we read out node with only the value with input
            tail_masked_input = input[:, i, :] * tail_node_mask.unsqueeze(1).float()
            head_masked_input = input[:, i, :] * head_node_mask.unsqueeze(1).float()

            if readout_type == "sum":
                head_read_out_batch = head_masked_input.sum(dim=0)
                tail_read_out_batch = tail_masked_input.sum(dim=0)
            elif readout_type == "mean":
                head_read_out_batch = head_masked_input.mean(dim=0)
                tail_read_out_batch = tail_masked_input.mean(dim=0)
            else:
                raise NotImplementedError

            tail_read_out_val[:, i, :] = tail_read_out_batch.unsqueeze(0).unsqueeze(0)
            head_read_out_val[:, i, :] = head_read_out_batch.unsqueeze(0).unsqueeze(0)

        tail_read_out = tail_read_out_val.repeat(num_node, 1, 1)
        head_read_out = head_read_out_val.repeat(num_node, 1, 1)
        return tail_read_out, head_read_out
    
class GeneralizedRelationalConv(MessagePassing):

    eps = 1e-6

    message2mul = {
        "transe": "add",
        "distmult": "mul",
    }

    # TODO for compile() - doesn't work currently
    # propagate_type = {"edge_index": torch.LongTensor, "size": Tuple[int, int]}

    def __init__(self, input_dim, output_dim, num_relation, query_input_dim, message_func="distmult",
                 aggregate_func="pna", layer_norm=False, activation="relu", has_readout=None, dependent=False, project_relations=False):
        super(GeneralizedRelationalConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.query_input_dim = query_input_dim
        self.message_func = message_func
        self.aggregate_func = aggregate_func
        self.dependent = dependent
        self.project_relations = project_relations
        self.readout_type = has_readout

        if self.readout_type is not None:
            if self.readout_type == "query":
                self.outward_readout = nn.Linear(input_dim, output_dim)
                self.inward_readout = nn.Linear(input_dim, output_dim)
            else:
                if self.readout_type == "mean":
                    self.readout = MeanReadout()
                elif self.readout_type == "sum":
                    self.readout = SumReadout()
                else:
                    raise NotImplementedError
            self.transform_readout = nn.Linear(input_dim, output_dim)

        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        if self.aggregate_func == "pna":
            self.linear = nn.Linear(input_dim * 13, output_dim)
        elif self.aggregate_func == "spna":
            self.linear = nn.Linear(input_dim * 7, output_dim)
        else:
            self.linear = nn.Linear(input_dim * 2, output_dim)  # , bias = False

        if dependent:
            # obtain relation embeddings as a projection of the query relation
            self.relation_linear = nn.Linear(query_input_dim, num_relation * input_dim)
        else:
            if not self.project_relations:
                # relation embeddings as an independent embedding matrix per each layer
                self.relation = nn.Embedding(num_relation, input_dim)
            else:
                # will be initialized after the pass over relation graph
                self.relation = None
                self.relation_projection = nn.Sequential(
                    nn.Linear(input_dim, input_dim),
                    nn.ReLU(),
                    nn.Linear(input_dim, input_dim)
                )

    def forward(self, input, query, boundary, edge_index, edge_type, size, edge_weight=None, r_index=None):
        batch_size = len(query)
        # print(self.num_relation, edge_type.max().item()+1)
        # self.num_relation = max(self.num_relation, edge_type.max().item()+1)
        if self.dependent:
            # layer-specific relation features as a projection of input "query" (relation) embeddings
            relation = self.relation_linear(query).view(batch_size, self.num_relation, self.input_dim)
        else:
            if not self.project_relations:
                # layer-specific relation features as a special embedding matrix unique to each layer
                relation = self.relation.weight.expand(batch_size, -1, -1)
            else:
                # NEW and only change: 
                # projecting relation features to unique features for this layer, then resizing for the current batch
                relation = self.relation_projection(self.relation)
        if edge_weight is None:
            edge_weight = torch.ones(len(edge_type), device=input.device)
        
        # note that we send the initial boundary condition (node states at layer0) to the message passing
        # correspond to Eq.6 on p5 in https://arxiv.org/pdf/2106.06935.pdf
        assert relation.device == edge_weight.device
        assert edge_index.device == edge_weight.device
        output = self.propagate(input=input, relation=relation, boundary=boundary, edge_index=edge_index,
                                edge_type=edge_type, size=size, edge_weight=edge_weight, r_index=r_index)
        return output

    def propagate(self, edge_index, size=None, **kwargs):
        if kwargs["edge_weight"].requires_grad or self.message_func == "rotate":
            # the rspmm cuda kernel only works for TransE and DistMult message functions
            # otherwise we invoke separate message & aggregate functions
            return super(GeneralizedRelationalConv, self).propagate(edge_index, size, **kwargs)

        for hook in self._propagate_forward_pre_hooks.values():
            res = hook(self, (edge_index, size, kwargs))
            if res is not None:
                edge_index, size, kwargs = res

        # in newer PyG, 
        # __check_input__ -> _check_input()
        # __collect__ -> _collect()
        # __fused_user_args__ -> _fuser_user_args
        size = self._check_input(edge_index, size)
        coll_dict = self._collect(self._fused_user_args, edge_index, size, kwargs)

        msg_aggr_kwargs = self.inspector.distribute("message_and_aggregate", coll_dict)
        for hook in self._message_and_aggregate_forward_pre_hooks.values():
            res = hook(self, (edge_index, msg_aggr_kwargs))
            if res is not None:
                edge_index, msg_aggr_kwargs = res
        out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)
        for hook in self._message_and_aggregate_forward_hooks.values():
            res = hook(self, (edge_index, msg_aggr_kwargs), out)
            if res is not None:
                out = res

        update_kwargs = self.inspector.distribute("update", coll_dict)
        out = self.update(out, **update_kwargs)

        for hook in self._propagate_forward_hooks.values():
            res = hook(self, (edge_index, size, kwargs), out)
            if res is not None:
                out = res

        return out

    def message(self, input_j, relation, boundary, edge_type):
        # print(relation.shape, relation.max(), self.node_dim, edge_type.min(), edge_type.max(), edge_type.shape)
        edge_type = torch.clamp(edge_type, 0, relation.shape[self.node_dim]-1)
        relation_j = relation.index_select(self.node_dim, edge_type)

        if self.message_func == "transe":
            message = input_j + relation_j
        elif self.message_func == "distmult":
            message = input_j * relation_j
        elif self.message_func == "rotate":
            x_j_re, x_j_im = input_j.chunk(2, dim=-1)
            r_j_re, r_j_im = relation_j.chunk(2, dim=-1)
            message_re = x_j_re * r_j_re - x_j_im * r_j_im
            message_im = x_j_re * r_j_im + x_j_im * r_j_re
            message = torch.cat([message_re, message_im], dim=-1)
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)
        # print(message.shape, boundary.shape, print(torch.cuda.memory_summary()))
        # augment messages with the boundary condition
        message = torch.cat([message, boundary], dim=self.node_dim)  # (num_edges + num_nodes, batch_size, input_dim)
        return message

    def aggregate(self, input, edge_weight, index, dim_size):
        # augment aggregation index with self-loops for the boundary condition
        dim_size = dim_size.item() if torch.is_tensor(dim_size) else dim_size
        index = torch.cat([index, torch.arange(dim_size, device=input.device)]) # (num_edges + num_nodes,)
        edge_weight = torch.cat([edge_weight, torch.ones(dim_size, device=input.device)])
        shape = [1] * input.ndim
        shape[self.node_dim] = -1
        edge_weight = edge_weight.view(shape)

        if self.aggregate_func == "pna":
            mean = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="mean")
            sq_mean = scatter(input ** 2 * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="mean")
            max = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="max")
            min = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="min")
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2)
            degree_out = degree(index, dim_size).unsqueeze(0).unsqueeze(-1)
            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            output = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        elif self.aggregate_func == "spna":
            mean = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="mean")
            sq_mean = scatter(input ** 2 * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="mean")
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2)
            degree_out = degree(index, dim_size).unsqueeze(0).unsqueeze(-1)
            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            output = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        else:
            output = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size,
                             reduce=self.aggregate_func)
        return output

    def message_and_aggregate(self, edge_index, input, relation, boundary, edge_type, edge_weight, index, dim_size, r_index=None):
        # fused computation of message and aggregate steps with the custom rspmm cuda kernel
        # speed up computation by several times
        # reduce memory complexity from O(|E|d) to O(|V|d), so we can apply it to larger graphs
        # print(relation.shape, relation.max(), self.node_dim, edge_type.min(), edge_type.max(), edge_type.shape)
        from .rspmm import generalized_rspmm
        batch_size, num_node = input.shape[:2]
        input = input.transpose(0, 1).flatten(1)
        relation = relation.transpose(0, 1).flatten(1)
        boundary = boundary.transpose(0, 1).flatten(1)
        degree_out = degree(index, dim_size).unsqueeze(-1) + 1

        # if r_index is not None and r_index.min()==r_index.max(): 
        #     n_rel = relation.size(0) // 2
        #     inverse_r_index = r_index.min()+n_rel if r_index.min() < n_rel else r_index.min()-n_rel
        #     # print(r_index, inverse_r_index, relation.shape, n_rel, len(edge_type.unique()))
        #     edge_mask = edge_type != inverse_r_index
        #     edge_index = edge_index[:, edge_mask]
        #     edge_type = edge_type[edge_mask]
        #     edge_weight = edge_weight[edge_mask]

        if self.message_func in self.message2mul:
            mul = self.message2mul[self.message_func]
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)
        # print(input.shape, relation.shape, index.shape)
        torch.cuda.synchronize()
        if self.aggregate_func == "sum":
            update = generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum="add", mul=mul)
            torch.cuda.synchronize()
            update = update.contiguous()
            update = update + boundary
        elif self.aggregate_func == "mean":
            update = generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum="add", mul=mul)
            update = (update + boundary) / degree_out
        elif self.aggregate_func == "max":
            update = generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum="max", mul=mul)
            update = torch.max(update, boundary)
        elif self.aggregate_func == "pna":
            # we use PNA with 4 aggregators (mean / max / min / std)
            # and 3 scalars (identity / log degree / reciprocal of log degree)
            sum = generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum="add", mul=mul)
            sq_sum = generalized_rspmm(edge_index, edge_type, edge_weight, relation ** 2, input ** 2, sum="add",
                                       mul=mul)
            max = generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum="max", mul=mul)
            min = generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum="min", mul=mul)
            mean = (sum + boundary) / degree_out
            sq_mean = (sq_sum + boundary ** 2) / degree_out
            max = torch.max(max, boundary)
            min = torch.min(min, boundary) # (node, batch_size * input_dim)
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2) # (node, batch_size * input_dim * 4)
            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1) # (node, 3)
            update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2) # (node, batch_size * input_dim * 4 * 3)
        elif self.aggregate_func == "spna":
            # we use PNA with 2 aggregators (mean / std)
            # and 3 scalars (identity / log degree / reciprocal of log degree)
            sum = generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum="add", mul=mul)
            sq_sum = generalized_rspmm(edge_index, edge_type, edge_weight, relation ** 2, input ** 2, sum="add",
                                       mul=mul)
            mean = (sum + boundary) / degree_out
            sq_mean = (sq_sum + boundary ** 2) / degree_out
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2) # (node, batch_size * input_dim * 4)
            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1) # (node, 3)
            update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2) # (node, batch_size * input_dim * 4 * 3)
        else:
            raise ValueError("Unknown aggregation function `%s`" % self.aggregate_func)
        update = update.view(num_node, batch_size, -1).transpose(0, 1)
        return update

    def update(self, update, input):
        if self.readout_type is not None:
            if self.readout_type == "query":
                ## Here we will aggregate all the rep with the specific query
                # edge_list = graph.edge_list
                # tail_read_out, head_read_out = self.calculate_read_out(
                #     edge_list, input, r_index, self.readout_type
                # )
                # output = self.linear(torch.cat([input, update], dim=-1))
                # tail_readout_update = self.outward_readout(tail_read_out)
                # output += tail_readout_update
                # head_readout_update = self.inward_readout(head_read_out)
                # output += head_readout_update
                output = self.linear(torch.cat([input, update], dim=-1))
            else:
                if self.readout_type == "mean":
                    readout_update = input.mean(dim=1)
                elif self.readout_type == "sum":
                    readout_update = input.sum(dim=1)
                readout_update = readout_update.repeat((1, input.size(1), 1))
                output = self.linear(torch.cat([input, update], dim=-1))
                output += self.transform_readout(readout_update)
        else:
            output = self.linear(torch.cat([input, update], dim=-1))
            # output = input + update
        if self.layer_norm:
            # normed_output = self.layer_norm(output) 
            # normed_output[output.sum(-1)==0] = 0 # bad
            # output = normed_output 
            output = self.layer_norm(output) # 输入的全0向量输出不为0

        if self.activation:
            output = self.activation(output)
        return output
    
    def calculate_read_out(self, edge_list, input, r_index, readout_type):
        num_node, batch_size, dimension = input.shape

        # Create a mask to identify the relevant edge types in r_index
        r_index_expanded = r_index.unsqueeze(0).expand(edge_list.size(0), -1)
        edge_type_mask = edge_list[:, 2].unsqueeze(1) == r_index_expanded

        # Now create two node masks, one for tail and one for head; TODO: rename and swap the order
        tail_read_out_val = torch.zeros(1, batch_size, dimension, device=self.device)

        head_read_out_val = torch.zeros(1, batch_size, dimension, device=self.device)

        # TODO: Need to be further optimized by using torch masked sum
        for i in range(batch_size):
            # Find tail and head indices for each edge type
            tail_node_mask = torch.zeros(num_node, dtype=torch.bool, device=self.device)
            head_node_mask = torch.zeros(num_node, dtype=torch.bool, device=self.device)

            # tail_indices: all the source nodes with relation type q
            tail_indices = edge_list[edge_type_mask[:, i], 0]

            # head_indices: all the target node with relation type q
            head_indices = edge_list[edge_type_mask[:, i], 1]

            # Update the tail and head node masks
            tail_node_mask[tail_indices] = True
            head_node_mask[head_indices] = True

            # we read out node with only the value with input
            tail_masked_input = input[:, i, :] * tail_node_mask.unsqueeze(1).float()
            head_masked_input = input[:, i, :] * head_node_mask.unsqueeze(1).float()

            if readout_type == "sum":
                head_read_out_batch = head_masked_input.sum(dim=0)
                tail_read_out_batch = tail_masked_input.sum(dim=0)
            elif readout_type == "mean":
                head_read_out_batch = head_masked_input.mean(dim=0)
                tail_read_out_batch = tail_masked_input.mean(dim=0)
            else:
                raise NotImplementedError

            tail_read_out_val[:, i, :] = tail_read_out_batch.unsqueeze(0).unsqueeze(0)
            head_read_out_val[:, i, :] = head_read_out_batch.unsqueeze(0).unsqueeze(0)

        tail_read_out = tail_read_out_val.repeat(num_node, 1, 1)
        head_read_out = head_read_out_val.repeat(num_node, 1, 1)
        return tail_read_out, head_read_out
    

class MixtureOfTrackRelationalConv(GeneralizedRelationalConv):

    eps = 1e-6

    message2mul = {
        "transe": "add",
        "distmult": "mul",
    }

    # TODO for compile() - doesn't work currently
    # propagate_type = {"edge_index": torch.LongTensor, "size": Tuple[int, int]}

    def __init__(self, input_dim, output_dim, num_relation, query_input_dim, message_func="distmult",
                 aggregate_func="pna", layer_norm=False, activation="relu", has_readout=None, dependent=False, project_relations=False):
        super(GeneralizedRelationalConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.query_input_dim = query_input_dim
        self.message_func = message_func
        self.aggregate_func = aggregate_func
        self.dependent = dependent
        self.project_relations = project_relations
        self.readout_type = has_readout

        if self.readout_type is not None:
            if self.readout_type == "query":
                self.outward_readout = nn.Linear(input_dim, output_dim)
                self.inward_readout = nn.Linear(input_dim, output_dim)
            else:
                if self.readout_type == "mean":
                    self.readout = MeanReadout()
                elif self.readout_type == "sum":
                    self.readout = SumReadout()
                else:
                    raise NotImplementedError
            self.transform_readout = nn.Linear(input_dim, output_dim)

        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        if self.aggregate_func == "pna":
            self.linear = nn.Linear(input_dim * 13, output_dim)
        elif self.aggregate_func == "spna":
            self.linear = nn.Linear(input_dim * 7, output_dim)
        else:
            self.linear = nn.Linear(input_dim * 2, output_dim)  # bias = False

        self.relation = None
        self.relation_projection = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
        )

    def forward(self, input, query, boundary, edge_index, edge_type, size, edge_weight=None, r_index=None):
        batch_size = len(query)
        # print(self.num_relation, edge_type.max().item()+1)
        # self.num_relation = max(self.num_relation, edge_type.max().item()+1)
        if self.dependent:
            # layer-specific relation features as a projection of input "query" (relation) embeddings
            relation = self.relation_linear(query).view(batch_size, self.num_relation, self.input_dim)
        else:
            if not self.project_relations:
                # layer-specific relation features as a special embedding matrix unique to each layer
                relation = self.relation.weight.expand(batch_size, -1, -1)
            else:
                # NEW and only change: 
                # projecting relation features to unique features for this layer, then resizing for the current batch
                relation = self.relation_projection(self.relation)
        if edge_weight is None:
            edge_weight = torch.ones(len(edge_type), device=input.device)
        
        # note that we send the initial boundary condition (node states at layer0) to the message passing
        # correspond to Eq.6 on p5 in https://arxiv.org/pdf/2106.06935.pdf
        output = self.propagate(input=input, relation=relation, boundary=boundary, edge_index=edge_index,
                                edge_type=edge_type, size=size, edge_weight=edge_weight, r_index=r_index)
        return output

    def propagate(self, edge_index, size=None, **kwargs):
        if kwargs["edge_weight"].requires_grad or self.message_func == "rotate":
            # the rspmm cuda kernel only works for TransE and DistMult message functions
            # otherwise we invoke separate message & aggregate functions
            return super(GeneralizedRelationalConv, self).propagate(edge_index, size, **kwargs)

        for hook in self._propagate_forward_pre_hooks.values():
            res = hook(self, (edge_index, size, kwargs))
            if res is not None:
                edge_index, size, kwargs = res

        # in newer PyG, 
        # __check_input__ -> _check_input()
        # __collect__ -> _collect()
        # __fused_user_args__ -> _fuser_user_args
        size = self._check_input(edge_index, size)
        coll_dict = self._collect(self._fused_user_args, edge_index, size, kwargs)

        msg_aggr_kwargs = self.inspector.distribute("message_and_aggregate", coll_dict)
        for hook in self._message_and_aggregate_forward_pre_hooks.values():
            res = hook(self, (edge_index, msg_aggr_kwargs))
            if res is not None:
                edge_index, msg_aggr_kwargs = res
        out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)
        for hook in self._message_and_aggregate_forward_hooks.values():
            res = hook(self, (edge_index, msg_aggr_kwargs), out)
            if res is not None:
                out = res

        update_kwargs = self.inspector.distribute("update", coll_dict)
        out = self.update(out, **update_kwargs)

        for hook in self._propagate_forward_hooks.values():
            res = hook(self, (edge_index, size, kwargs), out)
            if res is not None:
                out = res

        return out

    def message(self, input_j, relation, boundary, edge_type):
        # print(relation.shape, relation.max(), self.node_dim, edge_type.min(), edge_type.max(), edge_type.shape)
        edge_type = torch.clamp(edge_type, 0, relation.shape[self.node_dim]-1)
        relation_j = relation.index_select(self.node_dim, edge_type)

        if self.message_func == "transe":
            message = input_j + relation_j
        elif self.message_func == "distmult":
            message = input_j * relation_j
        elif self.message_func == "rotate":
            x_j_re, x_j_im = input_j.chunk(2, dim=-1)
            r_j_re, r_j_im = relation_j.chunk(2, dim=-1)
            message_re = x_j_re * r_j_re - x_j_im * r_j_im
            message_im = x_j_re * r_j_im + x_j_im * r_j_re
            message = torch.cat([message_re, message_im], dim=-1)
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)
        # print(message.shape, boundary.shape, print(torch.cuda.memory_summary()))
        # augment messages with the boundary condition
        message = torch.cat([message, boundary], dim=self.node_dim)  # (num_edges + num_nodes, batch_size, input_dim)
        return message

    def aggregate(self, input, edge_weight, index, dim_size):
        # augment aggregation index with self-loops for the boundary condition
        dim_size = dim_size.item() if torch.is_tensor(dim_size) else dim_size
        index = torch.cat([index, torch.arange(dim_size, device=input.device)]) # (num_edges + num_nodes,)
        edge_weight = torch.cat([edge_weight, torch.ones(dim_size, device=input.device)])
        shape = [1] * input.ndim
        shape[self.node_dim] = -1
        edge_weight = edge_weight.view(shape)

        if self.aggregate_func == "pna":
            mean = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="mean")
            sq_mean = scatter(input ** 2 * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="mean")
            max = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="max")
            min = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="min")
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2)
            degree_out = degree(index, dim_size).unsqueeze(0).unsqueeze(-1)
            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            output = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        elif self.aggregate_func == "spna":
            mean = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="mean")
            sq_mean = scatter(input ** 2 * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="mean")
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2)
            degree_out = degree(index, dim_size).unsqueeze(0).unsqueeze(-1)
            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            output = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        else:
            output = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size,
                             reduce=self.aggregate_func)
        return output

    def message_and_aggregate(self, edge_index, input, relation, boundary, edge_type, edge_weight, index, dim_size, r_index=None):
        # fused computation of message and aggregate steps with the custom rspmm cuda kernel
        # speed up computation by several times
        # reduce memory complexity from O(|E|d) to O(|V|d), so we can apply it to larger graphs
        # print(relation.shape, relation.max(), self.node_dim, edge_type.min(), edge_type.max(), edge_type.shape)
        from .rspmm import generalized_rspmm

        batch_size, num_node = input.shape[:2]
        input = input.transpose(0, 1).flatten(1)
        relation = relation.transpose(0, 1).flatten(1)
        boundary = boundary.transpose(0, 1).flatten(1)
        degree_out = degree(index, dim_size).unsqueeze(-1) + 1
        
        # if r_index is not None and r_index.min()==r_index.max(): 
        #     n_rel = relation.size(0) // 2
        #     inverse_r_index = r_index.min()+n_rel if r_index.min() < n_rel else r_index.min()-n_rel
        #     # print(r_index, inverse_r_index, relation.shape, n_rel, len(edge_type.unique()))
        #     edge_mask = edge_type != inverse_r_index
        #     edge_index = edge_index[:, edge_mask]
        #     edge_type = edge_type[edge_mask]
        #     edge_weight = edge_weight[edge_mask]

        if self.message_func in self.message2mul:
            mul = self.message2mul[self.message_func]
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)
        # print(input.shape, relation.shape, index.shape)
        torch.cuda.synchronize()
        if self.aggregate_func == "sum":
            update = generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum="add", mul=mul)
            torch.cuda.synchronize()
            update = update.contiguous()
            update = update + boundary
        elif self.aggregate_func == "mean":
            update = generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum="add", mul=mul)
            update = (update + boundary) / degree_out
        elif self.aggregate_func == "max":
            update = generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum="max", mul=mul)
            update = torch.max(update, boundary)
        elif self.aggregate_func == "pna":
            # we use PNA with 4 aggregators (mean / max / min / std)
            # and 3 scalars (identity / log degree / reciprocal of log degree)
            sum = generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum="add", mul=mul)
            sq_sum = generalized_rspmm(edge_index, edge_type, edge_weight, relation ** 2, input ** 2, sum="add",
                                       mul=mul)
            max = generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum="max", mul=mul)
            min = generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum="min", mul=mul)
            mean = (sum + boundary) / degree_out
            sq_mean = (sq_sum + boundary ** 2) / degree_out
            max = torch.max(max, boundary)
            min = torch.min(min, boundary) # (node, batch_size * input_dim)
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2) # (node, batch_size * input_dim * 4)
            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1) # (node, 3)
            update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2) # (node, batch_size * input_dim * 4 * 3)
        elif self.aggregate_func == "spna":
            # we use PNA with 2 aggregators (mean / std)
            # and 3 scalars (identity / log degree / reciprocal of log degree)
            sum = generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum="add", mul=mul)
            sq_sum = generalized_rspmm(edge_index, edge_type, edge_weight, relation ** 2, input ** 2, sum="add",
                                       mul=mul)
            mean = (sum + boundary) / degree_out
            sq_mean = (sq_sum + boundary ** 2) / degree_out
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2) # (node, batch_size * input_dim * 4)
            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1) # (node, 3)
            update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2) # (node, batch_size * input_dim * 4 * 3)
        else:
            raise ValueError("Unknown aggregation function `%s`" % self.aggregate_func)
        update = update.view(num_node, batch_size, -1).transpose(0, 1)
        return update

    def update(self, update, input):
        if self.readout_type is not None:
            if self.readout_type == "query":
                ## Here we will aggregate all the rep with the specific query
                # edge_list = graph.edge_list
                # tail_read_out, head_read_out = self.calculate_read_out(
                #     edge_list, input, r_index, self.readout_type
                # )
                # output = self.linear(torch.cat([input, update], dim=-1))
                # tail_readout_update = self.outward_readout(tail_read_out)
                # output += tail_readout_update
                # head_readout_update = self.inward_readout(head_read_out)
                # output += head_readout_update
                output = self.linear(torch.cat([input, update], dim=-1))
            else:
                if self.readout_type == "mean":
                    readout_update = input.mean(dim=1)
                elif self.readout_type == "sum":
                    readout_update = input.sum(dim=1)
                readout_update = readout_update.repeat((1, input.size(1), 1))
                output = self.linear(torch.cat([input, update], dim=-1))
                output += self.transform_readout(readout_update)
        else:
            output = self.linear(torch.cat([input, update], dim=-1))
            # output = input + update
        if self.layer_norm:
            # output_mask = (output.sum(-1)!=0)
            # output[output_mask] = self.layer_norm(output[output_mask]) 
            output = self.layer_norm(output) # 输入的全0向量输出不为0
        if self.activation:
            output = self.activation(output)
        return output
    
    def calculate_read_out(self, edge_list, input, r_index, readout_type):
        num_node, batch_size, dimension = input.shape

        # Create a mask to identify the relevant edge types in r_index
        r_index_expanded = r_index.unsqueeze(0).expand(edge_list.size(0), -1)
        edge_type_mask = edge_list[:, 2].unsqueeze(1) == r_index_expanded

        # Now create two node masks, one for tail and one for head; TODO: rename and swap the order
        tail_read_out_val = torch.zeros(1, batch_size, dimension, device=self.device)

        head_read_out_val = torch.zeros(1, batch_size, dimension, device=self.device)

        # TODO: Need to be further optimized by using torch masked sum
        for i in range(batch_size):
            # Find tail and head indices for each edge type
            tail_node_mask = torch.zeros(num_node, dtype=torch.bool, device=self.device)
            head_node_mask = torch.zeros(num_node, dtype=torch.bool, device=self.device)

            # tail_indices: all the source nodes with relation type q
            tail_indices = edge_list[edge_type_mask[:, i], 0]

            # head_indices: all the target node with relation type q
            head_indices = edge_list[edge_type_mask[:, i], 1]

            # Update the tail and head node masks
            tail_node_mask[tail_indices] = True
            head_node_mask[head_indices] = True

            # we read out node with only the value with input
            tail_masked_input = input[:, i, :] * tail_node_mask.unsqueeze(1).float()
            head_masked_input = input[:, i, :] * head_node_mask.unsqueeze(1).float()

            if readout_type == "sum":
                head_read_out_batch = head_masked_input.sum(dim=0)
                tail_read_out_batch = tail_masked_input.sum(dim=0)
            elif readout_type == "mean":
                head_read_out_batch = head_masked_input.mean(dim=0)
                tail_read_out_batch = tail_masked_input.mean(dim=0)
            else:
                raise NotImplementedError

            tail_read_out_val[:, i, :] = tail_read_out_batch.unsqueeze(0).unsqueeze(0)
            head_read_out_val[:, i, :] = head_read_out_batch.unsqueeze(0).unsqueeze(0)

        tail_read_out = tail_read_out_val.repeat(num_node, 1, 1)
        head_read_out = head_read_out_val.repeat(num_node, 1, 1)
        return tail_read_out, head_read_out
  

class MoEEnhancedRelationalConv(GeneralizedRelationalConv):

    def __init__(self, input_dim, output_dim, expert_num, noise_k, num_relation, query_input_dim, message_func="distmult",
                 aggregate_func="pna", layer_norm=False, activation="relu", dependent=False, project_relations=False):
        super(MoEEnhancedRelationalConv, self).__init__(input_dim, output_dim, num_relation, query_input_dim, message_func, 
                                                        aggregate_func, layer_norm, activation, dependent, project_relations)
        self.expert_num = expert_num # 4
        self.noise_k = noise_k # 2
        
        if self.aggregate_func == "pna":
            dim_scale = 13
        elif self.aggregate_func == "spna":
            dim_scale = 7
        else:
            dim_scale = 2
            
        # self.linear = nn.Linear(input_dim * dim_scale, output_dim)
        # self.layer_norm = nn.LayerNorm(output_dim)
        
        fnn_list = torch.nn.ModuleList()
        ln_list = torch.nn.ModuleList()
        for expert_idx in range(self.expert_num):
            fnn_list.append(nn.Linear(input_dim * dim_scale, output_dim))
            ln_list.append(nn.LayerNorm(output_dim))
        self.moe_linearln = MultiConvMoE(input_size=input_dim, output_size=output_dim, num_experts=expert_num, 
                                experts_conv=fnn_list, experts_bn=ln_list,  k=noise_k, layer_type="ent_conv_fnn")
        
    
    def forward(self, input, query, boundary, edge_index, edge_type, size, edge_weight=None):
        batch_size = len(query)

        if self.dependent:
            # layer-specific relation features as a projection of input "query" (relation) embeddings
            relation = self.relation_linear(query).view(batch_size, self.num_relation, self.input_dim)
        else:
            if not self.project_relations:
                # layer-specific relation features as a special embedding matrix unique to each layer
                relation = self.relation.weight.expand(batch_size, -1, -1)
            else:
                # NEW and only change: 
                # projecting relation features to unique features for this layer, then resizing for the current batch
                relation = self.relation_projection(self.relation)
        if edge_weight is None:
            edge_weight = torch.ones(len(edge_type), device=input.device)

        # note that we send the initial boundary condition (node states at layer0) to the message passing
        # correspond to Eq.6 on p5 in https://arxiv.org/pdf/2106.06935.pdf
        output, moe_loss = self.propagate(input=input, query=query, relation=relation, boundary=boundary, edge_index=edge_index,
                                edge_type=edge_type, size=size, edge_weight=edge_weight)
        return output, moe_loss
        
    def propagate(self, edge_index, size=None, **kwargs):
        if kwargs["edge_weight"].requires_grad or self.message_func == "rotate":
            # the rspmm cuda kernel only works for TransE and DistMult message functions
            # otherwise we invoke separate message & aggregate functions
            return super(GeneralizedRelationalConv, self).propagate(edge_index, size, **kwargs)

        for hook in self._propagate_forward_pre_hooks.values():
            res = hook(self, (edge_index, size, kwargs))
            if res is not None:
                edge_index, size, kwargs = res

        # in newer PyG, 
        # __check_input__ -> _check_input()
        # __collect__ -> _collect()
        # __fused_user_args__ -> _fuser_user_args
        size = self._check_input(edge_index, size)
        coll_dict = self._collect(self._fused_user_args, edge_index, size, kwargs)

        msg_aggr_kwargs = self.inspector.distribute("message_and_aggregate", coll_dict)
        for hook in self._message_and_aggregate_forward_pre_hooks.values():
            res = hook(self, (edge_index, msg_aggr_kwargs))
            if res is not None:
                edge_index, msg_aggr_kwargs = res
        out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)
        for hook in self._message_and_aggregate_forward_hooks.values():
            res = hook(self, (edge_index, msg_aggr_kwargs), out)
            if res is not None:
                out = res

        update_kwargs = self.inspector.distribute("update", coll_dict)
        update_kwargs["query"] = kwargs["query"]
        out, moe_loss = self.update(out, **update_kwargs) 
                
        for hook in self._propagate_forward_hooks.values():
            res = hook(self, (edge_index, size, kwargs), out)
            if res is not None:
                out = res

        return out, moe_loss
    
    def update(self, update, input, query):
        # node update as a function of old states (input) and this layer output (update)
        # output = self.linear(torch.cat([input, update], dim=-1))
        # if self.layer_norm:
        #     output = self.layer_norm(output)
        # moe_loss = 0
        moe_input = torch.cat([input, update], dim=-1)
        kwargs = {"layer_input":moe_input, "query":query}
        output, moe_loss = self.moe_linearln.forward(kwargs)

        if self.activation:
            output = self.activation(output)
        return output, moe_loss

# no linear and resdual parts in the update function
class SimplifiedRelationalConv(MessagePassing):   

    eps = 1e-6

    message2mul = {
        "transe": "add",
        "distmult": "mul",
    }

    # TODO for compile() - doesn't work currently
    # propagate_type = {"edge_index": torch.LongTensor, "size": Tuple[int, int]}

    def __init__(self, input_dim, output_dim, num_relation, query_input_dim, message_func="distmult",
                 aggregate_func="pna", layer_norm=False, activation="relu", dependent=False, project_relations=False):
        super(SimplifiedRelationalConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.query_input_dim = query_input_dim
        self.message_func = message_func
        self.aggregate_func = aggregate_func
        self.dependent = dependent
        self.project_relations = project_relations

        if dependent:
            # obtain relation embeddings as a projection of the query relation
            self.relation_linear = nn.Linear(query_input_dim, num_relation * input_dim)
        else:
            if not self.project_relations:
                # relation embeddings as an independent embedding matrix per each layer
                self.relation = nn.Embedding(num_relation, input_dim)
            else:
                # will be initialized after the pass over relation graph
                self.relation = None
                self.relation_projection = nn.Sequential(
                    nn.Linear(input_dim, input_dim),
                    nn.ReLU(),
                    nn.Linear(input_dim, input_dim)
                )


    def forward(self, input, query, boundary, edge_index, edge_type, size, edge_weight=None):
        batch_size = len(query)

        if self.dependent:
            # layer-specific relation features as a projection of input "query" (relation) embeddings
            relation = self.relation_linear(query).view(batch_size, self.num_relation, self.input_dim)
        else:
            if not self.project_relations:
                # layer-specific relation features as a special embedding matrix unique to each layer
                relation = self.relation.weight.expand(batch_size, -1, -1)
            else:
                # NEW and only change: 
                # projecting relation features to unique features for this layer, then resizing for the current batch
                relation = self.relation_projection(self.relation)
        if edge_weight is None:
            edge_weight = torch.ones(len(edge_type), device=input.device)

        # note that we send the initial boundary condition (node states at layer0) to the message passing
        # correspond to Eq.6 on p5 in https://arxiv.org/pdf/2106.06935.pdf
        output = self.propagate(input=input, relation=relation, boundary=boundary, edge_index=edge_index,
                                edge_type=edge_type, size=size, edge_weight=edge_weight)
        return output

    def propagate(self, edge_index, size=None, **kwargs):
        if kwargs["edge_weight"].requires_grad or self.message_func == "rotate":
            # the rspmm cuda kernel only works for TransE and DistMult message functions
            # otherwise we invoke separate message & aggregate functions
            return super(SimplifiedRelationalConv, self).propagate(edge_index, size, **kwargs)

        for hook in self._propagate_forward_pre_hooks.values():
            res = hook(self, (edge_index, size, kwargs))
            if res is not None:
                edge_index, size, kwargs = res

        # in newer PyG, 
        # __check_input__ -> _check_input()
        # __collect__ -> _collect()
        # __fused_user_args__ -> _fuser_user_args
        size = self._check_input(edge_index, size)
        coll_dict = self._collect(self._fused_user_args, edge_index, size, kwargs)

        msg_aggr_kwargs = self.inspector.distribute("message_and_aggregate", coll_dict)
        for hook in self._message_and_aggregate_forward_pre_hooks.values():
            res = hook(self, (edge_index, msg_aggr_kwargs))
            if res is not None:
                edge_index, msg_aggr_kwargs = res
        out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)
        for hook in self._message_and_aggregate_forward_hooks.values():
            res = hook(self, (edge_index, msg_aggr_kwargs), out)
            if res is not None:
                out = res

        update_kwargs = self.inspector.distribute("update", coll_dict)
        out = self.update(out, **update_kwargs)

        for hook in self._propagate_forward_hooks.values():
            res = hook(self, (edge_index, size, kwargs), out)
            if res is not None:
                out = res

        return out

    def message(self, input_j, relation, boundary, edge_type):
        relation_j = relation.index_select(self.node_dim, edge_type)

        if self.message_func == "transe":
            message = input_j + relation_j
        elif self.message_func == "distmult":
            message = input_j * relation_j
        elif self.message_func == "rotate":
            x_j_re, x_j_im = input_j.chunk(2, dim=-1)
            r_j_re, r_j_im = relation_j.chunk(2, dim=-1)
            message_re = x_j_re * r_j_re - x_j_im * r_j_im
            message_im = x_j_re * r_j_im + x_j_im * r_j_re
            message = torch.cat([message_re, message_im], dim=-1)
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)

        # augment messages with the boundary condition
        message = torch.cat([message, boundary], dim=self.node_dim)  # (num_edges + num_nodes, batch_size, input_dim)

        return message

    def aggregate(self, input, edge_weight, index, dim_size):
        # augment aggregation index with self-loops for the boundary condition
        index = torch.cat([index, torch.arange(dim_size, device=input.device)]) # (num_edges + num_nodes,)
        edge_weight = torch.cat([edge_weight, torch.ones(dim_size, device=input.device)])
        shape = [1] * input.ndim
        shape[self.node_dim] = -1
        edge_weight = edge_weight.view(shape)

        output = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size,
                             reduce=self.aggregate_func)

        return output

    def message_and_aggregate(self, edge_index, input, relation, boundary, edge_type, edge_weight, index, dim_size):
        # fused computation of message and aggregate steps with the custom rspmm cuda kernel
        # speed up computation by several times
        # reduce memory complexity from O(|E|d) to O(|V|d), so we can apply it to larger graphs
        from .rspmm import generalized_rspmm

        batch_size, num_node = input.shape[:2]
        input = input.transpose(0, 1).flatten(1)
        relation = relation.transpose(0, 1).flatten(1)
        boundary = boundary.transpose(0, 1).flatten(1)
        degree_out = degree(index, dim_size).unsqueeze(-1) + 1

        if self.message_func in self.message2mul:
            mul = self.message2mul[self.message_func]
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)

        update = generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum="add", mul=mul)
        # print(update.device, boundary.device)
        # update = update + boundary

        update = update.view(num_node, batch_size, -1).transpose(0, 1)
        return update

    def update(self, update, input):
        # node update as a function of old states (input) and this layer output (update)
        output = update
        return output