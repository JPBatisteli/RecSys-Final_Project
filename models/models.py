import dgl.nn
import torch.nn as nn
import math
from tqdm import tqdm
import torch as th
import pdb
import torch.nn.functional as F
import torch
import dgl
import dgl.function as fn
import dgl.nn as dglnn
from dgl.nn import GraphConv
from dgl import LaplacianPE, RandomWalkPE 
from models.layers import DGRecLayer

class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype = etype)
            return graph.edges[etype].data['score']

class BaseGraphModel(nn.Module):
    def __init__(self, args, dataloader):
        super().__init__()
        self.args = args
        self.hid_dim = args.embed_size
        self.layer_num = args.layers
        self.graph = dataloader.train_graph
        self.user_number = dataloader.user_number
        self.item_number = dataloader.item_number
        
        self.user_embedding = torch.nn.Parameter(torch.randn(self.graph.nodes('user').shape[0], self.hid_dim))
        self.item_embedding = torch.nn.Parameter(torch.randn(self.graph.nodes('item').shape[0], self.hid_dim))
        self.predictor = HeteroDotProductPredictor()
        self.build_model()
        self.node_features = {'user': self.user_embedding, 'item': self.item_embedding}

    def build_layer(self, idx):
        pass

    def build_model(self):
        self.layers = nn.ModuleList()
        for idx in range(self.layer_num):
            h2h = self.build_layer(idx)
            self.layers.append(h2h)

    def get_embedding(self):
        h = self.node_features
        graph_user2item = dgl.edge_type_subgraph(self.graph, ['rate'])
        graph_item2user = dgl.edge_type_subgraph(self.graph, ['rated by'])
        
        for layer in self.layers:
            user_feat = h['user']
            item_feat = h['item']

            h_item = layer(graph_user2item, (user_feat, item_feat))
            h_user = layer(graph_item2user, (item_feat, user_feat))

            h = {'user': h_user, 'item': h_item}
        return h

    def forward(self, graph_pos, graph_neg):
        h = self.get_embedding()
        score_pos = self.predictor(graph_pos, h, 'rate')
        score_neg = self.predictor(graph_neg, h, 'rate')
        return score_pos, score_neg

    def get_score(self, h, users):
        user_embed = h['user'][users]
        item_embed = h['item']
        scores = torch.mm(user_embed, item_embed.t())
        return scores
    
class DGRec(BaseGraphModel):
    def __init__(self, args, dataloader):
        super(DGRec, self).__init__(args, dataloader)
        # Single projection matrix instead of multiple heads
        self.W = torch.nn.Parameter(torch.randn(self.args.embed_size, self.args.embed_size))
        # Lightweight attention vector
        self.a = torch.nn.Parameter(torch.randn(self.args.embed_size))
        # Optional: Add layer normalization for stability
        self.layer_norm = nn.LayerNorm(self.args.embed_size)
        # Gating mechanism parameters
        self.gate_W = torch.nn.Parameter(torch.randn(self.args.embed_size, 1))
        
    def gated_attention(self, layers):
        # Convert list of layers to tensor
        tensor_layers = torch.stack(layers, dim=0)
        
        projected = torch.matmul(tensor_layers, self.W)  # [num_layers, batch_size, embed_dim]
        
        # Compute attention scores with reduced computations
        scores = torch.matmul(projected, self.a)  # [num_layers, batch_size]
        attention_weights = F.softmax(scores, dim=0).unsqueeze(-1)  # [num_layers, batch_size, 1]
        
        # Compute gates for each layer
        gates = torch.sigmoid(torch.matmul(tensor_layers, self.gate_W))  # [num_layers, batch_size, 1]
        
        # Combine attention and gating
        combined_weights = attention_weights * gates
        
        # Normalize weights
        combined_weights = combined_weights / (combined_weights.sum(dim=0, keepdim=True) + 1e-8)
        
        # Weight and sum the layers
        weighted_sum = torch.sum(tensor_layers * combined_weights, dim=0)  # [batch_size, embed_dim]
        
        # Optional: Apply layer normalization
        output = self.layer_norm(weighted_sum)
        return output
    
    def build_layer(self, idx):
        return DGRecLayer(self.args)

    def layer_attention(self, ls, W, a):
        tensor_layers = torch.stack(ls, dim = 0)
        weight = torch.matmul(tensor_layers, W)
        weight = F.softmax(torch.matmul(weight, a), dim = 0).unsqueeze(-1)
        tensor_layers = torch.sum(tensor_layers * weight, dim = 0)
        return tensor_layers

    def get_embedding(self):
        user_embed = [self.user_embedding]
        item_embed = [self.item_embedding]
        h = self.node_features

        for layer in self.layers:

            h_item = layer(self.graph, h, ('user', 'rate', 'item'))
            h_user = layer(self.graph, h, ('item', 'rated by', 'user'))
            h = {'user': h_user, 'item': h_item}
            user_embed.append(h_user)
            item_embed.append(h_item)
        # user_embed = self.layer_attention(user_embed, self.W, self.a) #([82633, 32])
        # item_embed = self.layer_attention(item_embed, self.W, self.a) #([136710, 32])
        user_embed = self.gated_attention(user_embed)
        item_embed = self.gated_attention(item_embed)
        h = {'user': user_embed, 'item': item_embed}
        return h

#Idea 1 -> enrich the representations
#Idea 2 -> change the layer attention mechanism
class DGRec_mod1(BaseGraphModel):
    def __init__(self, args, dataloader):
        super(DGRec_mod1, self).__init__(args, dataloader)
        self.W = torch.nn.Parameter(torch.randn(self.args.embed_size, self.args.embed_size))
        self.a = torch.nn.Parameter(torch.randn(self.args.embed_size))
        #Change 1:
        self.user_linear1 = torch.nn.Linear(args.embed_size, args.embed_size)
        self.user_linear2 = torch.nn.Linear(args.embed_size, args.embed_size)
        self.user_bn1 = torch.nn.BatchNorm1d(args.embed_size)
        self.user_bn2 = torch.nn.BatchNorm1d(args.embed_size)
        self.user_bn3 = torch.nn.BatchNorm1d(args.embed_size)
        self.item_linear1 = torch.nn.Linear(args.embed_size, args.embed_size)
        self.item_linear2 = torch.nn.Linear(args.embed_size, args.embed_size)
        self.item_bn1 = torch.nn.BatchNorm1d(args.embed_size)
        self.item_bn2 = torch.nn.BatchNorm1d(args.embed_size)
        self.item_bn3 = torch.nn.BatchNorm1d(args.embed_size)
        
    def build_layer(self, idx):
        return DGRecLayer(self.args)

    def trans1(self, h, linear, bn):
        h = linear(h)
        h = bn(h)
        return h
    
    def trans2(self, h, linear, bn1, bn2):
        h = bn1(h)
        h = linear(h)
        h = bn2(h)
        return h
    
    def layer_attention(self, ls, W, a):
        tensor_layers = torch.stack(ls, dim = 0)
        weight = torch.matmul(tensor_layers, W)
        weight = F.softmax(torch.matmul(weight, a), dim = 0).unsqueeze(-1)
        tensor_layers = torch.sum(tensor_layers * weight, dim = 0)
        return tensor_layers

    def get_embedding(self):
        user_embed = [self.user_embedding]
        item_embed = [self.item_embedding]
        h = self.node_features

        for layer in self.layers:
            #Additional transformation:
            h_item = self.trans1(h['item'], self.item_linear1, self.item_bn1)
            h_user = self.trans1(h['user'], self.user_linear1, self.user_bn1)
            h = {'user': h_user, 'item': h_item}
            
            h_item = layer(self.graph, h, ('user', 'rate', 'item')) #([136710, 32])
            h_user = layer(self.graph, h, ('item', 'rated by', 'user')) #([82633, 32])
            
            h_item = self.trans2(h_item, self.item_linear2, self.item_bn2, self.item_bn3)
            h_user = self.trans2(h_user, self.user_linear2, self.user_bn2, self.user_bn3)
            
            h = {'user': h_user, 'item': h_item}
            user_embed.append(h_user)
            item_embed.append(h_item)
        user_embed = self.layer_attention(user_embed, self.W, self.a)
        item_embed = self.layer_attention(item_embed, self.W, self.a)
        h = {'user': user_embed, 'item': item_embed}
        return h

