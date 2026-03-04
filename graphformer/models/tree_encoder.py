"""
Tree Encoder Components for Neuron Morphology Analysis

This module contains TreeLSTM and related components for encoding tree structures.
"""

import torch
import torch.nn as nn
import dgl


class MultiNodeAggregation(nn.Module):
    """Attention-based aggregation over all tree nodes instead of just root"""
    def __init__(self, h_size, aggregation_type="attention"):
        super(MultiNodeAggregation, self).__init__()
        self.h_size = h_size
        self.aggregation_type = aggregation_type

        if aggregation_type == "attention":
            # Learnable attention over nodes
            self.attention = nn.Sequential(
                nn.Linear(h_size, h_size),
                nn.Tanh(),
                nn.Linear(h_size, 1)
            )
        elif aggregation_type == "weighted":
            # Learnable weighted sum
            self.weight_net = nn.Sequential(
                nn.Linear(h_size, h_size // 2),
                nn.ReLU(),
                nn.Linear(h_size // 2, 1),
                nn.Sigmoid()
            )

    def forward(self, g, node_features, offsets):
        """
        Args:
            g: DGL graph
            node_features: (N, h_size) features for all nodes
            offsets: indices of root nodes for each tree in batch

        Returns:
            aggregated: (batch_size, h_size) aggregated features
        """
        batch_size = len(offsets)
        aggregated = []

        # For each tree in batch
        for i in range(batch_size):
            # Get start and end indices for this tree's nodes
            if i < batch_size - 1:
                start_idx = offsets[i] if i == 0 else offsets[i-1]
                end_idx = offsets[i+1]
            else:
                start_idx = offsets[i-1] if i > 0 else 0
                end_idx = len(node_features)

            # Extract this tree's node features
            tree_feats = node_features[start_idx:end_idx]  # (num_nodes_i, h_size)

            if self.aggregation_type == "attention":
                # Attention weights
                attn_scores = self.attention(tree_feats)  # (num_nodes_i, 1)
                attn_weights = torch.softmax(attn_scores, dim=0)

                # Weighted sum
                agg = (tree_feats * attn_weights).sum(dim=0)  # (h_size,)

            elif self.aggregation_type == "weighted":
                # Simple learnable weighting
                weights = self.weight_net(tree_feats)  # (num_nodes_i, 1)
                agg = (tree_feats * weights).sum(dim=0)  # (h_size,)

            elif self.aggregation_type == "mean":
                agg = tree_feats.mean(dim=0)  # (h_size,)

            elif self.aggregation_type == "max":
                agg = tree_feats.max(dim=0)[0]  # (h_size,)

            aggregated.append(agg)

        return torch.stack(aggregated, dim=0)  # (batch_size, h_size)


class TreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size, mode="sum"):
        super(TreeLSTMCell, self).__init__()
        self.h_size, self.mode = h_size, mode
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(h_size, h_size)

    def message_func(self, edges):
        return {"h": edges.src["h"], "c": edges.src["c"]}

    def reduce_func(self, nodes):
        if self.mode == "sum":
            h_cat = nodes.mailbox["h"].sum(dim=1)
        elif self.mode == "max":
            h_cat = nodes.mailbox["h"].max(dim=1)[0]
        elif self.mode == "mean":
            h_cat = nodes.mailbox["h"].mean(dim=1)
        else:
            raise NotImplementedError

        f = torch.sigmoid(self.U_f(nodes.mailbox["h"]))
        c = torch.sum(f * nodes.mailbox["c"], 1)
        return {"iou": self.U_iou(h_cat), "c": c}

    def apply_node_func(self, nodes):
        iou = nodes.data["iou"] + self.b_iou
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        c = i * u + nodes.data["c"]
        h = o * torch.tanh(c)
        return {"h": h, "c": c}


class BidirectionalTreeLSTMCell(nn.Module):
    """Bidirectional TreeLSTM Cell that processes tree in both bottom-up and top-down directions"""
    def __init__(self, x_size, h_size, mode="sum"):
        super(BidirectionalTreeLSTMCell, self).__init__()
        self.h_size = h_size
        self.mode = mode

        # Bottom-up (child to parent) parameters
        self.W_iou_bu = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou_bu = nn.Linear(h_size, 3 * h_size, bias=False)
        self.b_iou_bu = nn.Parameter(torch.zeros(1, 3 * h_size))
        self.U_f_bu = nn.Linear(h_size, h_size)

        # Top-down (parent to child) parameters
        self.W_iou_td = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou_td = nn.Linear(h_size, 3 * h_size, bias=False)
        self.b_iou_td = nn.Parameter(torch.zeros(1, 3 * h_size))
        self.U_f_td = nn.Linear(h_size, h_size)

    def message_func_bu(self, edges):
        """Bottom-up message function (from children to parent)"""
        return {"h_bu": edges.src["h_bu"], "c_bu": edges.src["c_bu"]}

    def reduce_func_bu(self, nodes):
        """Bottom-up reduce function"""
        if self.mode == "sum":
            h_cat = nodes.mailbox["h_bu"].sum(dim=1)
        elif self.mode == "max":
            h_cat = nodes.mailbox["h_bu"].max(dim=1)[0]
        elif self.mode == "mean":
            h_cat = nodes.mailbox["h_bu"].mean(dim=1)
        else:
            raise NotImplementedError

        f = torch.sigmoid(self.U_f_bu(nodes.mailbox["h_bu"]))
        c = torch.sum(f * nodes.mailbox["c_bu"], 1)
        return {"iou_bu": self.U_iou_bu(h_cat), "c_bu": c}

    def apply_node_func_bu(self, nodes):
        """Bottom-up apply node function"""
        iou = nodes.data["iou_bu"] + self.b_iou_bu
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        c = i * u + nodes.data["c_bu"]
        h = o * torch.tanh(c)
        return {"h_bu": h, "c_bu": c}

    def message_func_td(self, edges):
        """Top-down message function (from parent to children)"""
        return {"h_td": edges.dst["h_td"], "c_td": edges.dst["c_td"]}

    def reduce_func_td(self, nodes):
        """Top-down reduce function"""
        if nodes.mailbox["h_td"].shape[1] == 0:
            # Root node has no parent
            return {"iou_td": torch.zeros(nodes.batch_size(), 3 * self.h_size, device=nodes.mailbox["h_td"].device),
                    "c_td": torch.zeros(nodes.batch_size(), self.h_size, device=nodes.mailbox["h_td"].device)}

        # For non-root nodes, aggregate parent information
        h_parent = nodes.mailbox["h_td"][:, 0, :]  # Take first (and only) parent
        c_parent = nodes.mailbox["c_td"][:, 0, :]

        return {"iou_td": self.U_iou_td(h_parent), "c_td": c_parent}

    def apply_node_func_td(self, nodes):
        """Top-down apply node function"""
        iou = nodes.data["iou_td"] + self.b_iou_td
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        c = i * u + nodes.data["c_td"]
        h = o * torch.tanh(c)
        return {"h_td": h, "c_td": c}


class BidirectionalTreeLSTM(nn.Module):
    """Bidirectional TreeLSTM that combines bottom-up and top-down processing"""
    def __init__(self, x_size, h_size, num_classes, mode="sum", fc=True, bn=False,
                 node_aggregation=None):
        super(BidirectionalTreeLSTM, self).__init__()
        self.x_size = x_size
        self.h_size = h_size
        self.node_aggregation = node_aggregation

        # Input projection
        if bn:
            self.mlp1 = nn.Sequential(
                nn.Linear(x_size, h_size),
                nn.BatchNorm1d(h_size),
                nn.ReLU(),
            )
        else:
            self.mlp1 = nn.Sequential(
                nn.Linear(x_size, h_size),
                nn.ReLU(),
            )

        # Bidirectional TreeLSTM cell
        self.cell = BidirectionalTreeLSTMCell(h_size, h_size, mode=mode)

        # Node aggregation
        if node_aggregation:
            self.node_agg = MultiNodeAggregation(h_size * 2, aggregation_type=node_aggregation)

        # Classification head
        self.fc = fc
        if fc:
            self.linear = nn.Linear(h_size * 2, num_classes)

    def forward(self, batch):
        """Forward pass combining bottom-up and top-down TreeLSTM"""
        g = batch.graph.to(torch.device("cuda"))
        g = dgl.graph(g.edges())
        n = g.number_of_nodes()

        # Input projection
        feats = self.mlp1(batch.feats.cuda())

        # Initialize node states for both directions
        g.ndata["iou_bu"] = self.cell.W_iou_bu(feats)
        g.ndata["iou_td"] = self.cell.W_iou_td(feats)
        g.ndata["h_bu"] = torch.zeros((n, self.h_size)).cuda()
        g.ndata["c_bu"] = torch.zeros((n, self.h_size)).cuda()
        g.ndata["h_td"] = torch.zeros((n, self.h_size)).cuda()
        g.ndata["c_td"] = torch.zeros((n, self.h_size)).cuda()

        # Bottom-up pass
        dgl.prop_nodes_topo(
            g,
            message_func=self.cell.message_func_bu,
            reduce_func=self.cell.reduce_func_bu,
            apply_node_func=self.cell.apply_node_func_bu,
        )

        # Top-down pass (reverse topological order)
        dgl.prop_nodes_topo(
            g,
            message_func=self.cell.message_func_td,
            reduce_func=self.cell.reduce_func_td,
            apply_node_func=self.cell.apply_node_func_td,
            reverse=True,
        )

        # Combine bottom-up and top-down representations
        h_combined = torch.cat([g.ndata.pop("c_bu"), g.ndata.pop("c_td")], dim=1)

        # Aggregate node features
        if self.node_aggregation:
            h = self.node_agg(g, h_combined, batch.offset.long())
        else:
            h = h_combined[batch.offset.long()]

        # Classification
        if self.fc:
            return self.linear(h)
        return h


class TreeLSTM(nn.Module):
    def __init__(self, x_size, h_size, num_classes, mode="sum", fc=True, bn=False,
                 node_aggregation=None):
        super(TreeLSTM, self).__init__()
        self.x_size = x_size
        self.h_size = h_size
        self.node_aggregation = node_aggregation

        if bn:
            self.mlp1 = nn.Sequential(
                nn.Linear(x_size, h_size),
                nn.BatchNorm1d(h_size),
                nn.ReLU(),
            )
        else:
            self.mlp1 = nn.Sequential(
                nn.Linear(x_size, h_size),
                nn.ReLU(),
            )

        self.cell = TreeLSTMCell(h_size, h_size, mode=mode)

        if node_aggregation:
            self.node_agg = MultiNodeAggregation(h_size, aggregation_type=node_aggregation)

        self.fc = fc
        if fc:
            self.linear = nn.Linear(h_size, num_classes)

    def forward(self, batch):
        g = batch.graph.to(torch.device("cuda"))
        g = dgl.graph(g.edges())
        n = g.number_of_nodes()

        feats = self.mlp1(batch.feats.cuda())
        g.ndata["iou"] = self.cell.W_iou(feats)
        g.ndata["h"] = torch.zeros((n, self.h_size)).cuda()
        g.ndata["c"] = torch.zeros((n, self.h_size)).cuda()

        dgl.prop_nodes_topo(
            g,
            message_func=self.cell.message_func,
            reduce_func=self.cell.reduce_func,
            apply_node_func=self.cell.apply_node_func,
        )

        h = g.ndata.pop("c")

        if self.node_aggregation:
            h = self.node_agg(g, h, batch.offset.long())
        else:
            h = h[batch.offset.long()]

        if self.fc:
            return self.linear(h)
        return h


class TreeLSTM_wo_MLP(nn.Module):
    """TreeLSTM without initial MLP projection"""
    def __init__(self, x_size, h_size, num_classes, mode="sum", fc=True):
        super(TreeLSTM_wo_MLP, self).__init__()
        self.x_size = x_size
        self.h_size = h_size
        self.cell = TreeLSTMCell(x_size, h_size, mode=mode)
        self.fc = fc
        if fc:
            self.linear = nn.Linear(h_size, num_classes)

    def forward(self, batch):
        g = batch.graph.to(torch.device("cuda"))
        g = dgl.graph(g.edges())
        n = g.number_of_nodes()

        feats = batch.feats.cuda()
        g.ndata["iou"] = self.cell.W_iou(feats)
        g.ndata["h"] = torch.zeros((n, self.h_size)).cuda()
        g.ndata["c"] = torch.zeros((n, self.h_size)).cuda()

        dgl.prop_nodes_topo(
            g,
            message_func=self.cell.message_func,
            reduce_func=self.cell.reduce_func,
            apply_node_func=self.cell.apply_node_func,
        )

        h = g.ndata.pop("c")[batch.offset.long()]

        if self.fc:
            return self.linear(h)
        return h


class TreeLSTMCellv2(nn.Module):
    """TreeLSTM Cell with alternative architecture"""
    def __init__(self, x_size, h_size, mode="sum"):
        super(TreeLSTMCellv2, self).__init__()
        self.h_size = h_size
        self.mode = mode
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(2 * h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(2 * h_size, h_size)

    def message_func(self, edges):
        return {"h": edges.src["h"], "c": edges.src["c"]}

    def reduce_func(self, nodes):
        if self.mode == "sum":
            h_cat = nodes.mailbox["h"].sum(dim=1)
        elif self.mode == "max":
            h_cat = nodes.mailbox["h"].max(dim=1)[0]
        elif self.mode == "mean":
            h_cat = nodes.mailbox["h"].mean(dim=1)

        h_max = nodes.mailbox["h"].max(dim=1)[0]
        h_combined = torch.cat([h_cat, h_max], dim=1)

        f = torch.sigmoid(self.U_f(h_combined.unsqueeze(1).expand(-1, nodes.mailbox["h"].shape[1], -1)))
        c = torch.sum(f * nodes.mailbox["c"], 1)
        return {"iou": self.U_iou(h_combined), "c": c}

    def apply_node_func(self, nodes):
        iou = nodes.data["iou"] + self.b_iou
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        c = i * u + nodes.data["c"]
        h = o * torch.tanh(c)
        return {"h": h, "c": c}


class TreeLSTMDoubleCell(nn.Module):
    """TreeLSTM Cell with double hidden state (original implementation)"""
    def __init__(self, x_size, h_size, mode="sum"):
        super(TreeLSTMDoubleCell, self).__init__()
        self.W1_iouf = nn.Linear(x_size, 4 * h_size)
        self.U1_iouf = nn.Linear(h_size, 4 * h_size)
        self.W2_iouf = nn.Linear(h_size, 4 * h_size)
        self.U2_iouf = nn.Linear(h_size, 4 * h_size)
        self.mode = mode
        self.init_state = True
        self.h_size = h_size

    def message_func(self, edges):
        return {
            "h1": edges.src["h1"],
            "c1": edges.src["c1"],
            "h2": edges.src["h2"],
            "c2": edges.src["c2"],
        }

    def reduce_func(self, nodes):
        h1, c1 = nodes.mailbox["h1"], nodes.mailbox["c1"]
        h2, c2 = nodes.mailbox["h2"], nodes.mailbox["c2"]
        if self.mode == "sum":
            h1, c1, h2, c2 = h1.sum(-2), c1.sum(-2), h2.sum(-2), c2.sum(-2)
        elif self.mode == "mean":
            h1, c1, h2, c2 = h1.mean(-2), c1.mean(-2), h2.mean(-2), c2.mean(-2)
        else:
            raise ValueError("must in [sum, mean]")
        x_iouf = nodes.data["iouf"]
        xi, xo, xu, xf = torch.chunk(x_iouf, 4, 1)
        h_iouf1 = self.U1_iouf(h1)
        hi1, ho1, hu1, hf1 = torch.chunk(h_iouf1, 4, 1)
        i = torch.sigmoid(xi + hi1)
        f = torch.sigmoid(xf + hf1)
        o = torch.sigmoid(xo + ho1)
        u = torch.tanh(xu + hu1)
        c1 = i * u + f * c1
        h1 = o * torch.tanh(c1)

        x_iouf2 = self.W2_iouf(c1)
        xi, xo, xu, xf = torch.chunk(x_iouf2, 4, 1)
        h_iouf2 = self.U2_iouf(h2)
        hi2, ho2, hu2, hf2 = torch.chunk(h_iouf2, 4, 1)
        i = torch.sigmoid(xi + hi2)
        f = torch.sigmoid(xf + hf2)
        o = torch.sigmoid(xo + ho2)
        u = torch.tanh(xu + hu2)
        c2 = i * u + f * c2
        h2 = o * torch.tanh(c2)
        return {"h1": h1, "c1": c1, "h2": h2, "c2": c2}

    def apply_node_func(self, nodes):
        if self.init_state:
            iouf = nodes.data["iouf"]
            i, o, u, f = torch.chunk(iouf, 4, 1)
            i, o, u, f = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u), torch.sigmoid(f)
            c1 = i * u + f * nodes.data["c1"]
            h1 = o * torch.tanh(c1)

            iouf2 = self.W2_iouf(c1)
            i, o, u, f = torch.chunk(iouf2, 4, 1)
            i, o, u, f = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u), torch.sigmoid(f)
            c2 = i * u + f * nodes.data["c2"]
            h2 = o * torch.tanh(c2)
            self.init_state = False
            return {"h1": h1, "c1": c1, "h2": h2, "c2": c2}
        else:
            return {
                "h1": nodes.data["h1"],
                "c1": nodes.data["c1"],
                "h2": nodes.data["h2"],
                "c2": nodes.data["c2"],
            }


class TreeLSTMDouble(nn.Module):
    """TreeLSTM with double hidden state aggregation"""
    def __init__(self, x_size, h_size, num_classes, mode="sum", fc=True, bn=False, node_aggregation=None):
        super(TreeLSTMDouble, self).__init__()
        self.x_size, self.h_size = x_size, h_size
        self.node_aggregation = node_aggregation

        if bn:
            self.mlp1 = nn.Sequential(
                nn.Linear(x_size, h_size),
                nn.BatchNorm1d(h_size),
                nn.ReLU(),
                nn.Linear(h_size, 2 * h_size),
                nn.BatchNorm1d(2 * h_size),
                nn.ReLU(),
                nn.Linear(2 * h_size, h_size),
            )
        else:
            self.mlp1 = nn.Sequential(
                nn.Linear(x_size, h_size),
                nn.ReLU(),
                nn.Linear(h_size, 2 * h_size),
                nn.ReLU(),
                nn.Linear(2 * h_size, h_size),
            )

        self.cell = TreeLSTMDoubleCell(h_size, h_size, mode=mode)

        if node_aggregation:
            self.node_agg = MultiNodeAggregation(h_size, aggregation_type=node_aggregation)

        self.fc = fc
        if fc:
            self.linear = nn.Linear(h_size, num_classes)

    def forward_backbone(self, batch):
        g = batch.graph.to(torch.device("cuda"))
        # to heterogenous graph
        g = dgl.graph(g.edges())
        n = g.number_of_nodes()
        # feed embedding
        feats = self.mlp1(batch.feats.cuda())
        g.ndata["iouf"] = self.cell.W1_iouf(feats)
        g.ndata["h1"] = torch.zeros((n, self.h_size)).cuda()
        g.ndata["c1"] = torch.zeros((n, self.h_size)).cuda()
        g.ndata["h2"] = torch.zeros((n, self.h_size)).cuda()
        g.ndata["c2"] = torch.zeros((n, self.h_size)).cuda()
        # propagate
        dgl.prop_nodes_topo(
            g,
            message_func=self.cell.message_func,
            reduce_func=self.cell.reduce_func,
            apply_node_func=self.cell.apply_node_func,
        )
        logits = g.ndata.pop("c2")[batch.offset.long()]
        return logits

    def forward(self, batch):
        logits = self.forward_backbone(batch)
        if self.fc:
            logits = self.linear(logits)
            return logits
        else:
            return logits


class TreeLSTMv2(nn.Module):
    """TreeLSTM variant with improved architecture"""
    def __init__(self, x_size, h_size, num_classes, mode="sum", fc=True, bn=False):
        super(TreeLSTMv2, self).__init__()
        self.x_size, self.h_size = x_size, h_size

        if bn:
            self.mlp1 = nn.Sequential(
                nn.Linear(x_size, h_size),
                nn.BatchNorm1d(h_size),
                nn.ReLU(),
            )
        else:
            self.mlp1 = nn.Sequential(
                nn.Linear(x_size, h_size),
                nn.ReLU(),
            )

        self.cell = TreeLSTMCellv2(h_size, h_size, mode=mode)
        self.fc = fc
        if fc:
            self.linear = nn.Linear(h_size, num_classes)

    def forward(self, batch):
        g = batch.graph.to(torch.device("cuda"))
        g = dgl.graph(g.edges())
        n = g.number_of_nodes()

        feats = self.mlp1(batch.feats.cuda())
        g.ndata["iou"] = self.cell.W_iou(feats)
        g.ndata["h"] = torch.zeros((n, self.h_size)).cuda()
        g.ndata["c"] = torch.zeros((n, self.h_size)).cuda()

        dgl.prop_nodes_topo(
            g,
            message_func=self.cell.message_func,
            reduce_func=self.cell.reduce_func,
            apply_node_func=self.cell.apply_node_func,
        )

        h = g.ndata.pop("c")[batch.offset.long()]

        if self.fc:
            return self.linear(h)
        return h
