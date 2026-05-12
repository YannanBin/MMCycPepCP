import torch
import torch.nn as nn
from transformers import AutoModel
from torch_geometric.nn import GATConv, global_mean_pool
import torch.nn.functional as F

class TCFModule(nn.Module):
    def __init__(self, feature_dim, morgan_dim=1024, num_heads=2, dropout=0.3):
        super(TCFModule, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads

        # Linear layer to map flattened Cartesian product to feature_dim
        self.linear_gcs = nn.Linear(feature_dim * feature_dim, feature_dim)

        # Linear layers and LayerNorm for gcs1 and s
        self.linear_gcs1 = nn.Linear(feature_dim, feature_dim)
        self.norm_gcs1 = nn.LayerNorm(feature_dim)
        self.linear_s = nn.Linear(feature_dim, feature_dim)
        self.norm_s = nn.LayerNorm(feature_dim)

        # Crossmodal Transformer components
        self.query_weight = nn.Linear(feature_dim, feature_dim)
        self.key_weight = nn.Linear(feature_dim, feature_dim)
        self.value_weight = nn.Linear(feature_dim, feature_dim)

        # Multi-head attention output projection
        self.output_projection = nn.Linear(feature_dim, feature_dim)

        # Feedforward network
        self.feedforward = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(dropout)
        )

        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(feature_dim)  # After first addition
        # self.layer_norm2 = nn.LayerNorm(feature_dim)  # After feedforward
        self.layer_norm3 = nn.LayerNorm(feature_dim)  # After second addition

        self.dropout = nn.Dropout(dropout)

        # 融合权重系数(可训练参数)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(1.0))

    def cartesian_product(self, s, g, m):
        # s: SMILES feature [batch_size, feature_dim]
        # g: GNN 3D structure feature [batch_size, feature_dim]
        # m: CNN Morgan fingerprint feature [batch_size, feature_dim]

        # Compute Cartesian products: s ⊗ g and s ⊗ m
        batch_size = s.size(0)
        s = s.unsqueeze(2)  # [batch_size, feature_dim, 1]
        g = g.unsqueeze(1)  # [batch_size, 1, feature_dim]
        m = m.unsqueeze(1)  # [batch_size, 1, feature_dim]

        sg = torch.bmm(s, g)  # [batch_size, feature_dim, feature_dim]
        sm = torch.bmm(s, m)  # [batch_size, feature_dim, feature_dim]

        # 添加g * m 融合
        # g_exp = g.transpose(1, 2)  # [batch_size, feature_dim, 1]
        # m_exp = m # [batch_size, 1, feature_dim]
        # gm = torch.bmm(g_exp, m_exp)  # [batch_size, feature_dim, feature_dim]

        # Average the Cartesian products
        gcs = (sg + sm) / 2  # [batch_size, feature_dim, feature_dim]
        # gcs = (sg + sm + gm) / 3  # [batch_size, feature_dim, feature_dim]
        # 使用可学习的融合权重系数
        # weights = self.alpha + self.beta + self.gamma + 1e-6 # 防止除零
        # gcs = (self.alpha * sg + self.beta * sm + self.gamma * gm) / weights  # [batch_size, feature_dim, feature_dim]
        # weights = self.alpha + self.beta + 1e-6 # 防止除零
        # gcs = (self.alpha * sg + self.beta * sm) / weights  # [batch_size, feature_dim, feature_dim]

        # Flatten and map to feature_dim
        gcs_flat = gcs.view(batch_size, -1)  # [batch_size, feature_dim * feature_dim]
        gcs1 = self.linear_gcs(gcs_flat)  # [batch_size, feature_dim]

        return gcs1

    def crossmodal_attention(self, s_norm, gcs1_norm):
        # s_norm: SMILES feature after linear and norm [batch_size, feature_dim]
        # gcs1_norm: fused feature after linear and norm [batch_size, feature_dim]

        # Compute Q, K, V
        Q_s = self.query_weight(s_norm)  # [batch_size, feature_dim]
        K_gcs = self.key_weight(gcs1_norm)  # [batch_size, feature_dim]
        V_gsc = self.value_weight(gcs1_norm)  # [batch_size, feature_dim]

        # Reshape for multi-head attention
        batch_size = s_norm.size(0)
        Q_s = Q_s.view(batch_size, self.num_heads, self.feature_dim // self.num_heads).transpose(1, 0)
        K_gcs = K_gcs.view(batch_size, self.num_heads, self.feature_dim // self.num_heads).transpose(1, 0)
        V_gsc = V_gsc.view(batch_size, self.num_heads, self.feature_dim // self.num_heads).transpose(1, 0)

        # Compute attention scores
        d_k = self.feature_dim // self.num_heads
        scores = torch.matmul(Q_s, K_gcs.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        attention_weights = F.softmax(scores, dim=-1)  # [num_heads, batch_size, 1]
        attention_output = torch.matmul(attention_weights, V_gsc)  # [num_heads, batch_size, feature_dim // num_heads]

        # Concatenate heads and project
        attention_output = attention_output.transpose(1, 0).contiguous().view(batch_size, self.feature_dim)
        attention_output = self.output_projection(attention_output)
        attention_output = self.dropout(attention_output)

        return attention_output

    def forward(self, s, g, m):
        # s: SMILES feature [batch_size, feature_dim]
        # g: GNN 3D structure feature [batch_size, feature_dim]
        # m: CNN Morgan fingerprint feature [batch_size, morgan_dim]

        # m = self.morgan_projection(m)  # [batch_size, feature_dim]
        # Step 1: Cartesian product fusion
        gcs1 = self.cartesian_product(s, g, m)

        # Step 2: Linear and LayerNorm for gcs1 and s
        gcs1_norm = self.norm_gcs1(self.linear_gcs1(gcs1))  # [batch_size, feature_dim]
        # gcs1_norm = self.norm_gcs1(gcs1)  # [batch_size, feature_dim]
        s_norm = self.norm_s(self.linear_s(s))  # [batch_size, feature_dim] (s')
        # s_norm = self.norm_s(s)  # [batch_size, feature_dim] (s')

        # Step 3: Crossmodal attention
        attention_output = self.crossmodal_attention(s_norm, gcs1_norm)

        # Step 4: First addition with s_norm (s') and LayerNorm
        output = self.layer_norm1(attention_output + s_norm)

        # Step 5: Feedforward
        ff_output = self.feedforward(output)

        # Step 6: Second addition with output (s' + attention_output) and LayerNorm
        # final_output = self.layer_norm3(output + ff_output)
        final_output = output + ff_output

        return final_output

class ConcatModule(nn.Module):
    def __init__(self, feature_dim, dropout=0.3):
        super(ConcatModule, self).__init__()
        self.feature_dim = feature_dim
        self.linear = nn.Sequential(
            nn.Linear(3 * feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, s, g, m):
        # s: SMILES feature
        # g: GNN 3D feature
        # m: Morgan fingerprint feature
        concat = torch.cat([s, g, m], dim=1)  # [batch_size, 3 * feature_dim]
        fused = self.linear(concat)           # [batch_size, feature_dim]
        return fused



class MultiModalPeptideModel(nn.Module):
    def __init__(self, atom_types, pre_model_path, transformer_layer, num_labels=14, gnn_hidden_dim=128, cnn_hidden_dim=128, tcf_layer=4, tcf_interval=None):
        super(MultiModalPeptideModel, self).__init__()
        self.premodel = AutoModel.from_pretrained(pre_model_path)
        self.transformer_layer = min(transformer_layer, self.premodel.config.num_hidden_layers)
        self.num_labels = num_labels
        self.tcf_layer = min(tcf_layer, self.transformer_layer)
        self.tcf_interval = tcf_interval

        input_dim = len(atom_types) + 3  # 原子种类 + 3维坐标
        self.gat_conv1 = GATConv(input_dim, gnn_hidden_dim, heads=4, concat=True)
        self.gat_conv2 = GATConv(gnn_hidden_dim * 4, gnn_hidden_dim, heads=4, concat=True)
        self.gat_conv3 = GATConv(gnn_hidden_dim * 4, gnn_hidden_dim, heads=1, concat=True)

        self.cnn = nn.Sequential(
            nn.Conv1d(1, cnn_hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(cnn_hidden_dim, cnn_hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        transformer_hidden_dim = self.premodel.config.hidden_size
        self.tcf_modules = nn.ModuleList()
        if tcf_interval is None:
            # self.tcf_modules.append(TCFModule(transformer_hidden_dim))
            self.tcf_modules.append(ConcatModule(transformer_hidden_dim))
        else:
            num_tcf_insertions = len(range(self.tcf_layer, self.transformer_layer, tcf_interval))
            for _ in range(num_tcf_insertions):
                # self.tcf_modules.append(TCFModule(transformer_hidden_dim))
                self.tcf_modules.append(ConcatModule(transformer_hidden_dim))

        self.gnn_projection = nn.Linear(gnn_hidden_dim, transformer_hidden_dim)
        self.cnn_projection = nn.Linear(cnn_hidden_dim * 256, transformer_hidden_dim)
        # self.cnn_projection = nn.Linear(cnn_hidden_dim, transformer_hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(transformer_hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels),
        )

    def forward(self, input_ids, graph_data, morgan_fp, attention_mask=None):
    # def forward(self, input_ids, morgan_fp, attention_mask=None):
        transformer_outputs = self.premodel(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = transformer_outputs.hidden_states

        # 处理 GNN 特征
        x, edge_index, batch = graph_data.x, graph_data.edge_index, graph_data.batch
        x = self.gat_conv1(x, edge_index)
        x = F.relu(x)
        x = self.gat_conv2(x, edge_index)
        x = F.relu(x)
        x = self.gat_conv3(x, edge_index)
        gnn_output = global_mean_pool(x, batch)
        gnn_output = self.gnn_projection(gnn_output)

        # morgan_output = morgan_fp
        morgan_fp = morgan_fp.unsqueeze(1)  # 添加维度以适应 CNN 输入
        morgan_output = self.cnn(morgan_fp)
        morgan_output = morgan_output.view(morgan_output.size(0), -1)  # 展平 CNN 输出
        morgan_output = self.cnn_projection(morgan_output)

        tcf_idx = 0  # 初始化TCF模块索引计数器为0，用于跟踪当前使用的TCF模块
        premodel_output = hidden_states[0][:, 0, :]  # 提取第一层隐藏状态的第一个token的特征向量，作为初始输出
        # 从第二层开始遍历所有隐藏层
        for layer_idx in range(1, len(hidden_states)):
            layer_output = hidden_states[layer_idx][:, 0, :]
            if self.tcf_interval is None:
                if layer_idx == self.tcf_layer and tcf_idx < len(self.tcf_modules):
                    layer_output = self.tcf_modules[tcf_idx](layer_output, gnn_output, morgan_output)
                    tcf_idx += 1
            else:
                if layer_idx >= self.tcf_layer and (layer_idx - self.tcf_layer) % self.tcf_interval == 0 and tcf_idx < len(self.tcf_modules):
                    layer_output = self.tcf_modules[tcf_idx](layer_output, gnn_output, morgan_output)
                    tcf_idx += 1
            premodel_output = layer_output

        logits = self.classifier(premodel_output)
        return logits