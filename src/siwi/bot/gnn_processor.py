"""
GNN处理器
基于PyTorch Geometric实现图神经网络相似度计算
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
import os
from typing import List, Dict, Tuple, Optional


class SimpleGCN(torch.nn.Module):
    """简单的图卷积网络"""
    
    def __init__(self, in_channels: int = 16, hidden_channels: int = 32, out_channels: int = 16):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(0.3)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # 第一层GCN
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 第二层GCN
        x = self.conv2(x, edge_index)
        
        return x


class SimpleGAT(torch.nn.Module):
    """简单的图注意力网络（备选）"""
    
    def __init__(self, in_channels: int = 16, hidden_channels: int = 32, out_channels: int = 16, heads: int = 2):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels // heads, heads=heads, dropout=0.3)
        self.conv2 = GATConv(hidden_channels, out_channels, heads=1, dropout=0.3)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        
        return x


class GNNProcessor:
    """GNN处理器主类"""
    
    def __init__(self, graph_path: str = "/Users/wang/i/nebula-siwi/data/minimal_nba_graph.pt", 
                 model_type: str = "gcn"):
        """
        初始化GNN处理器
        
        Args:
            graph_path: 图数据文件路径
            model_type: 模型类型 ("gcn" 或 "gat")
        """
        self.graph_path = graph_path
        self.model_type = model_type
        
        # 加载图数据
        self._load_graph_data()
        
        # 初始化模型
        self._init_model()
        
        # 预计算嵌入
        self._precompute_embeddings()
        
        print(f"[INFO] GNN处理器初始化完成 (模型: {model_type})")
    
    def _load_graph_data(self):
        """加载图数据"""
        try:
            if not os.path.exists(self.graph_path):
                raise FileNotFoundError(f"图数据文件不存在: {self.graph_path}")
            
            saved_data = torch.load(self.graph_path, map_location='cpu')
            self.data = saved_data['data']
            self.node_map = saved_data['node_map']  # {idx: name}
            self.reverse_map = saved_data['reverse_map']  # {name: idx}
            
            print(f"[INFO] 已加载图数据: {len(self.node_map)}个节点, {self.data.edge_index.shape[1]}条边")
            
        except Exception as e:
            print(f"[ERROR] 加载图数据失败: {e}")
            # 创建备用的极简图
            self._create_fallback_graph()
    
    def _create_fallback_graph(self):
        """创建备用图数据"""
        print("[INFO] 创建备用图数据...")
        
        # 极简图：4个节点
        self.node_map = {
            0: "Yao Ming",
            1: "LeBron James", 
            2: "Lakers",
            3: "Rockets"
        }
        self.reverse_map = {name: idx for idx, name in self.node_map.items()}
        
        # 简单连接
        edge_index = torch.tensor([[0, 1, 0, 2], [2, 2, 3, 3]], dtype=torch.long)
        x = torch.randn(4, 16)
        
        self.data = Data(x=x, edge_index=edge_index)
    
    def _init_model(self):
        """初始化GNN模型"""
        input_dim = self.data.x.shape[1]
        
        if self.model_type == "gat":
            self.model = SimpleGAT(input_dim, 32, 16)
        else:
            self.model = SimpleGCN(input_dim, 32, 16)
        
        self.model.eval()  # 设置为评估模式
    
    def _precompute_embeddings(self):
        """预计算节点嵌入"""
        try:
            with torch.no_grad():
                self.embeddings = self.model(self.data)
            print(f"[INFO] 预计算嵌入完成: {self.embeddings.shape}")
        except Exception as e:
            print(f"[ERROR] 嵌入计算失败: {e}")
            # 使用原始特征作为备用
            self.embeddings = self.data.x
    
    def get_similar(self, node_name: str, top_k: int = 3, exclude_self: bool = True) -> List[str]:
        """
        查找与指定节点最相似的节点
        
        Args:
            node_name: 节点名称
            top_k: 返回top-k个相似节点
            exclude_self: 是否排除自身
            
        Returns:
            相似节点名称列表
        """
        # 查找节点索引
        if node_name not in self.reverse_map:
            return [f"节点 '{node_name}' 不存在于图中"]
        
        node_idx = self.reverse_map[node_name]
        target_emb = self.embeddings[node_idx]
        
        # 计算余弦相似度
        similarities = F.cosine_similarity(target_emb.unsqueeze(0), self.embeddings)
        
        # 排除自身
        if exclude_self:
            similarities[node_idx] = -1
        
        # 获取top-k
        top_indices = torch.topk(similarities, k=min(top_k, len(self.node_map)-1)).indices
        
        # 转换为名称
        similar_nodes = [self.node_map[idx.item()] for idx in top_indices]
        
        return similar_nodes
    
    def get_node_embedding(self, node_name: str) -> Optional[torch.Tensor]:
        """获取指定节点的嵌入向量"""
        if node_name not in self.reverse_map:
            return None
        
        node_idx = self.reverse_map[node_name]
        return self.embeddings[node_idx]
    
    def compute_similarity(self, node1: str, node2: str) -> float:
        """计算两个节点间的相似度"""
        if node1 not in self.reverse_map or node2 not in self.reverse_map:
            return 0.0
        
        idx1 = self.reverse_map[node1]
        idx2 = self.reverse_map[node2]
        
        emb1 = self.embeddings[idx1]
        emb2 = self.embeddings[idx2]
        
        similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
        return similarity.item()
    
    def get_all_nodes(self) -> List[str]:
        """获取所有节点名称"""
        return list(self.reverse_map.keys())
    
    def get_graph_stats(self) -> Dict:
        """获取图统计信息"""
        num_nodes = len(self.node_map)
        num_edges = self.data.edge_index.shape[1]
        
        return {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "node_feature_dim": self.data.x.shape[1],
            "embedding_dim": self.embeddings.shape[1],
            "model_type": self.model_type
        }


class GNNProcessorLite:
    """轻量版GNN处理器：如果完整版有问题时的备选方案"""
    
    def __init__(self):
        print("[INFO] 初始化轻量版GNN处理器...")
        
        # 硬编码的相似度映射
        self.similarity_map = {
            "Yao Ming": ["Tim Duncan", "LeBron James"],
            "LeBron James": ["Kobe Bryant", "Stephen Curry"], 
            "Kobe Bryant": ["LeBron James", "Tim Duncan"],
            "Stephen Curry": ["Klay Thompson", "Kevin Durant"],
            "Lakers": ["Warriors", "Celtics"],
            "Warriors": ["Lakers", "Spurs"],
            "Rockets": ["Lakers", "Spurs"]
        }
    
    def get_similar(self, node_name: str, top_k: int = 3, exclude_self: bool = True) -> List[str]:
        """查找相似节点（基于预定义映射）"""
        return self.similarity_map.get(node_name, [f"暂无与 {node_name} 相似的节点"])[:top_k]
    
    def get_all_nodes(self) -> List[str]:
        return list(self.similarity_map.keys())
    
    def get_graph_stats(self) -> Dict:
        return {"num_nodes": len(self.similarity_map), "model_type": "lite"}


def create_gnn_processor(use_lite: bool = False) -> GNNProcessor:
    """工厂函数：创建GNN处理器实例"""
    if use_lite:
        return GNNProcessorLite()
    
    try:
        return GNNProcessor()
    except Exception as e:
        print(f"[WARN] 完整GNN处理器初始化失败，使用轻量版: {e}")
        return GNNProcessorLite()


if __name__ == "__main__":
    # 测试GNN处理器
    print("=== 测试GNN处理器 ===")
    
    processor = create_gnn_processor()
    
    # 测试相似度查询
    test_nodes = ["Yao Ming", "LeBron James", "Lakers"]
    
    for node in test_nodes:
        print(f"\n与 {node} 最相似的节点:")
        similar = processor.get_similar(node, top_k=2)
        for i, sim_node in enumerate(similar, 1):
            print(f"  {i}. {sim_node}")
    
    # 测试图统计
    print(f"\n图统计信息: {processor.get_graph_stats()}")
