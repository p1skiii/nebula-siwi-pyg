import torch
from typing import List, Dict, Any, Optional, Tuple, Union

from torch_geometric.data import Data

from siwi.graph_backend.remote_backend import NebulaFeatureStore, NebulaGraphStore

class SimpleNeighborLoader:
    """简化版的邻居加载器
    
    这个类演示如何使用NebulaFeatureStore和NebulaGraphStore加载数据
    """
    
    def __init__(self, feature_store: NebulaFeatureStore, 
                graph_store: NebulaGraphStore,
                node_type: str = "player",
                edge_type: str = "follow"):
        """初始化加载器
        
        Args:
            feature_store: 特征存储
            graph_store: 图存储
            node_type: 节点类型
            edge_type: 边类型
        """
        self.feature_store = feature_store
        self.graph_store = graph_store
        self.node_type = node_type
        self.edge_type = edge_type
    
    def load_data(self, seed_nodes: List[str], node_indices: List[int], num_hops: int = 1) -> Data:
        """加载以种子节点为中心的子图数据
        
        Args:
            seed_nodes: 种子节点ID列表
            node_indices: 节点索引列表（对应于seed_nodes）
            num_hops: 跳数
            
        Returns:
            PyG Data对象
        """
        print(f"为{len(seed_nodes)}个种子节点加载{num_hops}跳邻居")
        
        if len(seed_nodes) == 0:
            # 如果没有种子节点，返回空数据
            return Data(
                x=torch.zeros((0, 1), dtype=torch.float),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                num_nodes=0,
                node_ids=[]
            )
        
        # 1. 使用第一个种子节点获取子图
        # 注意：这是一个简化的实现，实际应用中可能需要合并多个子图
        center_vid = seed_nodes[0]
        
        # 直接利用GraphStore的sampler获取子图
        # 这样确保了ID映射的一致性
        subgraph = self.graph_store.sampler.sample_subgraph(
            center_vid=center_vid,
            n_hops=num_hops,
            space_name=self.graph_store.space_name
        )
        
        # 2. 获取节点特征
        # 创建节点索引张量，使用子图的索引
        num_nodes = subgraph['num_nodes']
        node_indices_tensor = torch.arange(num_nodes, dtype=torch.long)
        
        # 获取节点特征
        node_features = torch.zeros((num_nodes, 1), dtype=torch.float)
        
        # 尝试从特征存储获取特征
        try:
            features = self.feature_store.get_tensor(
                group=self.node_type,
                name="embedding1",
                index=node_indices_tensor
            )
            if features.size(0) == num_nodes:
                node_features = features
        except Exception as e:
            print(f"无法获取节点特征: {e}")
            # 使用子图中的特征
            for idx, vid in enumerate(subgraph['idx_to_vid']):
                if vid in subgraph['node_features'] and 'embedding' in subgraph['node_features'][vid]:
                    node_features[idx] = subgraph['node_features'][vid]['embedding']
        
        # 3. 创建PyG Data对象
        data = Data(
            x=node_features,
            edge_index=subgraph['edge_index'],
            num_nodes=num_nodes
        )
        
        # 添加节点ID映射以便于查询
        data.node_ids = subgraph['idx_to_vid']
        
        return data
