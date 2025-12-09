"""
PyG与NebulaGraph集成模块
提供了从NebulaGraph加载数据到PyG的工具
"""

import torch
from typing import List, Dict, Any, Optional, Tuple, Union

from siwi.graph_backend.remote_backend import NebulaFeatureStore, NebulaGraphStore
from siwi.graph_backend.neighbor_loader import SimpleNeighborLoader

class NebulaToTorch:
    """NebulaGraph到PyTorch的转换器
    
    这个类是功能3的主要接口，提供了将NebulaGraph数据转换为PyG格式的方法
    """
    
    def __init__(self, space_name: str = "basketballplayer"):
        """初始化转换器
        
        Args:
            space_name: NebulaGraph图空间名称
        """
        # ID映射：字符串ID到索引的映射和反向映射
        self.id_to_idx = {}
        self.idx_to_id = []
        
        # 初始化特征存储
        self.feature_store = NebulaFeatureStore(space_name)
        # 设置ID映射函数
        self.feature_store.id_mapper = self.get_node_id_by_idx
        
        # 初始化图存储
        self.graph_store = NebulaGraphStore(space_name)
        # 设置ID映射函数
        self.graph_store.id_mapper = self.get_node_id_by_idx
        
        # 初始化加载器
        self.loader = SimpleNeighborLoader(
            feature_store=self.feature_store,
            graph_store=self.graph_store
        )
        
        print(f"NebulaToTorch初始化完成，连接到图空间: {space_name}")
    
    def get_node_id_by_idx(self, idx: int) -> str:
        """根据索引获取节点ID
        
        Args:
            idx: 节点索引
            
        Returns:
            节点ID字符串
        """
        if 0 <= idx < len(self.idx_to_id):
            return self.idx_to_id[idx]
        return f"unknown{idx}"
    
    def _get_or_add_id(self, node_id: str) -> int:
        """获取或添加节点ID映射
        
        Args:
            node_id: 节点ID字符串
            
        Returns:
            节点索引
        """
        if node_id not in self.id_to_idx:
            idx = len(self.idx_to_id)
            self.id_to_idx[node_id] = idx
            self.idx_to_id.append(node_id)
        return self.id_to_idx[node_id]
    
    def get_node_features(self, node_ids: List[str], node_type: str = "player") -> torch.Tensor:
        """获取节点特征
        
        Args:
            node_ids: 节点ID列表
            node_type: 节点类型
            
        Returns:
            特征张量
        """
        # 创建索引映射
        indices = []
        for node_id in node_ids:
            idx = self._get_or_add_id(node_id)
            indices.append(idx)
        
        # 转换为张量
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        
        # 获取特征
        features = self.feature_store.get_tensor(
            group=node_type,
            name="embedding1",
            index=indices_tensor
        )
        
        return features
        
        return features
    
    def get_subgraph(self, center_nodes: List[str], n_hops: int = 1) -> Dict:
        """获取子图
        
        Args:
            center_nodes: 中心节点ID列表
            n_hops: 跳数
            
        Returns:
            子图数据字典
        """
        # 创建索引映射
        indices = []
        for node_id in center_nodes:
            idx = self._get_or_add_id(node_id)
            indices.append(idx)
        
        # 使用加载器获取数据
        data = self.loader.load_data(center_nodes, indices, n_hops)
        
        # 更新全局ID映射，确保一致性
        if hasattr(data, 'node_ids') and data.node_ids:
            for idx, node_id in enumerate(data.node_ids):
                if node_id not in self.id_to_idx:
                    new_idx = len(self.idx_to_id)
                    self.id_to_idx[node_id] = new_idx
                    self.idx_to_id.append(node_id)
        
        # 转换为字典格式
        result = {
            "num_nodes": data.num_nodes,
            "features": data.x.tolist() if hasattr(data, 'x') else [],
            "edge_index": data.edge_index.tolist() if hasattr(data, 'edge_index') and data.edge_index.numel() > 0 else [],
            "node_ids": data.node_ids if hasattr(data, 'node_ids') else []
        }
        
        return result
