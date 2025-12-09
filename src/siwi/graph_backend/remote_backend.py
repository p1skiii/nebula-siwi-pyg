import torch
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np


from torch_geometric.data import GraphStore, FeatureStore


from siwi.graph_backend.feature_store import (
    get_entity_embedding,
    get_nebula_connection_pool,
)
from siwi.graph_backend.subgraph_sampler import SubgraphSampler

class NebulaFeatureStore(FeatureStore):
    """连接NebulaGraph和PyG的特征存储类
    
    这个类实现了PyG的FeatureStore接口，使PyG能够从NebulaGraph获取节点特征
    """
    
    def __init__(self, space_name: str = "basketballplayer"):
        """初始化NebulaFeatureStore
        
        Args:
            space_name: NebulaGraph图空间名称
        """
        self.space_name = space_name
        self.connection_pool = get_nebula_connection_pool()
        # 存储临时张量数据的字典，用于实现写入功能
        self._tensor_cache = {}
        # 存储所有可用的张量属性
        self._tensor_attrs = {}
        # ID映射函数，默认为None，可在外部设置
        self.id_mapper = None
        
    def _get_tensor(self, group: str, name: str, index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """获取指定节点的特征（内部方法）
        
        Args:
            group: 节点类型，例如"player"、"team"
            name: 特征名称，例如"embedding1"
            index: 节点索引张量
            
        Returns:
            特征张量
        """
        print(f"[内部]获取{group}节点的{name}特征，索引大小: {index.size() if index is not None else 'None'}")
        
        # 检查是否存在于临时缓存中
        key = (group, name)
        if key in self._tensor_cache:
            tensor_data = self._tensor_cache[key]
            if index is None:
                return tensor_data
            return tensor_data[index]
        
        # 如果未提供索引，返回空张量
        if index is None:
            return torch.tensor([], dtype=torch.float)
        
        # 将索引转换为节点ID
        if self.id_mapper:
            # 使用提供的ID映射函数
            node_ids = [self.id_mapper(idx.item()) for idx in index]
        else:
            # 使用默认格式
            node_ids = [f"{group}{idx.item()}" for idx in index]
        
        # 获取每个节点的特征
        features = []
        for node_id in node_ids:
            # 调用功能1获取特征
            embedding = get_entity_embedding(node_id, group, name)
            
            # 如果找不到特征，使用零向量
            if embedding is None:
                features.append(torch.tensor([0.0], dtype=torch.float))
            else:
                # 将特征转换为张量
                features.append(torch.tensor([float(embedding)], dtype=torch.float))
        
        # 堆叠所有特征
        if features:
            return torch.cat(features, dim=0)
        else:
            return torch.zeros((len(node_ids), 1), dtype=torch.float)
            
    def _get_tensor_size(self, group: str, name: str) -> Tuple[int, ...]:
        """获取张量的大小
        
        Args:
            group: 节点类型
            name: 特征名称
            
        Returns:
            张量的大小（形状）
        """
        print(f"获取{group}节点的{name}特征大小")
        
        # 检查是否存在于临时缓存中
        key = (group, name)
        if key in self._tensor_cache:
            return self._tensor_cache[key].size()
        
        # 默认返回单特征向量大小
        # 在实际应用中，应该根据实际情况确定特征大小
        return (1,)
        
    def _put_tensor(self, group: str, name: str, tensor: torch.Tensor, index: Optional[torch.Tensor] = None) -> bool:
        """存储张量（内部方法）
        
        Args:
            group: 节点类型
            name: 特征名称
            tensor: 要存储的张量
            index: 节点索引张量
            
        Returns:
            是否成功存储
        """
        print(f"[内部]存储{group}节点的{name}特征，张量大小: {tensor.size()}")
        
        # 存储在临时缓存中（真实实现应将数据写入NebulaGraph）
        key = (group, name)
        
        # 如果该特征尚未注册，将其添加到属性列表中
        if group not in self._tensor_attrs:
            self._tensor_attrs[group] = set()
        self._tensor_attrs[group].add(name)
        
        if index is None:
            # 存储整个张量
            self._tensor_cache[key] = tensor
        else:
            # 创建或更新部分张量
            if key not in self._tensor_cache:
                # 如果还不存在该张量，先创建一个全零张量
                # 注意：这是一个简化的实现，可能需要更复杂的处理
                self._tensor_cache[key] = torch.zeros((max(index) + 1, tensor.size(1)), dtype=tensor.dtype)
            
            # 更新指定索引的值
            self._tensor_cache[key][index] = tensor
            
        return True
        
    def _remove_tensor(self, group: str, name: str) -> bool:
        """移除张量（内部方法）
        
        Args:
            group: 节点类型
            name: 特征名称
            
        Returns:
            是否成功移除
        """
        print(f"[内部]移除{group}节点的{name}特征")
        
        # 从临时缓存中移除
        key = (group, name)
        if key in self._tensor_cache:
            del self._tensor_cache[key]
            
            # 从属性列表中移除
            if group in self._tensor_attrs and name in self._tensor_attrs[group]:
                self._tensor_attrs[group].remove(name)
                
            return True
        
        return False
        
    def get_all_tensor_attrs(self) -> Dict[str, List[str]]:
        """获取所有可用的张量属性
        
        Returns:
            所有可用的张量属性，格式为{group: [attr1, attr2, ...]}
        """
        print("获取所有张量属性")
        
        # 将set转换为list返回
        return {group: list(attrs) for group, attrs in self._tensor_attrs.items()}
    
    def get_tensor(self, group: str, name: str, index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """获取指定节点的特征
        
        Args:
            group: 节点类型，例如"player"、"team"
            name: 特征名称，例如"embedding1"
            index: 节点索引张量
            
        Returns:
            特征张量
        """
        print(f"获取{group}节点的{name}特征，索引大小: {index.size() if index is not None else 'None'}")
        
        # 调用内部方法
        return self._get_tensor(group, name, index)
    
    def get_all(self, group: str, name: str) -> torch.Tensor:
        """获取所有节点的特征（不推荐用于大图）
        
        Args:
            group: 节点类型
            name: 特征名称
            
        Returns:
            特征张量
        """
        print(f"警告：尝试获取所有{group}节点的{name}特征，这在大图上可能很慢")
        # 在实际应用中，您应该避免获取所有节点的特征
        return torch.tensor([], dtype=torch.float)


# 实现GraphStore接口
class NebulaGraphStore(GraphStore):
    """连接NebulaGraph和PyG的图存储类
    
    这个类实现了PyG的GraphStore接口，使PyG能够从NebulaGraph获取图结构
    """
    
    def __init__(self, space_name: str = "basketballplayer"):
        """初始化NebulaGraphStore
        
        Args:
            space_name: NebulaGraph图空间名称
        """
        self.space_name = space_name
        self.connection_pool = get_nebula_connection_pool()
        # 使用您已实现的子图采样器
        self.sampler = SubgraphSampler(self.connection_pool)
        # 存储临时边数据的字典，用于实现写入功能
        self._edge_cache = {}
        # 存储所有可用的边属性
        self._edge_attrs = {}
        # ID映射函数，默认为None，可在外部设置
        self.id_mapper = None
    
    def get_edge_index(self, edge_type: Union[str, Tuple[str, str, str]], 
                      layout: str = "coo", 
                      size: Optional[Tuple[int, int]] = None,
                      index: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """获取边索引
        
        Args:
            edge_type: 边类型，例如"follow"或("player", "follow", "player")
            layout: 数据布局，"coo"或"csr"
            size: 图大小，(源节点数, 目标节点数)
            index: 可选的源节点和目标节点索引
            
        Returns:
            边索引张量，形状为[2, num_edges]
        """
        print(f"获取{edge_type}类型的边，布局: {layout}")
        
        # 直接调用内部方法
        return self._get_edge_index(edge_type, layout, size, index)
    
    def get_all_edge_index(self, edge_type: Union[str, Tuple[str, str, str]], 
                          layout: str = "coo",
                          size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """获取所有边（不推荐用于大图）
        
        Args:
            edge_type: 边类型
            layout: 数据布局
            size: 图大小
            
        Returns:
            边索引张量
        """
        print(f"警告：尝试获取所有{edge_type}类型的边，这在大图上可能很慢")
        return torch.zeros((2, 0), dtype=torch.long)
    
    def _get_edge_index(self, edge_type: Union[str, Tuple[str, str, str]], 
                       layout: str = "coo", 
                       size: Optional[Tuple[int, int]] = None,
                       index: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """获取边索引（内部方法）
        
        Args:
            edge_type: 边类型，例如"follow"或("player", "follow", "player")
            layout: 数据布局，"coo"或"csr"
            size: 图大小，(源节点数, 目标节点数)
            index: 可选的源节点和目标节点索引
            
        Returns:
            边索引张量，形状为[2, num_edges]
        """
        print(f"[内部]获取{edge_type}类型的边，布局: {layout}")
        
        # 检查是否存在于临时缓存中
        key = (edge_type, layout)
        if key in self._edge_cache:
            edge_data = self._edge_cache[key]
            if index is None:
                return edge_data
            # 如果提供了索引，过滤边
            src_indices, dst_indices = index
            mask = (edge_data[0].unsqueeze(1) == src_indices).any(dim=1) & (edge_data[1].unsqueeze(1) == dst_indices).any(dim=1)
            return edge_data[:, mask]
        
        # 只支持COO格式
        if layout != "coo":
            raise NotImplementedError(f"不支持{layout}布局，只支持coo")
        
        # 处理边类型
        if isinstance(edge_type, tuple) and len(edge_type) == 3:
            # 如果是(src_type, edge_name, dst_type)格式
            src_type, edge_name, dst_type = edge_type
        else:
            # 如果是字符串格式
            edge_name = edge_type
            src_type = dst_type = None
        
        # 如果提供了索引（源节点和目标节点）
        if index is not None:
            src_indices, dst_indices = index
            
            # 将索引转换为ID列表
            if self.id_mapper:
                # 使用提供的ID映射函数
                src_ids = [self.id_mapper(idx.item()) for idx in src_indices]
                dst_ids = [self.id_mapper(idx.item()) for idx in dst_indices]
            else:
                # 使用默认格式
                src_ids = [str(idx.item()) for idx in src_indices]
                dst_ids = [str(idx.item()) for idx in dst_indices]
            
            # 使用SubgraphSampler高效地收集边
            edges = []
            for src_idx, src_id in enumerate(src_ids):
                # 从该源节点获取一跳子图
                subgraph = self.sampler.sample_subgraph(
                    center_vid=src_id,
                    n_hops=1,
                    space_name=self.space_name
                )
                
                # 处理子图边
                if 'edge_index' in subgraph and subgraph['edge_index'] is not None:
                    edge_index = subgraph['edge_index']
                    num_edges = edge_index.shape[1]
                    
                    # 提取有用的连接
                    for i in range(num_edges):
                        # 获取边的源和目标索引
                        sub_src_idx = int(edge_index[0, i])
                        sub_dst_idx = int(edge_index[1, i])
                        
                        # 获取实际ID
                        sub_src_id = subgraph['idx_to_vid'][sub_src_idx]
                        sub_dst_id = subgraph['idx_to_vid'][sub_dst_idx]
                        
                        # 只保留符合条件的边：源节点是src_id，目标节点在dst_ids中
                        if sub_src_id == src_id and sub_dst_id in dst_ids:
                            # 将边加入结果，使用输入索引
                            dst_idx = dst_ids.index(sub_dst_id)
                            edges.append((src_idx, dst_idx))
            
            # 如果没有边，返回空张量
            if not edges:
                return torch.zeros((2, 0), dtype=torch.long)
            
            # 构建边索引张量
            edge_array = np.array(edges, dtype=np.int64).T
            return torch.from_numpy(edge_array)
            
        else:
            # 如果没有提供索引，返回空边集
            return torch.zeros((2, 0), dtype=torch.long)
    
    def _put_edge_index(self, edge_type: Union[str, Tuple[str, str, str]],
                      edge_index: torch.Tensor,
                      layout: str = "coo",
                      size: Optional[Tuple[int, int]] = None) -> bool:
        """存储边索引（内部方法）
        
        Args:
            edge_type: 边类型
            edge_index: 边索引张量
            layout: 数据布局
            size: 图大小
            
        Returns:
            是否成功存储
        """
        print(f"[内部]存储{edge_type}类型的边，布局: {layout}")
        
        # 只支持COO格式
        if layout != "coo":
            raise NotImplementedError(f"不支持{layout}布局，只支持coo")
        
        # 存储在临时缓存中（真实实现应将数据写入NebulaGraph）
        key = (edge_type, layout)
        self._edge_cache[key] = edge_index
        
        # 将边类型添加到属性列表中
        if isinstance(edge_type, tuple) and len(edge_type) == 3:
            src_type, edge_name, dst_type = edge_type
        else:
            edge_name = edge_type
            src_type = dst_type = None
        
        # 注册边类型
        self._edge_attrs[edge_name] = {'src_type': src_type, 'dst_type': dst_type}
        
        return True
    
    def _remove_edge_index(self, edge_type: Union[str, Tuple[str, str, str]],
                         layout: str = "coo") -> bool:
        """移除边索引（内部方法）
        
        Args:
            edge_type: 边类型
            layout: 数据布局
            
        Returns:
            是否成功移除
        """
        print(f"[内部]移除{edge_type}类型的边，布局: {layout}")
        
        # 从临时缓存中移除
        key = (edge_type, layout)
        if key in self._edge_cache:
            del self._edge_cache[key]
            
            # 从属性列表中移除
            if isinstance(edge_type, tuple) and len(edge_type) == 3:
                _, edge_name, _ = edge_type
            else:
                edge_name = edge_type
                
            if edge_name in self._edge_attrs:
                del self._edge_attrs[edge_name]
                
            return True
        
        return False
    
    def get_all_edge_attrs(self) -> List[Union[str, Tuple[str, str, str]]]:
        """获取所有可用的边属性
        
        Returns:
            所有可用的边类型列表
        """
        print("获取所有边属性")
        
        # 将边类型转换为(src_type, edge_name, dst_type)格式或字符串格式
        edge_types = []
        for edge_name, attrs in self._edge_attrs.items():
            src_type = attrs.get('src_type')
            dst_type = attrs.get('dst_type')
            
            if src_type and dst_type:
                edge_types.append((src_type, edge_name, dst_type))
            else:
                edge_types.append(edge_name)
        
        return edge_types
