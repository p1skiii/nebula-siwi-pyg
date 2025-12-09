import torch
import numpy as np
from typing import Dict, List, Tuple, Set, Union, Optional
from nebula3.gclient.net import ConnectionPool

from siwi.graph_backend.feature_store import get_nebula_connection_pool

class SubgraphSampler:
    """从NebulaGraph中提取子图并转换为PyG可用的格式"""
    
    def __init__(self, connection_pool: Optional[ConnectionPool] = None):
        """初始化子图采样器
        
        Args:
            connection_pool: NebulaGraph连接池，如果为None则使用默认连接池
        """
        self.connection_pool = connection_pool or get_nebula_connection_pool()
        # 缓存从VID到连续整数ID的映射
        self._vid_to_idx_map = {}
        # 缓存从连续整数ID到VID的映射
        self._idx_to_vid_map = []
        # 存储不同类型的边
        self._edge_indices = {}
        # 节点类型信息
        self._node_types = {}
    
    def sample_subgraph(self, 
                        center_vid: str, 
                        n_hops: int = 1, 
                        space_name: str = "basketballplayer",
                        use_bidirectional: bool = True,
                        max_nodes: int = 1000) -> Dict:
        """从指定节点出发，采样n_hops跳的子图
        
        Args:
            center_vid: 中心节点的VID (如 "player142" 表示姚明)
            n_hops: 采样的跳数，默认为1
            space_name: NebulaGraph图空间名称
            use_bidirectional: 是否生成双向边 (PyG通常期望无向图格式)
            max_nodes: 最大节点数限制，防止子图过大
            
        Returns:
            包含子图信息的字典，可以直接用于构建PyG的Data对象
        """
        # 重置状态
        self._vid_to_idx_map = {}
        self._idx_to_vid_map = []
        self._edge_indices = {}
        self._node_types = {}
        
        # 获取会话
        session = self.connection_pool.get_session("root", "nebula")
        try:
            # 使用指定的图空间
            session.execute(f"USE {space_name}")
            
            # 1. 获取子图数据
            if n_hops <= 2:
                # 对于小跳数使用GO语句
                subgraph_data = self._get_subgraph_using_go(session, center_vid, n_hops, max_nodes)
            else:
                # 对于更大跳数使用GET SUBGRAPH
                subgraph_data = self._get_subgraph_using_subgraph(session, center_vid, n_hops, max_nodes)
            
            # 2. 生成PyG格式的edge_index
            edge_index = self._create_edge_index(subgraph_data['edges'], use_bidirectional)
            
            # 3. 获取相关节点的属性
            node_features = self._get_node_features(session, subgraph_data['nodes'])
            
            # 4. 构建结果字典
            result = {
                'center_node_idx': self._vid_to_idx_map.get(center_vid, 0),
                'edge_index': edge_index,
                'num_nodes': len(self._idx_to_vid_map),
                'vid_to_idx': self._vid_to_idx_map.copy(),
                'idx_to_vid': self._idx_to_vid_map.copy(),
                'node_types': self._node_types,
                'edge_indices_by_type': self._edge_indices,
                'node_features': node_features
            }
            
            return result
            
        finally:
            session.release()
    
    def _get_subgraph_using_go(self, session, center_vid: str, n_hops: int, max_nodes: int) -> Dict:
        """使用GO语句获取子图
        
        适合1-2跳的小规模子图
        """
        nodes = set([center_vid])
        edges = []
        
        # 获取中心节点的类型
        type_query = f'MATCH (v) WHERE id(v) == "{center_vid}" RETURN labels(v) as types'
        resp = session.execute(type_query)
        if resp.is_succeeded() and resp.row_size() > 0:
            node_types = resp.row_values(0)[0].as_list()
            if node_types and not node_types[0].is_empty():
                self._node_types[center_vid] = node_types[0].as_string()
        
        # 对每一跳进行查询
        for hop in range(1, n_hops + 1):
            # 获取所有外向边
            out_query = f'''
            GO {hop} STEPS FROM "{center_vid}" OVER * 
            YIELD DISTINCT id($^) as src, id($$) as dst, type(edge) as edge_type
            '''
            resp = session.execute(out_query)
            
            if resp.is_succeeded():
                for i in range(resp.row_size()):
                    row = resp.row_values(i)
                    src = row[0].as_string()
                    dst = row[1].as_string()
                    edge_type = row[2].as_string()
                    
                    nodes.add(src)
                    nodes.add(dst)
                    edges.append((src, dst, edge_type))
                    
                    # 获取目标节点的类型
                    if dst not in self._node_types:
                        type_query = f'MATCH (v) WHERE id(v) == "{dst}" RETURN labels(v) as types'
                        type_resp = session.execute(type_query)
                        if type_resp.is_succeeded() and type_resp.row_size() > 0:
                            node_types = type_resp.row_values(0)[0].as_list()
                            if node_types and not node_types[0].is_empty():
                                self._node_types[dst] = node_types[0].as_string()
                    
                    # 检查是否超过节点数限制
                    if len(nodes) >= max_nodes:
                        break
                
                # 如果已经达到节点数上限，则提前结束采样
                if len(nodes) >= max_nodes:
                    break
        
        return {
            'nodes': list(nodes),
            'edges': edges
        }
    
    def _get_subgraph_using_subgraph(self, session, center_vid: str, n_hops: int, max_nodes: int) -> Dict:
        """使用GET SUBGRAPH语句获取子图
        
        适合更大规模的子图
        """
        # GET SUBGRAPH语句
        query = f'''
        GET SUBGRAPH {n_hops} STEPS FROM "{center_vid}" YIELD VERTICES AS nodes, EDGES AS relationships
        '''
        
        nodes = set([center_vid])
        edges = []
        
        resp = session.execute(query)
        if resp.is_succeeded():
            # 解析结果比较复杂，需要根据NebulaGraph的返回格式进行处理
            # 此示例假设返回了一个可以按行遍历的结果集
            for i in range(resp.row_size()):
                row_data = resp.row_values(i)
                
                # 处理节点
                # 注意: 这部分代码需要根据实际的返回格式调整
                if row_data[0].is_vertex():
                    vertex = row_data[0]
                    vid = vertex.get_id().as_string()
                    nodes.add(vid)
                    
                    # 获取节点类型
                    node_type = vertex.tags()[0]  # 假设使用第一个tag作为节点类型
                    self._node_types[vid] = node_type
                
                # 处理边
                if len(row_data) > 1 and row_data[1].is_edge():
                    edge = row_data[1]
                    src = edge.get_src().as_string()
                    dst = edge.get_dst().as_string()
                    edge_type = edge.name()  # 边类型
                    
                    nodes.add(src)
                    nodes.add(dst)
                    edges.append((src, dst, edge_type))
                
                # 检查是否超过节点数限制
                if len(nodes) >= max_nodes:
                    break
                    
        return {
            'nodes': list(nodes),
            'edges': edges
        }
    
    def _get_vid_idx(self, vid: str) -> int:
        """将VID映射为连续整数索引
        
        如果VID之前未见过，则分配一个新的索引
        """
        if vid not in self._vid_to_idx_map:
            idx = len(self._idx_to_vid_map)
            self._vid_to_idx_map[vid] = idx
            self._idx_to_vid_map.append(vid)
        return self._vid_to_idx_map[vid]
    
    def _create_edge_index(self, 
                          edges: List[Tuple[str, str, str]], 
                          bidirectional: bool = True) -> torch.Tensor:
        """将边列表转换为PyG格式的edge_index
        
        Args:
            edges: 边列表，每个元素为 (src_vid, dst_vid, edge_type)
            bidirectional: 是否添加反向边
            
        Returns:
            形状为[2, num_edges]的edge_index张量
        """
        if not edges:
            return torch.zeros((2, 0), dtype=torch.long)
        
        # 按边类型分组
        edge_indices_by_type = {}
        
        # 处理所有边
        for src_vid, dst_vid, edge_type in edges:
            src_idx = self._get_vid_idx(src_vid)
            dst_idx = self._get_vid_idx(dst_vid)
            
            # 按边类型存储边
            if edge_type not in edge_indices_by_type:
                edge_indices_by_type[edge_type] = []
            
            edge_indices_by_type[edge_type].append((src_idx, dst_idx))
            
            # 如果需要双向边，添加反向边
            if bidirectional:
                edge_indices_by_type[edge_type].append((dst_idx, src_idx))
        
        # 创建每种类型的edge_index
        for edge_type, edge_list in edge_indices_by_type.items():
            edge_array = np.array(edge_list, dtype=np.int64).T
            self._edge_indices[edge_type] = torch.from_numpy(edge_array)
        
        # 合并所有边类型创建总的edge_index
        all_edges = []
        for edge_list in edge_indices_by_type.values():
            all_edges.extend(edge_list)
        
        if not all_edges:
            return torch.zeros((2, 0), dtype=torch.long)
        
        # 转换为PyG期望的COO格式
        edge_array = np.array(all_edges, dtype=np.int64).T
        return torch.from_numpy(edge_array)
    
        # 确保正确缩进这个方法，使它成为类的一部分
    def _get_node_features(self, session, node_vids: List[str]) -> Dict:
        """获取节点的特征
        
        尝试获取节点的embedding1特征
        """
        features = {}
        
        # 对于每个节点类型分别查询
        node_vids_by_type = {}
        for vid in node_vids:
            node_type = self._node_types.get(vid, 'unknown')
            if node_type not in node_vids_by_type:
                node_vids_by_type[node_type] = []
            node_vids_by_type[node_type].append(vid)
        
        # 按类型批量查询节点特征
        for node_type, vids in node_vids_by_type.items():
            if node_type == 'unknown':
                continue
                
            # 分批查询以避免查询过大
            batch_size = 100
            for i in range(0, len(vids), batch_size):
                batch_vids = vids[i:i+batch_size]
                vid_str = '", "'.join(batch_vids)
                
                query = f'''
                FETCH PROP ON {node_type} "{vid_str}" 
                YIELD id(vertex) AS id, properties(vertex).name AS name, 
                      properties(vertex).embedding1 AS embedding
                '''
                
                resp = session.execute(query)
                if resp.is_succeeded():
                    for j in range(resp.row_size()):
                        row = resp.row_values(j)
                        vid = row[0].as_string()
                        
                        # 处理name字段
                        name = ""
                        if not row[1].is_empty() and row[1].is_string():
                            name = row[1].as_string()
                        
                        # 初始化基本特征
                        features[vid] = {'name': name}
                        
                        # 处理embedding字段 - 修复这里的错误处理
                        if not row[2].is_empty():
                            try:
                                if row[2].is_double():
                                    embedding = row[2].as_double()
                                    features[vid]['embedding'] = torch.tensor([float(embedding)], dtype=torch.float)
                                elif row[2].is_int():
                                    # 如果是整数，转换为浮点数
                                    embedding = float(row[2].as_int())
                                    features[vid]['embedding'] = torch.tensor([embedding], dtype=torch.float)
                                # 可以添加其他类型的处理...
                            except Exception as e:
                                print(f"无法转换节点 {vid} 的embedding1值: {e}")
        
        return features
    
    def convert_to_pyg_data(self, subgraph: Dict) -> object:
        """将子图转换为PyG的Data对象
        
        需要安装PyG: pip install torch_geometric
        """
        try:
            from torch_geometric.data import Data
            
            # 创建节点特征矩阵，维度为 [num_nodes, 1]
            # 每个节点的特征是其embedding1值
            x = torch.zeros((subgraph['num_nodes'], 1), dtype=torch.float)
            for idx, vid in enumerate(subgraph['idx_to_vid']):
                if vid in subgraph['node_features'] and 'embedding' in subgraph['node_features'][vid]:
                    x[idx] = subgraph['node_features'][vid]['embedding']
            
            # 节点类型
            node_type = torch.zeros(subgraph['num_nodes'], dtype=torch.long)
            for idx, vid in enumerate(subgraph['idx_to_vid']):
                node_type[idx] = 0  # 默认类型
                if vid in self._node_types:
                    if self._node_types[vid] == 'player':
                        node_type[idx] = 0
                    elif self._node_types[vid] == 'team':
                        node_type[idx] = 1
            
            # 创建PyG数据对象
            data = Data(
                x=x,
                edge_index=subgraph['edge_index'],
                node_type=node_type,
                # 可添加其他属性
                center_node_idx=subgraph['center_node_idx']
            )
            
            # 存储VID映射，便于后续查询
            data.vid_to_idx = subgraph['vid_to_idx']
            data.idx_to_vid = subgraph['idx_to_vid']
            
            return data
            
        except ImportError:
            print("PyTorch Geometric未安装，无法创建Data对象")
            return None
