"""
MVP问答系统 - NBA实体关系路径发现
实现5步核心功能：
1. 简单实体提取
2. 图数据连接和子图检索
3. 核心路径发现算法
4. 基础GNN集成
5. 文本结果输出
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Set
import re
from collections import deque
import numpy as np

# PyG imports
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# Siwi imports
from siwi.subgraph_sampler import SubgraphSampler
from siwi.feature_store import get_nebula_connection_pool
from siwi.pyg_integration import NebulaToTorch


class SimpleEntityExtractor:
    """步骤1: 简单实体提取器"""
    
    def __init__(self):
        # NBA球员和球队的映射 (从现有的YAML数据中提取)
        self.player_map = {
            "Tim Duncan": "player100",
            "Tony Parker": "player101", 
            "LeBron James": "player116",
            "Stephen Curry": "player117",
            "Kobe Bryant": "player115",
            "Yao Ming": "player133",
            "Klay Thompson": "player142",
            "Kevin Durant": "player119",
            "James Harden": "player120",
            "Chris Paul": "player121",
            "Russell Westbrook": "player118"
        }
        
        self.team_map = {
            "Warriors": "team200",
            "Lakers": "team210", 
            "Rockets": "team202",
            "Spurs": "team204",
            "Celtics": "team217",
            "Heat": "team229",
            "Cavaliers": "team216"
        }
        
        # 合并所有实体映射
        self.entity_map = {**self.player_map, **self.team_map}
        
        # 反向映射 (ID -> 名称)
        self.id_to_name = {v: k for k, v in self.entity_map.items()}
    
    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """从文本中提取实体 
        返回: [(实体名称, 实体ID), ...]
        """
        entities = []
        text_lower = text.lower()
        
        # 简单字符串匹配
        for entity_name, entity_id in self.entity_map.items():
            if entity_name.lower() in text_lower:
                entities.append((entity_name, entity_id))
        
        return entities
    
    def parse_hops_from_text(self, text: str) -> int:
        """从文本中提取跳数"""
        # 查找数字，默认为2跳
        hop_patterns = [
            r'(\d+)\s*hop', 
            r'(\d+)\s*step',
            r'within\s+(\d+)',
            r'(\d+)\s*度'
        ]
        
        for pattern in hop_patterns:
            match = re.search(pattern, text.lower())
            if match:
                return min(int(match.group(1)), 4)  # 最大4跳
        
        return 2  # 默认2跳


class PathFinder:
    """步骤3: 核心路径发现算法"""
    
    def __init__(self, subgraph_sampler: SubgraphSampler):
        self.sampler = subgraph_sampler
    
    def find_paths_bfs(self, subgraph_data: Dict, start_entity: str, 
                      end_entity: str, max_hops: int = 4) -> List[List[str]]:
        """使用BFS算法查找两个实体间的路径"""
        if start_entity not in subgraph_data.get('vid_to_idx', {}):
            return []
        if end_entity not in subgraph_data.get('vid_to_idx', {}):
            return []
        
        start_idx = subgraph_data['vid_to_idx'][start_entity]
        end_idx = subgraph_data['vid_to_idx'][end_entity]
        
        # BFS队列：(当前节点索引, 路径)
        queue = deque([(start_idx, [start_entity])])
        visited = set([start_idx])
        paths = []
        
        edge_index = subgraph_data.get('edge_index')
        if edge_index is None:
            return []
        
        # 构建邻接表
        adj_list = {}
        num_edges = edge_index.shape[1] if hasattr(edge_index, 'shape') else 0
        
        for i in range(num_edges):
            src_idx = int(edge_index[0, i])
            dst_idx = int(edge_index[1, i])
            
            if src_idx not in adj_list:
                adj_list[src_idx] = []
            adj_list[src_idx].append(dst_idx)
        
        # BFS搜索
        while queue and len(paths) < 5:  # 最多找5条路径
            current_idx, path = queue.popleft()
            
            if len(path) > max_hops + 1:  # 路径长度超限
                continue
                
            if current_idx == end_idx:
                paths.append(path.copy())
                continue
            
            # 扩展邻居
            for neighbor_idx in adj_list.get(current_idx, []):
                if neighbor_idx not in visited or len(path) <= 2:  # 允许短路径重访
                    neighbor_vid = subgraph_data['idx_to_vid'][neighbor_idx]
                    new_path = path + [neighbor_vid]
                    queue.append((neighbor_idx, new_path))
                    
                    if len(path) > 2:  # 长路径标记访问
                        visited.add(neighbor_idx)
        
        return paths
    
    def find_multihop_relationships(self, entity_a: str, entity_b: str, 
                                   max_hops: int = 3) -> Dict:
        """发现两个实体间的多跳关系"""
        # 从实体A出发采样子图
        subgraph_a = self.sampler.sample_subgraph(
            center_vid=entity_a,
            n_hops=max_hops,
            max_nodes=500
        )
        
        # 如果实体B在子图中，直接查找路径
        if entity_b in subgraph_a.get('vid_to_idx', {}):
            paths = self.find_paths_bfs(subgraph_a, entity_a, entity_b, max_hops)
            return {
                'found': True,
                'paths': paths,
                'subgraph': subgraph_a,
                'method': 'direct_subgraph'
            }
        
        # 否则从实体B也采样子图寻找交集
        subgraph_b = self.sampler.sample_subgraph(
            center_vid=entity_b,
            n_hops=max_hops//2,
            max_nodes=500
        )
        
        # 查找交集节点
        nodes_a = set(subgraph_a.get('vid_to_idx', {}).keys())
        nodes_b = set(subgraph_b.get('vid_to_idx', {}).keys())
        intersection = nodes_a & nodes_b
        
        if intersection:
            # 通过交集节点构建路径
            bridge_paths = []
            for bridge_node in list(intersection)[:3]:  # 最多检查3个桥接节点
                paths_a_to_bridge = self.find_paths_bfs(subgraph_a, entity_a, bridge_node, max_hops//2)
                paths_bridge_to_b = self.find_paths_bfs(subgraph_b, bridge_node, entity_b, max_hops//2)
                
                for path_a in paths_a_to_bridge[:2]:
                    for path_b in paths_bridge_to_b[:2]:
                        # 连接路径（去除重复的桥接节点）
                        combined_path = path_a + path_b[1:]
                        bridge_paths.append(combined_path)
            
            return {
                'found': True,
                'paths': bridge_paths[:5],  # 最多5条路径
                'subgraph': subgraph_a,  # 返回主子图
                'method': 'bridge_nodes',
                'bridge_nodes': list(intersection)
            }
        
        return {
            'found': False,
            'paths': [],
            'subgraph': subgraph_a,
            'method': 'no_connection'
        }


class SimpleGNN(torch.nn.Module):
    """步骤4: 简单的图神经网络"""
    
    def __init__(self, input_dim: int = 1, hidden_dim: int = 16, output_dim: int = 8):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(0.5)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # 第一层GCN
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 第二层GCN
        x = self.conv2(x, edge_index)
        return x
    
    def get_node_embeddings(self, data: Data) -> torch.Tensor:
        """获取节点嵌入"""
        return self.forward(data.x, data.edge_index)


class MVPQuestionAnswering:
    """MVP问答系统主类 - 整合所有5个步骤"""
    
    def __init__(self, space_name: str = "basketballplayer"):
        self.space_name = space_name
        
        # 初始化各组件
        self.entity_extractor = SimpleEntityExtractor()
        
        # 初始化NebulaGraph连接
        self.connection_pool = get_nebula_connection_pool()
        self.subgraph_sampler = SubgraphSampler(self.connection_pool)
        
        # 初始化路径发现器
        self.path_finder = PathFinder(self.subgraph_sampler)
        
        # 初始化PyG集成和GNN
        self.nebula_to_torch = NebulaToTorch(space_name)
        self.gnn_model = SimpleGNN()
        
        print("MVP问答系统初始化完成")
    
    def answer_question(self, question: str) -> str:
        """核心问答方法 - 整合5个步骤"""
        print(f"\n=== 处理问题: {question} ===")
        
        # 步骤1: 实体提取
        entities = self.entity_extractor.extract_entities(question)
        if len(entities) < 2:
            return self._handle_insufficient_entities(question, entities)
        
        entity_a_name, entity_a_id = entities[0]
        entity_b_name, entity_b_id = entities[1]
        
        print(f"步骤1 - 提取实体: {entity_a_name} ({entity_a_id}) 和 {entity_b_name} ({entity_b_id})")
        
        # 步骤2: 解析跳数
        max_hops = self.entity_extractor.parse_hops_from_text(question)
        print(f"步骤2 - 解析跳数: {max_hops}")
        
        # 步骤3: 路径发现
        print("步骤3 - 开始路径发现...")
        path_result = self.path_finder.find_multihop_relationships(
            entity_a_id, entity_b_id, max_hops
        )
        
        if not path_result['found']:
            return f"在{max_hops}跳范围内，{entity_a_name}和{entity_b_name}之间没有找到连接路径。"
        
        paths = path_result['paths']
        print(f"找到 {len(paths)} 条路径")
        
        # 步骤4: GNN增强 (简单演示)
        try:
            subgraph = path_result['subgraph']
            pyg_data = self.subgraph_sampler.convert_to_pyg_data(subgraph)
            
            if pyg_data is not None and hasattr(pyg_data, 'x') and hasattr(pyg_data, 'edge_index'):
                # 获取GNN嵌入
                with torch.no_grad():
                    node_embeddings = self.gnn_model.get_node_embeddings(pyg_data)
                print(f"步骤4 - GNN处理: 生成了 {node_embeddings.shape[0]} 个节点的嵌入")
            else:
                print("步骤4 - GNN处理: 子图数据不完整，跳过GNN处理")
        except Exception as e:
            print(f"步骤4 - GNN处理出错: {e}")
        
        # 步骤5: 生成文本结果
        answer = self._format_answer(entity_a_name, entity_b_name, paths, max_hops, path_result)
        print("步骤5 - 生成答案完成")
        
        return answer
    
    def _handle_insufficient_entities(self, question: str, entities: List) -> str:
        """处理实体提取不足的情况"""
        if len(entities) == 0:
            return ("抱歉，我无法从问题中识别出NBA球员或球队。\n"
                   "请尝试问题如：'Yao Ming和Lakers之间有什么关系？'")
        elif len(entities) == 1:
            entity_name, entity_id = entities[0]
            return (f"我识别出了{entity_name}，但需要两个实体来查找关系。\n"
                   f"请指定另一个NBA球员或球队。")
    
    def _format_answer(self, entity_a: str, entity_b: str, paths: List[List[str]], 
                      max_hops: int, path_result: Dict) -> str:
        """步骤5: 格式化最终答案"""
        if not paths:
            return f"在{max_hops}跳内没有找到{entity_a}和{entity_b}之间的连接。"
        
        answer_lines = [
            f"在{max_hops}跳范围内，{entity_a}和{entity_b}之间找到了{len(paths)}条连接路径：\n"
        ]
        
        for i, path in enumerate(paths[:3], 1):  # 最多显示3条路径
            # 转换路径中的ID为名称
            path_names = []
            for entity_id in path:
                entity_name = self.entity_extractor.id_to_name.get(entity_id, entity_id)
                path_names.append(entity_name)
            
            path_str = " → ".join(path_names)
            answer_lines.append(f"路径{i}: {path_str} ({len(path)-1}跳)")
        
        # 添加方法说明
        method = path_result.get('method', 'unknown')
        if method == 'direct_subgraph':
            answer_lines.append(f"\n✓ 通过直接子图搜索发现连接")
        elif method == 'bridge_nodes':
            bridge_count = len(path_result.get('bridge_nodes', []))
            answer_lines.append(f"\n✓ 通过{bridge_count}个桥接节点发现连接")
        
        return "\n".join(answer_lines)


# 便捷函数
def create_mvp_system() -> MVPQuestionAnswering:
    """创建MVP问答系统实例"""
    return MVPQuestionAnswering()


def ask_question(question: str, qa_system: MVPQuestionAnswering = None) -> str:
    """便捷问答函数"""
    if qa_system is None:
        qa_system = create_mvp_system()
    
    return qa_system.answer_question(question)


if __name__ == "__main__":
    # 测试用例
    qa_system = create_mvp_system()
    
    test_questions = [
        "What relationships exist between Yao Ming and Lakers within 3 hops?",
        "How are Stephen Curry and LeBron James connected within 2 hops?",
        "Find path from Kobe Bryant to Warriors in 2 steps",
        "姚明和湖人队有什么关系？"
    ]
    
    for question in test_questions:
        print(f"\n{'='*50}")
        print(f"问题: {question}")
        print(f"{'='*50}")
        answer = qa_system.answer_question(question)
        print(f"答案: {answer}")
