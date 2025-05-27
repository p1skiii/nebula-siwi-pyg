"""
极简图数据生成器
创建一个小规模的NBA图数据用于GNN处理
"""

import torch
from torch_geometric.data import Data
import pickle


def create_minimal_nba_graph():
    """创建极简NBA图数据"""
    
    # 定义节点：5个球员 + 3个球队
    nodes = {
        0: "Yao Ming",      # player133
        1: "LeBron James",  # player116  
        2: "Kobe Bryant",   # player115
        3: "Stephen Curry", # player117
        4: "Tim Duncan",    # player100
        5: "Lakers",        # team210
        6: "Warriors",      # team200
        7: "Rockets"        # team202
    }
    
    # 定义边：球员-球队关系（serve）和球员-球员关系（follow/teammate）
    edges = [
        # 球员服务球队关系
        (0, 7),  # Yao Ming -> Rockets
        (1, 5),  # LeBron -> Lakers
        (2, 5),  # Kobe -> Lakers
        (3, 6),  # Curry -> Warriors
        
        # 反向边（双向关系）
        (7, 0),  # Rockets -> Yao Ming
        (5, 1),  # Lakers -> LeBron
        (5, 2),  # Lakers -> Kobe
        (6, 3),  # Warriors -> Curry
        
        # 球员间关系（友谊/尊敬）
        (0, 1),  # Yao -> LeBron
        (1, 2),  # LeBron -> Kobe
        (2, 3),  # Kobe -> Curry
        (3, 4),  # Curry -> Duncan
        (4, 0),  # Duncan -> Yao (形成环)
        
        # 更多连接增加图的连通性
        (0, 4),  # Yao -> Duncan
        (1, 3),  # LeBron -> Curry
    ]
    
    # 转换为PyG格式
    edge_index = torch.tensor([[e[0] for e in edges], [e[1] for e in edges]], dtype=torch.long)
    
    # 生成节点特征（随机特征用于演示）
    num_nodes = len(nodes)
    node_features = torch.randn(num_nodes, 16)  # 16维特征
    
    # 创建PyG Data对象
    data = Data(x=node_features, edge_index=edge_index)
    
    # 添加节点映射信息
    data.node_map = nodes
    data.reverse_map = {name: idx for idx, name in nodes.items()}
    
    print(f"创建极简NBA图: {num_nodes}个节点, {len(edges)}条边")
    print(f"节点: {list(nodes.values())}")
    
    return data, nodes


def save_minimal_graph(filepath="/Users/wang/i/nebula-siwi/data/minimal_nba_graph.pt"):
    """保存极简图数据"""
    import os
    
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    data, nodes = create_minimal_nba_graph()
    
    # 保存数据
    torch.save({
        'data': data,
        'node_map': nodes,
        'reverse_map': data.reverse_map
    }, filepath)
    
    print(f"极简图数据已保存到: {filepath}")
    return filepath


if __name__ == "__main__":
    # 创建并保存极简图
    save_minimal_graph()
    
    # 测试加载
    data, nodes = create_minimal_nba_graph()
    print(f"\n图数据详情:")
    print(f"节点特征形状: {data.x.shape}")
    print(f"边索引形状: {data.edge_index.shape}")
    print(f"节点映射: {nodes}")
