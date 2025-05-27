#!/usr/bin/env python3
"""
测试MVP问答系统的功能
验证5个核心步骤是否正常工作
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from siwi.mvp_qa_system import MVPQuestionAnswering, SimpleEntityExtractor, PathFinder
from siwi.subgraph_sampler import SubgraphSampler
from siwi.feature_store import get_nebula_connection_pool


def test_entity_extraction():
    """测试步骤1: 实体提取"""
    print("\n=== 测试步骤1: 实体提取 ===")
    
    extractor = SimpleEntityExtractor()
    
    test_cases = [
        "What is the relationship between Yao Ming and Lakers?",
        "How are Stephen Curry and LeBron James connected?",
        "Find path from Kobe Bryant to Warriors",
        "姚明和湖人队的关系",
        "No entities here"
    ]
    
    for text in test_cases:
        entities = extractor.extract_entities(text)
        hops = extractor.parse_hops_from_text(text)
        print(f"文本: '{text}'")
        print(f"  实体: {entities}")
        print(f"  跳数: {hops}")
        print()


def test_subgraph_sampling():
    """测试步骤2: 子图采样"""
    print("\n=== 测试步骤2: 子图采样 ===")
    
    try:
        connection_pool = get_nebula_connection_pool()
        sampler = SubgraphSampler(connection_pool)
        
        # 测试从姚明开始的2跳子图
        subgraph = sampler.sample_subgraph(
            center_vid="player133",  # Yao Ming
            n_hops=2,
            max_nodes=100
        )
        
        print(f"姚明2跳子图:")
        print(f"  节点数: {subgraph.get('num_nodes', 0)}")
        print(f"  边数: {subgraph.get('edge_index').shape[1] if subgraph.get('edge_index') is not None else 0}")
        print(f"  中心节点索引: {subgraph.get('center_node_idx', 'N/A')}")
        
        # 显示部分节点
        idx_to_vid = subgraph.get('idx_to_vid', [])
        print(f"  前5个节点: {idx_to_vid[:5]}")
        
        return subgraph
        
    except Exception as e:
        print(f"子图采样测试失败: {e}")
        return None


def test_path_finding():
    """测试步骤3: 路径发现"""
    print("\n=== 测试步骤3: 路径发现 ===")
    
    try:
        connection_pool = get_nebula_connection_pool()
        sampler = SubgraphSampler(connection_pool)
        path_finder = PathFinder(sampler)
        
        # 测试姚明到湖人队的路径
        result = path_finder.find_multihop_relationships(
            "player133",  # Yao Ming
            "team210",    # Lakers
            max_hops=3
        )
        
        print(f"姚明到湖人队的路径查找:")
        print(f"  找到连接: {result['found']}")
        print(f"  路径数量: {len(result.get('paths', []))}")
        print(f"  查找方法: {result.get('method', 'unknown')}")
        
        # 显示路径
        for i, path in enumerate(result.get('paths', [])[:3]):
            print(f"  路径{i+1}: {' → '.join(path)}")
        
        return result
        
    except Exception as e:
        print(f"路径发现测试失败: {e}")
        return None


def test_gnn_processing():
    """测试步骤4: GNN处理"""
    print("\n=== 测试步骤4: GNN处理 ===")
    
    try:
        from siwi.mvp_qa_system import SimpleGNN
        import torch
        
        # 创建模拟数据
        gnn = SimpleGNN(input_dim=1, hidden_dim=8, output_dim=4)
        
        # 模拟图数据
        x = torch.randn(10, 1)  # 10个节点，1维特征
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        
        # 前向传播
        with torch.no_grad():
            embeddings = gnn(x, edge_index)
        
        print(f"GNN处理测试:")
        print(f"  输入节点数: {x.shape[0]}")
        print(f"  输入特征维度: {x.shape[1]}")
        print(f"  输出嵌入维度: {embeddings.shape[1]}")
        print(f"  边数: {edge_index.shape[1]}")
        
        return True
        
    except Exception as e:
        print(f"GNN处理测试失败: {e}")
        return False


def test_full_qa_system():
    """测试步骤5: 完整问答系统"""
    print("\n=== 测试步骤5: 完整问答系统 ===")
    
    try:
        qa_system = MVPQuestionAnswering()
        
        test_questions = [
            "What relationships exist between Yao Ming and Lakers within 3 hops?",
            "How are Stephen Curry and Warriors connected within 2 hops?",
            "Find path from LeBron James to Spurs"
        ]
        
        for question in test_questions:
            print(f"\n问题: {question}")
            print("-" * 50)
            answer = qa_system.answer_question(question)
            print(f"答案: {answer}")
            print()
        
        return True
        
    except Exception as e:
        print(f"完整问答系统测试失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("开始测试MVP问答系统...")
    
    # 测试各个步骤
    test_entity_extraction()
    test_subgraph_sampling()
    test_path_finding()
    test_gnn_processing()
    test_full_qa_system()
    
    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    main()
