#!/usr/bin/env python3
"""
端到端测试脚本
测试BERT NLU + GNN + Siwi完整流程
"""

import sys
import os
sys.path.append('/Users/wang/i/nebula-siwi/src')

from siwi.bot.bot import SiwiBot
from siwi.feature_store import get_nebula_connection_pool


def test_bert_classifier():
    """测试BERT分类器"""
    print("=== 测试BERT分类器 ===")
    
    from siwi.bot.bert_classifier import create_bert_classifier
    
    classifier = create_bert_classifier()
    
    test_sentences = [
        "What is the relationship between Yao Ming and Lakers?",
        "Who is similar to LeBron James?",
        "Which team did Kobe Bryant serve?", 
        "Who does Stephen Curry follow?",
        "Find someone like Yao Ming",
        "Hello world"
    ]
    
    for sentence in test_sentences:
        print(f"\n输入: {sentence}")
        result = classifier.get(sentence)
        print(f"输出: {result}")


def test_gnn_processor():
    """测试GNN处理器"""
    print("\n=== 测试GNN处理器 ===")
    
    from siwi.bot.gnn_processor import create_gnn_processor
    
    try:
        processor = create_gnn_processor()
        
        test_nodes = ["Yao Ming", "LeBron James", "Lakers", "Warriors"]
        
        for node in test_nodes:
            print(f"\n查找与 {node} 相似的节点:")
            similar = processor.get_similar(node, top_k=2)
            print(f"结果: {similar}")
        
        # 测试图统计
        stats = processor.get_graph_stats()
        print(f"\n图统计: {stats}")
        
    except Exception as e:
        print(f"GNN测试失败: {e}")


def test_end_to_end():
    """测试端到端流程"""
    print("\n=== 端到端测试 ===")
    
    try:
        # 初始化连接
        print("初始化NebulaGraph连接...")
        connection_pool = get_nebula_connection_pool()
        
        # 初始化Bot
        print("初始化SiwiBot...")
        bot = SiwiBot(connection_pool)
        
        # 测试查询
        test_queries = [
            "Who is similar to Yao Ming?",                           # GNN查询
            "What is the relationship between Yao Ming and Lakers?",  # 增强关系查询  
            "Which team did LeBron James serve?",                    # 传统服务查询
            "Find someone like Stephen Curry",                       # GNN相似度
            "How are Kobe Bryant and Lakers connected?",             # 关系查询
            "Random question"                                        # Fallback测试
        ]
        
        for query in test_queries:
            print(f"\n{'='*50}")
            print(f"查询: {query}")
            print(f"{'='*50}")
            
            try:
                answer = bot.query(query)
                print(f"回答: {answer}")
            except Exception as e:
                print(f"查询失败: {e}")
        
        print(f"\n{'='*50}")
        print("端到端测试完成！")
        
    except Exception as e:
        print(f"端到端测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_actions_integration():
    """测试Actions集成"""
    print("\n=== 测试Actions集成 ===")
    
    from siwi.bot.actions import SiwiActions, GNNAction
    
    actions = SiwiActions()
    
    # 测试GNN Action
    test_intent = {
        "entities": {"Yao Ming": "player"},
        "intents": ("find_similar",)
    }
    
    print(f"测试意图: {test_intent}")
    action = actions.get(test_intent)
    print(f"选择的Action: {type(action).__name__}")
    
    if isinstance(action, GNNAction):
        print("GNN Action选择成功！")
    else:
        print("GNN Action选择失败！")


def main():
    """主测试函数"""
    print("🚀 开始BERT + GNN + Siwi集成测试")
    print("="*60)
    
    # 分步测试
    try:
        test_bert_classifier()
    except Exception as e:
        print(f"BERT测试失败: {e}")
    
    try:
        test_gnn_processor()
    except Exception as e:
        print(f"GNN测试失败: {e}")
    
    try:
        test_actions_integration()
    except Exception as e:
        print(f"Actions测试失败: {e}")
    
    # 端到端测试
    try:
        test_end_to_end()
    except Exception as e:
        print(f"端到端测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 所有测试完成！")


if __name__ == "__main__":
    main()
