#!/usr/bin/env python3
"""
MVP问答系统演示脚本
展示核心功能和技术栈集成
"""

import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from siwi.mvp_qa_system import MVPQuestionAnswering, SimpleEntityExtractor


def print_section_header(title):
    """打印节标题"""
    print(f"\n{'='*60}")
    print(f"🔹 {title}")
    print('='*60)


def demo_entity_extraction():
    """演示实体提取功能"""
    print_section_header("步骤1: 智能实体提取")
    
    extractor = SimpleEntityExtractor()
    
    test_cases = [
        "What relationships exist between Yao Ming and Lakers within 3 hops?",
        "How are Stephen Curry and Warriors connected?",
        "Find path from LeBron James to Kobe Bryant within 2 steps",
        "What is the relationship between Tim Duncan and Spurs?"
    ]
    
    for question in test_cases:
        entities = extractor.extract_entities(question)
        hops = extractor.parse_hops_from_text(question)
        
        print(f"\n📝 问题: {question}")
        print(f"   ✨ 提取实体: {entities}")
        print(f"   🔢 跳数解析: {hops}")


def demo_path_finding():
    """演示路径发现功能"""
    print_section_header("步骤2-5: 端到端问答演示")
    
    print("🚀 正在初始化MVP问答系统...")
    qa_system = MVPQuestionAnswering()
    print("✅ 系统初始化完成！")
    
    # 精选演示问题
    demo_questions = [
        {
            'question': "How are Stephen Curry and Warriors connected within 2 hops?",
            'description': "测试球员与球队的直接关系"
        },
        {
            'question': "What relationships exist between Tim Duncan and Spurs within 2 hops?",
            'description': "测试另一个球员-球队关系"
        },
        {
            'question': "Find path from LeBron James to Kevin Durant within 3 hops",
            'description': "测试球员间的多跳关系"
        }
    ]
    
    for i, demo in enumerate(demo_questions, 1):
        print(f"\n🎯 演示 {i}: {demo['description']}")
        print(f"📋 问题: {demo['question']}")
        print("-" * 50)
        
        start_time = time.time()
        answer = qa_system.answer_question(demo['question'])
        end_time = time.time()
        
        print(f"💡 答案: {answer}")
        print(f"⏱️ 处理时间: {end_time - start_time:.2f}秒")


def demo_technical_details():
    """演示技术细节"""
    print_section_header("技术栈演示")
    
    print("🏗️ 系统架构:")
    print("   1️⃣ NebulaGraph - 分布式图数据库")
    print("   2️⃣ PyTorch Geometric - 图神经网络框架")
    print("   3️⃣ 子图采样 - 高效图数据检索")
    print("   4️⃣ BFS算法 - 多跳路径发现")
    print("   5️⃣ GNN - 图嵌入学习")
    
    print("\n🔧 核心组件:")
    print("   • SimpleEntityExtractor - 实体识别和跳数解析")
    print("   • SubgraphSampler - NebulaGraph子图采样")
    print("   • PathFinder - BFS路径搜索算法")
    print("   • SimpleGNN - 图卷积神经网络")
    print("   • MVPQuestionAnswering - 端到端问答系统")
    
    print("\n📊 数据流:")
    print("   用户问题 → 实体提取 → 子图采样 → 路径发现 → GNN增强 → 结果输出")


def demo_supported_entities():
    """演示支持的实体"""
    print_section_header("支持的NBA实体")
    
    qa_system = MVPQuestionAnswering()
    extractor = qa_system.entity_extractor
    
    print("🏀 支持的球员:")
    players = list(extractor.player_map.keys())
    for i in range(0, len(players), 3):
        batch = players[i:i+3]
        print(f"   {' | '.join(batch)}")
    
    print("\n🏟️ 支持的球队:")
    teams = list(extractor.team_map.keys())
    for i in range(0, len(teams), 3):
        batch = teams[i:i+3]
        print(f"   {' | '.join(batch)}")


def demo_system_capabilities():
    """演示系统能力"""
    print_section_header("系统能力演示")
    
    print("✨ 核心功能:")
    print("   🔍 智能实体识别 - 从自然语言中提取NBA球员和球队")
    print("   🕸️ 多跳路径发现 - 发现实体间1-4跳的关系路径")
    print("   🧠 图神经网络 - 使用GNN学习图嵌入表示")
    print("   📊 子图采样 - 从大图中高效提取相关子图")
    print("   💬 自然语言回答 - 生成易读的路径描述")
    
    print("\n🎯 技术特点:")
    print("   • 实时图查询 - 直接从NebulaGraph查询，无需预加载")
    print("   • 可扩展架构 - 模块化设计，易于扩展新功能")
    print("   • 多策略搜索 - 支持直接子图和桥接节点策略")
    print("   • 错误容错 - 优雅处理查询失败和异常情况")
    print("   • 性能优化 - 限制搜索范围，避免大图遍历")


def main():
    """主演示函数"""
    print("🏀 NBA实体关系MVP问答系统")
    print("技术栈: NebulaGraph + PyTorch Geometric + GNN")
    print("目标: 演示图数据库与图神经网络的端到端集成")
    
    try:
        # 演示各个功能
        demo_entity_extraction()
        demo_supported_entities()
        demo_technical_details()
        demo_system_capabilities()
        demo_path_finding()
        
        print_section_header("演示完成")
        print("🎉 MVP系统演示完成！")
        print("✅ 所有核心功能正常工作")
        print("🚀 系统已准备好处理实际问答任务")
        
        print("\n🔗 下一步:")
        print("   • 运行 'python mvp_cli.py' 进入交互模式")
        print("   • 运行 'python src/siwi/mvp_web_api.py' 启动Web API")
        print("   • 扩展实体库以支持更多NBA球员和球队")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出错: {e}")
        print("请检查NebulaGraph连接和环境配置")


if __name__ == "__main__":
    main()
