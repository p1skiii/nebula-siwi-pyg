#!/usr/bin/env python3
"""
MVP问答系统的命令行接口
提供交互式的问答体验
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from siwi.mvp_qa_system import MVPQuestionAnswering


def print_banner():
    """打印欢迎横幅"""
    print("=" * 60)
    print("🏀 NBA实体关系MVP问答系统 🏀")
    print("=" * 60)
    print("功能: 发现NBA球员和球队之间的多跳关系路径")
    print("技术栈: NebulaGraph + PyTorch Geometric + GNN")
    print()
    print("支持的问题类型:")
    print("• What relationships exist between [Entity A] and [Entity B] within N hops?")
    print("• How are [Entity A] and [Entity B] connected?")
    print("• Find path from [Entity A] to [Entity B]")
    print()
    print("支持的实体:")
    print("球员: Yao Ming, LeBron James, Stephen Curry, Kobe Bryant, Tim Duncan等")
    print("球队: Lakers, Warriors, Spurs, Celtics, Heat等")
    print()
    print("输入 'help' 查看更多帮助，输入 'quit' 退出")
    print("=" * 60)


def print_help():
    """打印帮助信息"""
    print("\n📖 帮助信息:")
    print()
    print("🔸 示例问题:")
    print("  • What relationships exist between Yao Ming and Lakers within 3 hops?")
    print("  • How are Stephen Curry and Warriors connected within 2 hops?")
    print("  • Find path from LeBron James to Kobe Bryant")
    print("  • What is the relationship between Tim Duncan and Spurs?")
    print()
    print("🔸 支持的跳数: 1-4跳 (默认2跳)")
    print("🔸 支持的球员: Yao Ming, LeBron James, Stephen Curry, Kobe Bryant, 等")
    print("🔸 支持的球队: Lakers, Warriors, Spurs, Celtics, Heat, 等")
    print()
    print("🔸 命令:")
    print("  • help - 显示此帮助信息")
    print("  • examples - 显示示例问题")
    print("  • entities - 显示所有支持的实体")
    print("  • quit/exit - 退出程序")
    print()


def print_examples():
    """打印示例问题"""
    examples = [
        "What relationships exist between Yao Ming and Lakers within 3 hops?",
        "How are Stephen Curry and Warriors connected within 2 hops?",
        "Find path from LeBron James to Kobe Bryant",
        "What is the relationship between Tim Duncan and Spurs?",
        "How are Kevin Durant and Warriors connected?",
        "Find path from Chris Paul to Lakers within 3 hops"
    ]
    
    print("\n📝 示例问题:")
    for i, example in enumerate(examples, 1):
        print(f"  {i}. {example}")
    print()


def print_entities(qa_system):
    """打印支持的实体"""
    print("\n👥 支持的实体:")
    print()
    
    print("🏀 球员:")
    for name, entity_id in qa_system.entity_extractor.player_map.items():
        print(f"  • {name} ({entity_id})")
    
    print("\n🏟️ 球队:")
    for name, entity_id in qa_system.entity_extractor.team_map.items():
        print(f"  • {name} ({entity_id})")
    print()


def main():
    """主函数"""
    print_banner()
    
    # 初始化MVP问答系统
    print("🔄 正在初始化系统...")
    try:
        qa_system = MVPQuestionAnswering()
        print("✅ 系统初始化完成！")
        print()
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        print("请检查NebulaGraph连接和环境配置")
        return
    
    # 交互循环
    while True:
        try:
            # 获取用户输入
            question = input("🤖 请输入您的问题: ").strip()
            
            # 处理特殊命令
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 再见！")
                break
            elif question.lower() == 'help':
                print_help()
                continue
            elif question.lower() == 'examples':
                print_examples()
                continue
            elif question.lower() == 'entities':
                print_entities(qa_system)
                continue
            elif not question:
                print("❓ 请输入一个问题或命令")
                continue
            
            # 处理问题
            print(f"\n🔍 分析问题: {question}")
            print("⏳ 正在处理...")
            
            try:
                answer = qa_system.answer_question(question)
                print(f"\n💡 答案:")
                print(f"{answer}")
                print()
                
            except Exception as e:
                print(f"\n❌ 处理问题时出错: {e}")
                print("请尝试重新表述问题或检查实体名称是否正确")
                print()
                
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，再见！")
            break
        except EOFError:
            print("\n\n👋 再见！")
            break


if __name__ == "__main__":
    main()
