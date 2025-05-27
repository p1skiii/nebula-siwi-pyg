from siwi.bot.actions import SiwiActions
from siwi.bot.bert_classifier import create_bert_classifier
# from siwi.bot.classifier import SiwiClassifier  # 注释掉旧分类器


class SiwiBot():
    def __init__(self, connection_pool) -> None:
        print("[INFO] 初始化SiwiBot (BERT + GNN集成版)")
        
        # 使用新的BERT分类器
        self.classifier = create_bert_classifier()
        self.actions = SiwiActions()
        self.connection_pool = connection_pool
        
        print("[INFO] SiwiBot初始化完成")

    def query(self, sentence):
        """处理用户查询 - 支持BERT NLU + GNN"""
        print(f"[INFO] SiwiBot处理查询: {sentence}")
        
        # 步骤1: BERT NLU处理
        intent = self.classifier.get(sentence)
        print(f"[DEBUG] NLU结果: {intent}")
        
        # 步骤2: 获取对应的Action
        action = self.actions.get(intent)
        print(f"[DEBUG] 选择的Action: {type(action).__name__}")
        
        # 步骤3: 执行Action
        result = action.execute(self.connection_pool)
        print(f"[DEBUG] Action执行完成")
        
        return result
