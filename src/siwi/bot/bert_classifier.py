"""
BERT-based NLU Classifier
使用Hugging Face预训练模型进行实体识别和意图分类
"""

import re
from typing import Dict, List, Tuple
from transformers import pipeline
import logging

# 设置日志
logging.getLogger("transformers").setLevel(logging.WARNING)


class BERTClassifier:
    """BERT分类器：替代原有的SiwiClassifier"""
    
    def __init__(self):
        print("[INFO] 初始化BERT分类器...")
        
        # 使用轻量模型避免下载大文件
        self.use_lite_mode = True
        
        if not self.use_lite_mode:
            # NER模型 - 使用更轻量的模型
            try:
                print("[INFO] 加载轻量NER模型...")
                self.ner_pipeline = pipeline(
                    "ner", 
                    model="distilbert-base-uncased",  # 更小的模型
                    grouped_entities=True,
                    device=-1
                )
            except Exception as e:
                print(f"[WARN] NER模型加载失败: {e}")
                self.ner_pipeline = None
            
            # 意图识别 - 使用更轻量的模型
            try:
                print("[INFO] 加载轻量意图分类模型...")
                self.intent_pipeline = pipeline(
                    "zero-shot-classification",
                    model="distilbert-base-uncased",  # 更小的模型
                    device=-1
                )
            except Exception as e:
                print(f"[WARN] 意图模型加载失败: {e}")
                self.intent_pipeline = None
        else:
            print("[INFO] 使用轻量模式，跳过模型下载")
            self.ner_pipeline = None
            self.intent_pipeline = None
        
        # 定义支持的意图标签（对应intents.yaml）
        self.candidate_labels = [
            'relationship',  # 关系查询
            'serve',        # 服务历史
            'friend',       # 朋友/关注关系  
            'find_similar', # GNN相似度查询（新增）
            'fallback'      # 兜底
        ]
        
        # NBA实体映射（从YAML文件中提取的核心实体）
        self.nba_entities = {
            # 球员
            "Yao Ming": "player133",
            "Tim Duncan": "player100", 
            "Tony Parker": "player101",
            "LeBron James": "player116",
            "Stephen Curry": "player117",
            "Kobe Bryant": "player115",
            "Klay Thompson": "player142",
            "Kevin Durant": "player119",
            "James Harden": "player120",
            "Chris Paul": "player121",
            "Russell Westbrook": "player118",
            
            # 球队
            "Warriors": "team200",
            "Lakers": "team210",
            "Rockets": "team202", 
            "Spurs": "team204",
            "Celtics": "team217",
            "Heat": "team229",
            "Cavaliers": "team216"
        }
        
        print("[INFO] BERT分类器初始化完成")
    
    def get(self, sentence: str) -> Dict:
        """
        主要方法：从句子中提取意图和实体
        返回格式与SiwiClassifier兼容：{"entities": {...}, "intents": (...)}
        """
        print(f"[DEBUG] BERT分类器处理: {sentence}")
        
        entities_result = {}
        intents_result = []
        
        # 步骤1: 实体识别
        entities_result = self._extract_entities(sentence)
        
        # 步骤2: 意图识别
        intent = self._classify_intent(sentence, entities_result)
        intents_result.append(intent)
        
        result = {
            "entities": entities_result,
            "intents": tuple(intents_result)
        }
        
        print(f"[DEBUG] BERT分类结果: {result}")
        return result
    
    def _extract_entities(self, sentence: str) -> Dict:
        """实体提取：结合BERT NER和NBA实体库"""
        entities = {}
        
        # 方法1: 直接字符串匹配NBA实体（更可靠）
        sentence_lower = sentence.lower()
        for entity_name, entity_id in self.nba_entities.items():
            if entity_name.lower() in sentence_lower:
                entities[entity_name] = self._get_entity_type(entity_id)
        
        # 方法2: BERT NER补充（仅在非轻量模式下）
        if not self.use_lite_mode and self.ner_pipeline is not None:
            try:
                ner_output = self.ner_pipeline(sentence)
                for entity in ner_output:
                    entity_text = entity['word'].replace('##', '')  # 处理子词
                    if entity_text not in entities and len(entity_text) > 2:
                        # 简单映射BERT的实体类型
                        bert_type = entity['entity_group'].lower()
                        if bert_type in ['per', 'person']:
                            entities[entity_text] = 'player'
                        elif bert_type in ['org', 'organization']:
                            entities[entity_text] = 'team'
            except Exception as e:
                print(f"[WARN] BERT NER处理失败: {e}")
        
        return entities
    
    def _get_entity_type(self, entity_id: str) -> str:
        """根据实体ID判断类型"""
        if entity_id.startswith('player'):
            return 'player'
        elif entity_id.startswith('team'):
            return 'team'
        else:
            return 'unknown'
    
    def _classify_intent(self, sentence: str, entities: Dict) -> str:
        """意图分类：结合规则和BERT"""
        
        # 规则1: 基于关键词的快速判断
        sentence_lower = sentence.lower()
        
        # GNN相似度查询
        if any(word in sentence_lower for word in ['similar', 'like', 'comparable', 'resemble']):
            return 'find_similar'
        
        # 关系查询
        if any(word in sentence_lower for word in ['relationship', 'relation', 'connect', 'between']):
            return 'relationship'
        
        # 服务历史
        if any(word in sentence_lower for word in ['serve', 'served', 'team', 'play for']):
            return 'serve'
        
        # 朋友/关注
        if any(word in sentence_lower for word in ['follow', 'friend', 'follows']):
            return 'friend'
        
        # 规则2: BERT Zero-Shot分类
        try:
            intent_output = self.intent_pipeline(sentence, self.candidate_labels)
            top_intent = intent_output['labels'][0]
            confidence = intent_output['scores'][0]
            
            print(f"[DEBUG] BERT意图分类: {top_intent} (置信度: {confidence:.3f})")
            
            # 如果置信度太低，回退到fallback
            if confidence < 0.3:
                return 'fallback'
            
            return top_intent
            
        except Exception as e:
            print(f"[WARN] BERT意图分类失败: {e}")
            return 'fallback'


class BERTClassifierLite:
    """轻量版BERT分类器：如果完整版有问题时的备选方案"""
    
    def __init__(self):
        print("[INFO] 初始化轻量版BERT分类器...")
        self.nba_entities = {
            "Yao Ming": "player133",
            "LeBron James": "player116", 
            "Stephen Curry": "player117",
            "Kobe Bryant": "player115",
            "Warriors": "team200",
            "Lakers": "team210",
            "Rockets": "team202"
        }
    
    def get(self, sentence: str) -> Dict:
        """简化版分类器"""
        entities = {}
        
        # 简单实体匹配
        sentence_lower = sentence.lower()
        for entity_name, entity_id in self.nba_entities.items():
            if entity_name.lower() in sentence_lower:
                entities[entity_name] = 'player' if entity_id.startswith('player') else 'team'
        
        # 简单意图分类
        if 'similar' in sentence_lower:
            intent = 'find_similar'
        elif any(word in sentence_lower for word in ['relationship', 'connect', 'between']):
            intent = 'relationship'
        elif 'serve' in sentence_lower:
            intent = 'serve'
        elif 'follow' in sentence_lower:
            intent = 'friend'
        else:
            intent = 'fallback'
        
        return {"entities": entities, "intents": (intent,)}


# 工厂函数：根据环境选择合适的分类器
def create_bert_classifier(use_optimized: bool = True):
    """创建BERT分类器实例"""
    
    if use_optimized:
        try:
            from siwi.bot.optimized_bert_classifier import create_bert_classifier as create_optimized
            print("[INFO] 使用优化版BERT分类器")
            return create_optimized(prefer_offline=False, use_fast=False)
        except ImportError as e:
            print(f"[WARN] 优化版BERT分类器导入失败: {e}")
    
    # 回退到原版本
    try:
        return BERTClassifier()
    except Exception as e:
        print(f"[WARN] 完整BERT分类器初始化失败，使用轻量版: {e}")
        return BERTClassifierLite()


if __name__ == "__main__":
    # 测试用例
    classifier = create_bert_classifier()
    
    test_sentences = [
        "What is the relationship between Yao Ming and Lakers?",
        "Who is similar to LeBron James?", 
        "Which team did Kobe Bryant serve?",
        "Who does Stephen Curry follow?",
        "Hello there"
    ]
    
    for sentence in test_sentences:
        print(f"\n测试: {sentence}")
        result = classifier.get(sentence)
        print(f"结果: {result}")
