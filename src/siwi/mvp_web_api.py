"""
MVP问答系统的Web API接口
提供简单的HTTP端点来访问路径发现功能
"""

from flask import Flask, request, jsonify
from siwi.mvp_qa_system import MVPQuestionAnswering
import traceback
import time

# 创建Flask应用
mvp_app = Flask(__name__)

# 全局MVP系统实例 (避免重复初始化)
qa_system = None

def get_qa_system():
    """获取QA系统实例 (单例模式)"""
    global qa_system
    if qa_system is None:
        print("初始化MVP问答系统...")
        qa_system = MVPQuestionAnswering()
        print("MVP问答系统初始化完成")
    return qa_system


@mvp_app.route("/")
def index():
    """主页"""
    return """
    <h1>NBA实体关系MVP问答系统</h1>
    <p>这是一个演示系统，可以回答NBA球员和球队之间的关系问题。</p>
    
    <h2>API端点:</h2>
    <ul>
        <li><code>POST /api/mvp/question</code> - 问答接口</li>
        <li><code>GET /api/mvp/entities</code> - 获取支持的实体列表</li>
        <li><code>GET /api/mvp/examples</code> - 获取示例问题</li>
    </ul>
    
    <h2>示例问题:</h2>
    <ul>
        <li>What relationships exist between Yao Ming and Lakers within 3 hops?</li>
        <li>How are Stephen Curry and Warriors connected within 2 hops?</li>
        <li>Find path from LeBron James to Kobe Bryant</li>
    </ul>
    """


@mvp_app.route("/api/mvp/question", methods=["POST"])
def answer_question():
    """主要问答接口"""
    try:
        # 获取请求数据
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({
                'success': False,
                'error': '请提供question字段',
                'example': {'question': 'What is the relationship between Yao Ming and Lakers?'}
            }), 400
        
        question = data['question'].strip()
        if not question:
            return jsonify({
                'success': False,
                'error': '问题不能为空'
            }), 400
        
        # 记录开始时间
        start_time = time.time()
        
        # 获取QA系统并处理问题
        system = get_qa_system()
        answer = system.answer_question(question)
        
        # 计算处理时间
        processing_time = time.time() - start_time
        
        return jsonify({
            'success': True,
            'question': question,
            'answer': answer,
            'processing_time': round(processing_time, 2),
            'timestamp': time.time()
        })
        
    except Exception as e:
        print(f"问答处理错误: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': str(e),
            'type': 'internal_error'
        }), 500


@mvp_app.route("/api/mvp/entities", methods=["GET"])
def get_entities():
    """获取支持的实体列表"""
    try:
        system = get_qa_system()
        extractor = system.entity_extractor
        
        return jsonify({
            'success': True,
            'players': extractor.player_map,
            'teams': extractor.team_map,
            'total_entities': len(extractor.entity_map)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@mvp_app.route("/api/mvp/examples", methods=["GET"])
def get_examples():
    """获取示例问题"""
    examples = [
        {
            'question': 'What relationships exist between Yao Ming and Lakers within 3 hops?',
            'description': '查找姚明和湖人队在3跳内的关系路径',
            'expected_entities': ['Yao Ming', 'Lakers'],
            'expected_hops': 3
        },
        {
            'question': 'How are Stephen Curry and Warriors connected within 2 hops?',
            'description': '查找库里和勇士队在2跳内的连接',
            'expected_entities': ['Stephen Curry', 'Warriors'],
            'expected_hops': 2
        },
        {
            'question': 'Find path from LeBron James to Kobe Bryant',
            'description': '查找詹姆斯到科比的路径',
            'expected_entities': ['LeBron James', 'Kobe Bryant'],
            'expected_hops': 2
        },
        {
            'question': 'What is the relationship between Tim Duncan and Spurs within 1 hop?',
            'description': '查找邓肯和马刺队的直接关系',
            'expected_entities': ['Tim Duncan', 'Spurs'],
            'expected_hops': 1
        }
    ]
    
    return jsonify({
        'success': True,
        'examples': examples,
        'count': len(examples)
    })


@mvp_app.route("/api/mvp/health", methods=["GET"])
def health_check():
    """健康检查"""
    try:
        # 简单测试系统是否正常
        system = get_qa_system()
        
        return jsonify({
            'success': True,
            'status': 'healthy',
            'system_initialized': system is not None,
            'timestamp': time.time()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'status': 'unhealthy',
            'error': str(e)
        }), 500


@mvp_app.route("/api/mvp/test", methods=["GET"])
def test_system():
    """测试系统功能"""
    try:
        system = get_qa_system()
        
        # 执行一个简单的测试问题
        test_question = "How are Stephen Curry and Warriors connected?"
        answer = system.answer_question(test_question)
        
        return jsonify({
            'success': True,
            'test_question': test_question,
            'test_answer': answer,
            'status': 'system_working'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'status': 'system_error'
        }), 500


if __name__ == "__main__":
    print("启动MVP问答系统Web服务...")
    mvp_app.run(host="0.0.0.0", port=5001, debug=True)
