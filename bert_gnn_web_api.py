"""
BERT + GNN集成版Web API
提供RESTful接口支持新的NLU和GNN功能
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import sys
import os

# 添加项目路径
sys.path.append('/Users/wang/i/nebula-siwi/src')

from siwi.bot.bot import SiwiBot
from siwi.feature_store import get_nebula_connection_pool
from siwi.bot.bert_classifier import create_bert_classifier
from siwi.bot.gnn_processor import create_gnn_processor

app = Flask(__name__)
CORS(app)

# 全局变量
siwi_bot = None
bert_classifier = None
gnn_processor = None


def initialize_system():
    """初始化系统组件"""
    global siwi_bot, bert_classifier, gnn_processor
    
    try:
        print("[INFO] 初始化系统组件...")
        
        # 初始化连接池
        connection_pool = get_nebula_connection_pool()
        
        # 初始化Bot
        siwi_bot = SiwiBot(connection_pool)
        
        # 初始化独立组件（用于单独测试）
        bert_classifier = create_bert_classifier()
        gnn_processor = create_gnn_processor(use_lite=True)
        
        print("[INFO] 系统初始化完成")
        return True
        
    except Exception as e:
        print(f"[ERROR] 系统初始化失败: {e}")
        return False


@app.route('/')
def home():
    """主页"""
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Siwi BERT+GNN API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { background: #007bff; color: white; padding: 5px 10px; border-radius: 3px; }
            pre { background: #f8f9fa; padding: 10px; border-radius: 3px; overflow-x: auto; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Siwi BERT+GNN API</h1>
            <p>NBA知识图谱问答系统 - 集成BERT NLU + 图神经网络</p>
            
            <h2>📡 API接口</h2>
            
            <div class="endpoint">
                <h3><span class="method">POST</span> /api/query</h3>
                <p>主要问答接口</p>
                <pre>
{
    "question": "Who is similar to Yao Ming?"
}
                </pre>
            </div>
            
            <div class="endpoint">
                <h3><span class="method">POST</span> /api/bert/classify</h3>
                <p>BERT分类器测试</p>
                <pre>
{
    "sentence": "What is the relationship between Yao Ming and Lakers?"
}
                </pre>
            </div>
            
            <div class="endpoint">
                <h3><span class="method">POST</span> /api/gnn/similar</h3>
                <p>GNN相似度查询</p>
                <pre>
{
    "node": "Yao Ming",
    "top_k": 3
}
                </pre>
            </div>
            
            <div class="endpoint">
                <h3><span class="method">GET</span> /api/health</h3>
                <p>系统健康检查</p>
            </div>
            
            <h2>💡 示例查询</h2>
            <ul>
                <li>"Who is similar to Yao Ming?" - GNN相似度</li>
                <li>"What is the relationship between Yao Ming and Lakers?" - 关系查询</li>
                <li>"Which team did LeBron James serve?" - 服务历史</li>
                <li>"Find someone like Stephen Curry" - 相似球员</li>
            </ul>
        </div>
    </body>
    </html>
    """)


@app.route('/api/query', methods=['POST'])
def query():
    """主要问答接口"""
    try:
        data = request.get_json()
        question = data.get('question', '')
        
        if not question:
            return jsonify({'error': '问题不能为空'}), 400
        
        if siwi_bot is None:
            return jsonify({'error': '系统未初始化'}), 500
        
        # 使用SiwiBot处理查询
        answer = siwi_bot.query(question)
        
        return jsonify({
            'question': question,
            'answer': answer,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/bert/classify', methods=['POST'])
def bert_classify():
    """BERT分类器测试接口"""
    try:
        data = request.get_json()
        sentence = data.get('sentence', '')
        
        if not sentence:
            return jsonify({'error': '句子不能为空'}), 400
        
        if bert_classifier is None:
            return jsonify({'error': 'BERT分类器未初始化'}), 500
        
        # 使用BERT分类器
        result = bert_classifier.get(sentence)
        
        return jsonify({
            'sentence': sentence,
            'entities': result['entities'],
            'intents': result['intents'],
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/gnn/similar', methods=['POST'])
def gnn_similar():
    """GNN相似度查询接口"""
    try:
        data = request.get_json()
        node = data.get('node', '')
        top_k = data.get('top_k', 3)
        
        if not node:
            return jsonify({'error': '节点名称不能为空'}), 400
        
        if gnn_processor is None:
            return jsonify({'error': 'GNN处理器未初始化'}), 500
        
        # 使用GNN查找相似节点
        similar_nodes = gnn_processor.get_similar(node, top_k=top_k)
        
        return jsonify({
            'node': node,
            'similar_nodes': similar_nodes,
            'top_k': top_k,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/gnn/stats', methods=['GET'])
def gnn_stats():
    """GNN图统计信息"""
    try:
        if gnn_processor is None:
            return jsonify({'error': 'GNN处理器未初始化'}), 500
        
        stats = gnn_processor.get_graph_stats()
        all_nodes = gnn_processor.get_all_nodes()
        
        return jsonify({
            'stats': stats,
            'all_nodes': all_nodes,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """系统健康检查"""
    health_status = {
        'siwi_bot': siwi_bot is not None,
        'bert_classifier': bert_classifier is not None,
        'gnn_processor': gnn_processor is not None,
        'overall': False
    }
    
    health_status['overall'] = all(health_status.values())
    
    return jsonify({
        'health': health_status,
        'status': 'healthy' if health_status['overall'] else 'degraded'
    })


@app.route('/api/examples', methods=['GET'])
def get_examples():
    """获取示例查询"""
    examples = [
        {
            "question": "Who is similar to Yao Ming?",
            "type": "GNN相似度查询",
            "description": "使用图神经网络查找相似球员"
        },
        {
            "question": "What is the relationship between Yao Ming and Lakers?",
            "type": "关系查询",
            "description": "查找两个实体间的关系路径"
        },
        {
            "question": "Which team did LeBron James serve?",
            "type": "服务历史",
            "description": "查询球员的服务历史"
        },
        {
            "question": "Find someone like Stephen Curry",
            "type": "相似度搜索",
            "description": "自然语言相似度查询"
        },
        {
            "question": "Who does Tim Duncan follow?",
            "type": "关注关系",
            "description": "查询球员的关注关系"
        }
    ]
    
    return jsonify({
        'examples': examples,
        'status': 'success'
    })


if __name__ == '__main__':
    print("🚀 启动Siwi BERT+GNN Web API")
    print("="*50)
    
    # 初始化系统
    if initialize_system():
        print("✅ 系统初始化成功，启动Web服务")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("❌ 系统初始化失败，无法启动Web服务")
        sys.exit(1)
