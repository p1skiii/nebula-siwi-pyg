import torch
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config

NEBULA_HOST = '127.0.0.1'
NEBULA_PORT = 9669
NEBULA_USER = 'root'
NEBULA_PASSWORD = 'nebula'
NEBULA_GRAPH_SPACE = 'basketballplayer'

_connection_pool = None

def get_nebula_connection_pool():
    global _connection_pool
    if _connection_pool is None:
        config = Config()
        config.max_connection_pool_size = 10
        _connection_pool = ConnectionPool()
        if not _connection_pool.init([(NEBULA_HOST, NEBULA_PORT)], config):
            raise RuntimeError("Failed to initialize NebulaGraph connection pool")
    return _connection_pool

def get_entity_embedding(entity_id: str, entity_tag: str = "player", embedding_field: str = "embedding1") -> float | None:
    """
    从 NebulaGraph 中获取指定实体的 embedding 值。
    
    参数:
    - entity_id: 实体ID
    - entity_tag: 实体类型标签，默认为"player"
    - embedding_field: embedding字段名，默认为"embedding1"
    
    返回:
    - 浮点数形式的embedding值
    - 如果获取失败，返回None
    """
    pool = get_nebula_connection_pool()
    session = None
    try:
        session = pool.get_session(NEBULA_USER, NEBULA_PASSWORD)
        session.execute(f"USE {NEBULA_GRAPH_SPACE};")
        
        query = f'FETCH PROP ON {entity_tag} "{entity_id}" YIELD properties(vertex).{embedding_field}'
        
        result = session.execute(query)
        if not result.is_succeeded() or result.row_size() == 0:
            # 查询失败或没有结果，直接返回None
            return None
        
        embedding_value_wrapper = result.row_values(0)[0]
        if embedding_value_wrapper.is_empty():
            # 属性不存在，返回None
            return None
        
        # 返回浮点数值
        return embedding_value_wrapper.as_double()

    except Exception:
        # 出现异常，返回None
        return None
    finally:
        if session:
            session.release()

def convert_embedding_to_tensor(embedding_value: float | None) -> torch.Tensor | None:
    """
    将embedding值转换为PyTorch Tensor。
    
    参数:
    - embedding_value: 单个浮点数或None
    
    返回:
    - 1维Tensor或None
    """
    if embedding_value is None:
        return None
    
    try:
        # 将单个值转换为1维tensor
        return torch.tensor([float(embedding_value)], dtype=torch.float32)
    except Exception as e:
        print(f"Error converting to PyTorch Tensor: {e}")
        return None

def get_entity_embedding_tensor(entity_id: str, entity_tag: str = "player", 
                              embedding_field: str = "embedding1") -> torch.Tensor | None:
    """
    获取实体的embedding并转换为Tensor。
    """
    embedding_value = get_entity_embedding(entity_id, entity_tag, embedding_field)
    return convert_embedding_to_tensor(embedding_value)

