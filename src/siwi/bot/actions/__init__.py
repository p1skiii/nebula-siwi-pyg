import importlib
import siwi
import yaml
from siwi.bot.gnn_processor import create_gnn_processor


class SiwiActions():
    def __init__(self) -> None:
        self.intent_map = {}
        self.load_data()

    def load_data(self) -> None:
        # load data from yaml files
        module_path = f"{ siwi.__path__[0] }/bot/test/data"

        with open(f"{ module_path }/example_intents.yaml", "r") as file:
            self.intent_map = yaml.safe_load(file)["intents"]

    def get(self, intent: dict):
        """
        returns SiwiActionBase - 支持BERT NLU + GNN集成
        """
        if len(intent["intents"]) > 0:
            intent_name = intent["intents"][0]
        else:
            intent_name = "fallback"

        print(f"[DEBUG] SiwiActions处理意图: {intent_name}")

        # 优先处理GNN相似度查询
        if intent_name == "find_similar":
            return GNNAction(intent)
        
        # 使用增强版关系查询（结合GNN）
        if intent_name == "relationship":
            return ImprovedRelationshipAction(intent)
        
        # 其他意图使用原有的映射机制
        if intent_name in self.intent_map:
            cls_name = self.intent_map.get(intent_name).get("action")
            action_cls = getattr(
                importlib.import_module("siwi.bot.actions"), cls_name)
            action = action_cls(intent)
            return action
        else:
            # 未知意图，返回fallback
            print(f"[WARN] 未知意图 {intent_name}，使用fallback")
            return FallbackAction(intent)


class SiwiActionBase():
    def __init__(self, intent: dict):
        """
        intent:
        {
            "entities": entities,
            "intents": intents
        }
        """
        self.load_test_data()
        self.error = False

    def load_test_data(self) -> None:
        module_path = f"{ siwi.__path__[0] }/bot/test/data"

        with open(f"{ module_path }/example_players.yaml", "r") as file:
            self.players = yaml.safe_load(file)

        with open(f"{ module_path }/example_teams.yaml", "r") as file:
            self.teams = yaml.safe_load(file)

        self.player_names = {
            value: key for (key, value) in self.players.items()
            }
        self.team_names = {
            value: key for (key, value) in self.teams.items()
            }

    def _name(self, vid: str) -> str:
        if vid.startswith("player"):
            return self.player_names.get(vid, "unknown player")
        elif vid.startswith("team"):
            return self.team_names.get(vid, "unkonwn team")
        else:
            return "unkonwn"

    def _vid(self, name: str) -> str:
        if name in self.players:
            return self.players[name]
        elif name in self.teams:
            return self.teams[name]
        else:
            print(
                f"[ERROR] Something went wrong, unknown vertex name { name }")
            raise

    def _error_check(self):
        if self.error:
            return "Opps, something went wrong."


class FallbackAction(SiwiActionBase):
    def __init__(self, intent):
        super().__init__(intent)

    def execute(self, connection_pool=None):
        """
        TBD: query some information via nbi_api in fallback case:
        https://github.com/swar/nba_api/blob/master/docs/examples/Basics.ipynb
        """
        return """
Sorry I don't understand your questions for now.
Here are supported question patterns:

relation:
    - What is the relationship between Yao Ming and Lakers?
    - How does Yao Ming and Lakers connected?
serving:
    - Which team had Yao Ming served?
friendship:
    - Whom does Tim Duncan follow?
    - Who are Yao Ming's friends?
"""


class RelationshipAction(SiwiActionBase):
    """
    USE basketballplayer;
    FIND NOLOOP PATH
    FROM "player100" TO "team204" OVER * BIDIRECT UPTO 4 STEPS YIELD path AS p;
    """
    def __init__(self, intent):
        print(f"[DEBUG] RelationshipAction intent: { intent }")
        super().__init__(intent)
        try:
            self.entity_left, self.entity_right = intent["entities"]
            self.left_vid = self._vid(self.entity_left)
            self.right_vid = self._vid(self.entity_right)
        except Exception:
            print(
                f"[WARN] RelationshipAction entities recognition Failure "
                f"will fallback to FallbackAction, "
                f"intent: { intent }"
                )
            self.error = True

    def execute(self, connection_pool) -> str:
        self._error_check()
        query = (
            f'USE basketballplayer;'
            f'FIND NOLOOP PATH '
            f'FROM "{self.left_vid}" TO "{self.right_vid}" '
            f'OVER * BIDIRECT UPTO 4 STEPS YIELD path AS p;'
            )
        print(
            f"[DEBUG] query for RelationshipAction :\n\t{ query }"
            )
        with connection_pool.session_context("root", "nebula") as session:
            result = session.execute(query)

        if not result.is_succeeded():
            return (
                f"Something is wrong on Graph Database connection when query "
                f"{ query }"
                )

        if result.is_empty():
            return (
                f"There is no relationship between "
                f"{ self.entity_left } and { self.entity_right }"
                )
        path = result.row_values(0)[0].as_path()
        relationships = path.relationships()
        relations_str = self._name(
            relationships[0].start_vertex_id().as_string())
        for rel_index in range(path.length()):
            rel = relationships[rel_index]
            relations_str += (
                f" { rel.edge_name() }s "
                f"{ self._name(rel.end_vertex_id().as_string()) }")
        return (
            f"There are at least { result.row_size() } relations between "
            f"{ self.entity_left } and { self.entity_right }, "
            f"one relation path is: { relations_str }."
            )


class ServeAction(SiwiActionBase):
    """
    USE basketballplayer;
    MATCH p=(v)-[e:serve*1]->(v1)
    WHERE id(v) == "player133"
         RETURN p LIMIT 100
    """
    def __init__(self, intent):
        print(f"[DEBUG] ServeAction intent: { intent }")
        super().__init__(intent)
        try:
            self.player0 = list(intent["entities"].keys())[0]
            self.player0_vid = self._vid(self.player0)
        except Exception:
            print(
                f"[WARN] ServeAction entities recognition Failure "
                f"will fallback to FallbackAction, "
                f"intent: { intent }"
                )
            self.error = True

    def execute(self, connection_pool) -> str:
        self._error_check()
        query = (
            f'USE basketballplayer;'
            f'MATCH p=(v)-[e:serve*1]->(v1) '
            f'WHERE id(v) == "{ self.player0_vid }" '
            f'    RETURN p LIMIT 100;'
            )
        print(
            f"[DEBUG] query for RelationshipAction :\n\t{ query }"
            )
        with connection_pool.session_context("root", "nebula") as session:
            result = session.execute(query)

        if not result.is_succeeded():
            return (
                f"Something is wrong on Graph Database connection when query "
                f"{ query }"
                )

        if result.is_empty():
            return (
                f"There is no teams served by "
                f"{ self.player0 }"
                )
        serving_teams_str = ""
        for index in range(result.row_size()):
            rel = result.row_values(index)[0].as_path().relationships()[0]
            serving_teams_str += (
                f"{ self._name(rel.end_vertex_id().as_string()) } "
                f"from { rel.properties()['start_year'] } "
                f"to { rel.properties()['start_year'] }; "
                )
        return (
            f"{ self.player0 } had served { result.row_size() } team"
            f"{'s' if result.row_size() > 1 else ''}. "
            f"{ serving_teams_str }"
            )


class FollowAction(SiwiActionBase):
    """
    USE basketballplayer;
    MATCH p=(v)-[e:follow*1]->(v1)
    WHERE id(v) == "player133"
         RETURN p LIMIT 100
    """
    def __init__(self, intent):
        print(f"[DEBUG] FollowAction intent: { intent }")
        super().__init__(intent)
        try:
            self.player0 = list(intent["entities"].keys())[0]
            self.player0_vid = self._vid(self.player0)
        except Exception:
            print(
                f"[WARN] ServeAction entities recognition Failure "
                f"will fallback to FallbackAction, "
                f"intent: { intent }"
                )
            self.error = True

    def execute(self, connection_pool) -> str:
        self._error_check()
        query = (
            f'USE basketballplayer;'
            f'MATCH p=(v)-[e:follow*1]->(v1) '
            f'WHERE id(v) == "{ self.player0_vid }" '
            f'    RETURN p LIMIT 100;'
            )
        print(
            f"[DEBUG] query for RelationshipAction :\n\t{ query }"
            )
        with connection_pool.session_context("root", "nebula") as session:
            result = session.execute(query)

        if not result.is_succeeded():
            return (
                f"Something is wrong on Graph Database connection when query "
                f"{ query }"
                )

        if result.is_empty():
            return (
                f"There is no players followed by "
                f"{ self.player0 }"
                )
        following_players_str = ""
        for index in range(result.row_size()):
            rel = result.row_values(index)[0].as_path().relationships()[0]
            following_players_str += (
                f"{ self._name(rel.end_vertex_id().as_string()) } "
                f"in degree { rel.properties()['degree'] }; "
                )
        return (
            f"{ self.player0 } had followed { result.row_size() } player"
            f"{'s' if result.row_size() > 1 else ''}. "
            f"{ following_players_str }"
            )


class GNNAction(SiwiActionBase):
    """
    GNN相似度查询Action
    处理"find_similar"意图，使用图神经网络查找相似的球员/球队
    """
    
    def __init__(self, intent):
        print(f"[DEBUG] GNNAction intent: {intent}")
        super().__init__(intent)
        
        # 初始化GNN处理器
        try:
            self.gnn_processor = create_gnn_processor()
            print("[INFO] GNN处理器初始化成功")
        except Exception as e:
            print(f"[ERROR] GNN处理器初始化失败: {e}")
            self.gnn_processor = None
            self.error = True
        
        # 提取目标实体
        try:
            entities = list(intent["entities"].keys())
            if entities:
                self.target_entity = entities[0]
                print(f"[DEBUG] GNN目标实体: {self.target_entity}")
            else:
                print("[WARN] GNN Action 未找到目标实体")
                self.error = True
        except Exception as e:
            print(f"[WARN] GNN Action 实体提取失败: {e}")
            self.error = True
    
    def execute(self, connection_pool=None) -> str:
        """执行GNN相似度查询"""
        if self.error or self.gnn_processor is None:
            return ("抱歉，GNN系统暂时不可用。请尝试其他查询，"
                   "如：'What is the relationship between Yao Ming and Lakers?'")
        
        try:
            # 使用GNN查找相似节点
            similar_nodes = self.gnn_processor.get_similar(
                self.target_entity, 
                top_k=3, 
                exclude_self=True
            )
            
            if not similar_nodes:
                return f"暂时找不到与 {self.target_entity} 相似的球员或球队。"
            
            # 检查是否是错误消息
            if len(similar_nodes) == 1 and "不存在" in similar_nodes[0]:
                return similar_nodes[0]
            
            # 格式化返回结果
            result_lines = [f"基于图神经网络分析，与 {self.target_entity} 最相似的是：\n"]
            
            for i, similar_node in enumerate(similar_nodes, 1):
                result_lines.append(f"{i}. {similar_node}")
            
            # 添加相似度分数（如果可用）
            try:
                similarity_scores = []
                for node in similar_nodes:
                    score = self.gnn_processor.compute_similarity(self.target_entity, node)
                    similarity_scores.append(f"相似度: {score:.3f}")
                
                # 添加分数信息
                result_lines.append(f"\n💡 这些结果基于图结构和节点嵌入计算得出")
                
            except Exception as e:
                print(f"[DEBUG] 相似度分数计算失败: {e}")
            
            return "\n".join(result_lines)
            
        except Exception as e:
            print(f"[ERROR] GNN Action执行失败: {e}")
            return f"查询 {self.target_entity} 的相似节点时出现错误，请稍后重试。"


class ImprovedRelationshipAction(SiwiActionBase):
    """
    增强版关系查询Action
    结合传统图查询和GNN嵌入相似度
    """
    
    def __init__(self, intent):
        print(f"[DEBUG] ImprovedRelationshipAction intent: {intent}")
        super().__init__(intent)
        
        try:
            self.entity_left, self.entity_right = intent["entities"]
            self.left_vid = self._vid(self.entity_left)
            self.right_vid = self._vid(self.entity_right)
            
            # 初始化GNN处理器（可选）
            try:
                self.gnn_processor = create_gnn_processor(use_lite=True)
            except:
                self.gnn_processor = None
                
        except Exception:
            print(f"[WARN] ImprovedRelationshipAction entities recognition failure")
            self.error = True
    
    def execute(self, connection_pool) -> str:
        """执行增强版关系查询"""
        if self.error:
            return self._error_check()
        
        # 首先执行传统的图查询
        traditional_result = self._execute_traditional_query(connection_pool)
        
        # 如果有GNN处理器，添加相似度信息
        if self.gnn_processor:
            try:
                similarity = self.gnn_processor.compute_similarity(
                    self.entity_left, self.entity_right
                )
                gnn_info = f"\n\n🤖 GNN分析：{self.entity_left} 和 {self.entity_right} 的嵌入相似度为 {similarity:.3f}"
                return traditional_result + gnn_info
            except:
                pass
        
        return traditional_result
    
    def _execute_traditional_query(self, connection_pool) -> str:
        """执行传统的图数据库查询"""
        query = (
            f'USE basketballplayer;'
            f'FIND NOLOOP PATH '
            f'FROM "{self.left_vid}" TO "{self.right_vid}" '
            f'OVER * BIDIRECT UPTO 4 STEPS YIELD path AS p;'
        )
        
        print(f"[DEBUG] query for ImprovedRelationshipAction: {query}")
        
        with connection_pool.session_context("root", "nebula") as session:
            result = session.execute(query)

        if not result.is_succeeded():
            return f"图数据库查询失败: {query}"

        if result.is_empty():
            return f"{self.entity_left} 和 {self.entity_right} 之间没有直接的关系路径"

        path = result.row_values(0)[0].as_path()
        relationships = path.relationships()
        relations_str = self._name(relationships[0].start_vertex_id().as_string())
        
        for rel_index in range(path.length()):
            rel = relationships[rel_index]
            relations_str += f" {rel.edge_name()}s {self._name(rel.end_vertex_id().as_string())}"
        
        return (f"找到 {result.row_size()} 条关系路径，"
               f"其中一条是: {relations_str}")
