# import os
# from langchain_experimental.graph_transformers import LLMGraphTransformer
# from langchain_openai import ChatOpenAI,OpenAI
# from langchain_core.documents import Document
# # from langchain.graph_stores import Neo4jGraphStore
# from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
# from llama_index.core.query_engine import RetrieverQueryEngine
# from llama_index.core.retrievers import KnowledgeGraphRAGRetriever
# # from llama_index.core.response_synthesis import ResponseSynthesizer
# from llama_index.core.data_structs import Node
# from llama_index.core.response_synthesizers import ResponseMode
# from llama_index.core import get_response_synthesizer
# # from langchain_google_genai import ChatGoogleGenerativeAI
# import google.generativeai as genai

# class KnowledgeGraphPipeline:
#     def __init__(self, text):
#         """
#         Initializes the Knowledge Graph Pipeline with text input, OpenAI API key, and Neo4j credentials.
#         """
#         self.texts = text
#         self.neo4j_url = "neo4j+s://e968b27a.databases.neo4j.io"
#         self.neo4j_username = "neo4j"
#         self.neo4j_password = "jfIivFR8ZFturZpYMs483we2q9WOyfowIV18BkgFxOs"
#         self.database="neo4j"
#         # os.environ["OPENAI_API_KEY"] = "openai_api_key"
#         self.llm = OpenAI(
#             temperature=0,
#             # model_name='gpt-4o-mini'
#         )
#         self.graph_store = Neo4jPropertyGraphStore(url=self.neo4j_url, username=self.neo4j_username, password=self.neo4j_password,database=self.database)
# #         response_synthesizer = get_response_synthesizer(
# #     response_mode=ResponseMode.COMPACT
# # )
#         self.graph_documents = self.extract_knowledge_graph()
#         # self.response_synthesizer = ResponseSynthesizer(self.llm)
        

#     def extract_knowledge_graph(self):
#         documents = [Document(page_content=self.texts)]
#         llm_transformer = LLMGraphTransformer(llm=self.llm)
#         graph_documents = llm_transformer.convert_to_graph_documents(documents)
#         # self.graph_store.write_graph(graph_documents)
#         graph_handler = MyGraphHandler(self.neo4j_url, self.neo4j_username, self.neo4j_password,self.database)
#         with self.graph_store.driver.session() as session:
#             # Handle nodes
#             for node in graph_documents.nodes:
#                 graph_handler.create_or_update_node(session, node)
#             # Handle relationships
#             for edge in graph_documents.edges:
#                 graph_handler.create_or_update_relationship(session, edge)
#         return graph_documents

#     def query_and_synthesize(self, query):
#         """
#         Retrieves relevant knowledge from the graph and synthesizes an answer using the query.
#         """
#         graph_rag_retriever = KnowledgeGraphRAGRetriever(storage_context=self.graph_store.storage_context, verbose=True)
#         query_engine = RetrieverQueryEngine.from_args(graph_rag_retriever)
#         retrieved_context = query_engine.query(query)
#         model = genai.GenerativeModel("gemini-1.5-flash")
#         response = model.generate_content(f"You are supposed to answer to the query {query} based on the retrieved context {retrieved_context} which is retrieved from a graph store. Answer to the question accordingly based on the context.")
#         # response = self.response_synthesizer.synthesize(query, retrieved_context)
#         print(f"Query: {query}")
#         print(f"Answer: {response}\n")
#         return response


# class MyGraphHandler:
#     def __init__(self, neo4j_url, neo4j_username, neo4j_password,database):
#         self.graph_store = Neo4jPropertyGraphStore(url=neo4j_url, username=neo4j_username, password=neo4j_password,database=database)

#     def check_if_node_exists(self, session, node):
#         """
#         Check if a node with a specific ID exists in the graph.
#         """
#         query = """
#         MATCH (n {id: $id})
#         RETURN count(n) > 0 AS exists
#         """
#         result = session.run(query, id=node.id)
#         return result.single()["exists"]

#     def create_or_update_node(self, session, node):
#         """
#         Creates or updates nodes in the graph.
#         """
#         if self.check_if_node_exists(session, node):
#             # Node exists, update it
#             session.write_transaction(self._update_node, node)
#         else:
#             # Node doesn't exist, create it
#             session.write_transaction(self._create_node, node)

#     def create_or_update_relationship(self, session, edge):
#         # source_id = edge['source']
#         # target_id = edge['target']
#         # relation = edge['relation']
#         # Create or update the relationship using a MERGE query
#         session.write_transaction(self._create_or_update_relationship, edge)

#     # Helper methods for creating and updating nodes
#     @staticmethod
#     def _create_node(tx, node):
#         query = """
#         MERGE (n:{label} {id: $id})
#         SET n += $properties
#         """.format(label=node.label)
        
#         tx.run(query, id=node.id, properties=node.properties)

#     @staticmethod
#     def _update_node(tx, node):
#         query = """
#         MATCH (n {id: $id})
#         SET n += $properties
#         """
#         tx.run(query, id=node.id, properties=node.properties)

#     # Helper method for creating or updating relationships
#     @staticmethod
#     def _create_or_update_relationship(tx, edge):
#         query = """
#         MATCH (a {id: $start_id})
#         MATCH (b {id: $end_id})
#         MERGE (a)-[r:{type}]->(b)
#         SET r += $properties
#         """
#         tx.run(query, start_id=edge['start_node'], end_id=edge['end_node'], properties=edge['properties'], type=edge['type'])
























# ####### version 2
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import OpenAI
# from langchain_core.documents import Document
from llama_index.core import Document
import json

from llama_index.core import Settings
from llama_index.core import KnowledgeGraphIndex
from llama_index.core import StorageContext

from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI


class KnowledgeGraphPipeline:
    def __init__(self, text):
        """
        Initializes the Knowledge Graph Pipeline with text input, OpenAI API key, and Neo4j credentials.
        """
        self.texts = text
        self.neo4j_url = "neo4j+s://e968b27a.databases.neo4j.io"
        self.neo4j_username = "neo4j"
        self.neo4j_password = "jfIivFR8ZFturZpYMs483we2q9WOyfowIV18BkgFxOs"
        self.database="neo4j"
        self.llm = OpenAI(
            temperature=0,
            # model_name='gpt-4o-mini'
        )
        Settings.llm = self.llm
        self.graph_store = Neo4jGraphStore(url=self.neo4j_url, username=self.neo4j_username, password=self.neo4j_password,database=self.database)
        self.storage_context = StorageContext.from_defaults(graph_store=self.graph_store)
        # self.graph_documents = self.extract_knowledge_graph()
        self.graph = Neo4jGraph(url="neo4j+s://e968b27a.databases.neo4j.io", username="neo4j", password="jfIivFR8ZFturZpYMs483we2q9WOyfowIV18BkgFxOs")
        self.chain = GraphCypherQAChain.from_llm(
    ChatOpenAI(temperature=0), graph=self.graph, verbose=True,allow_dangerous_requests=True
)
    
    def extract_knowledge_graph(self,query):
        # documents = [Document(page_content=self.texts)]
        text_list = [self.texts]
        documents = [Document(text=t) for t in text_list]
        # llm_transformer = LLMGraphTransformer(llm=self.llm)
        # graph_documents = llm_transformer.convert_to_graph_documents(documents)
        # self.graph_store.write_graph(graph_documents)
        index = KnowledgeGraphIndex.from_documents(documents,storage_context=self.storage_context,max_triplets_per_chunk=4)
        query_engine = index.as_query_engine(llm=self.llm)
        res = query_engine.query("list the ports")
        response = self.chain.invoke({"query": query})
        return response

