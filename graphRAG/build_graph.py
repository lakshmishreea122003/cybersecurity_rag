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
        self.neo4j_url = ""
        self.neo4j_username = ""
        self.neo4j_password = ""
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

