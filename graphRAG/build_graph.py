import os
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import OpenAI
from langchain_core.documents import Document
from langchain.graph_stores import Neo4jGraphStore
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import KnowledgeGraphRAGRetriever
from llama_index.core.response_synthesis import ResponseSynthesizer


class KnowledgeGraphPipeline:
    def __init__(self, text):
        """
        Initializes the Knowledge Graph Pipeline with text input, OpenAI API key, and Neo4j credentials.
        """
        self.texts = text
        self.neo4j_url = "neo4j_url"
        self.neo4j_username = "neo4j_username"
        self.neo4j_password = "neo4j_password"
        os.environ["OPENAI_API_KEY"] = "openai_api_key"
        self.llm = OpenAI(
            temperature=0,
            max_output_tokens=1000,
            model_name='gpt-3.5-turbo'
        )
        self.graph_store = Neo4jGraphStore(url=self.neo4j_url, username=self.neo4j_username, password=self.neo4j_password)
        self.response_synthesizer = ResponseSynthesizer(self.llm)
        self.graph_documents = self.extract_knowledge_graph()

    def extract_knowledge_graph(self):
        """
        Extracts a knowledge graph from the provided texts using LLMGraphTransformer.
        """
        documents = [Document(page_content=self.texts)]
        llm_transformer = LLMGraphTransformer(llm=self.llm)
        graph_documents = llm_transformer.convert_to_graph_documents(documents)
        # self.graph_store.write_graph(graph_documents)
        graph_handler = MyGraphHandler(self.neo4j_url, self.neo4j_username, self.neo4j_password)
        with self.graph_store.driver.session() as session:
            # Handle nodes
            for node in graph_documents.nodes:
                graph_handler.create_or_update_node(session, node)
            # Handle relationships
            for edge in graph_documents.edges:
                graph_handler.create_or_update_relationship(session, edge)
        return graph_documents

    def query_and_synthesize(self, query):
        """
        Retrieves relevant knowledge from the graph and synthesizes an answer using the query.
        """
        graph_rag_retriever = KnowledgeGraphRAGRetriever(storage_context=self.graph_store.storage_context, verbose=True)
        query_engine = RetrieverQueryEngine.from_args(graph_rag_retriever)
        retrieved_context = query_engine.query(query)
        response = self.response_synthesizer.synthesize(query, retrieved_context)
        print(f"Query: {query}")
        print(f"Answer: {response}\n")
        return response


class MyGraphHandler:
    def __init__(self, neo4j_url, neo4j_username, neo4j_password):
        self.graph_store = Neo4jGraphStore(
            url=neo4j_url, 
            username=neo4j_username, 
            password=neo4j_password
        )

    def check_if_node_exists(self, session, node):
        """
        Check if a node with a specific ID exists in the graph.
        """
        query = """
        MATCH (n {id: $id})
        RETURN count(n) > 0 AS exists
        """
        result = session.run(query, id=node.id)
        return result.single()["exists"]

    def create_or_update_node(self, session, node):
        """
        Creates or updates nodes in the graph.
        """
        if self.check_if_node_exists(session, node):
            # Node exists, update it
            session.write_transaction(self._update_node, node)
        else:
            # Node doesn't exist, create it
            session.write_transaction(self._create_node, node)

    def create_or_update_relationship(self, session, edge):
        """
        Creates or updates relationships in the graph.
        """
        source_id = edge['source']
        target_id = edge['target']
        relation = edge['relation']

        # Create or update the relationship using a MERGE query
        session.write_transaction(self._create_or_update_relationship, edge)

    # Helper methods for creating and updating nodes
    @staticmethod
    def _create_node(tx, node):
        query = """
        MERGE (n:{label} {id: $id})
        SET n += $properties
        """.format(label=node.label)
        
        tx.run(query, id=node.id, properties=node.properties)

    @staticmethod
    def _update_node(tx, node):
        query = """
        MATCH (n {id: $id})
        SET n += $properties
        """
        tx.run(query, id=node.id, properties=node.properties)

    # Helper method for creating or updating relationships
    @staticmethod
    def _create_or_update_relationship(tx, edge):
        query = """
        MATCH (a {id: $start_id})
        MATCH (b {id: $end_id})
        MERGE (a)-[r:{type}]->(b)
        SET r += $properties
        """
        tx.run(query, start_id=edge['start_node'], end_id=edge['end_node'], properties=edge['properties'], type=edge['type'])
