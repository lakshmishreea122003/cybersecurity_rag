# cybersecurity_rag

## Introduction
I developed a graph-based Retrieval-Augmented Generation (RAG) system for cybersecurity using LangGraph and LangChain. This system processes raw text and images from penetration testing outputs, constructing and updating a knowledge graph through LLMs to generate accurate, context-aware responses. The final output provides plaintext responses, integrating relevant context from the knowledge graph to enhance insight into vulnerabilities and system weaknesses.

## Project Workflow
![Project Flow](https://github.com/lakshmishreea122003/cybersecurity_rag/blob/main/flowcharts/Screenshot%202024-11-14%20084524.jpg)
-Text to Knowledge Graph: The project converts input text into a knowledge graph using an LLM (Large Language Model) transformer, creating nodes and relationships from the text.
Graph Storage: The generated knowledge graph is stored in a Neo4j database for efficient retrieval and querying.
-Node and Relationship Management: The project handles the creation and updating of nodes and relationships in the graph using Cypher queries.
-Graph Querying: Users can query the knowledge graph with natural language queries to retrieve relevant information about entities and their relationships in the cybersecurity domain.
-Response Synthesis: The project synthesizes answers to queries by retrieving relevant context from the graph and generating responses using OpenAI's GPT model.
-RAG (Retriever-Augmented Generation): The system uses a retrieval-augmented generation approach, combining the graph's retrieved knowledge with GPT for answer synthesis.
-Cybersecurity Focus: The project is designed to enhance understanding and response generation for cybersecurity-related knowledge through a graph-based approach.
-Automated Knowledge Updates: The graph can be continuously updated with new knowledge, keeping the information up-to-date and relevant for ongoing queries.


