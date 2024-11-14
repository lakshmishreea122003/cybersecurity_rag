from pymongo.mongo_client import MongoClient
from urllib.parse import quote_plus
import streamlit as st

class DatabaseClient:
    def __init__(self):
        self.username = quote_plus("username")
        self.password = quote_plus("password")
        self.uri = f"mongodb+srv://{self.username}:{self.password}@graphrag.6b59m.mongodb.net/?retryWrites=true&w=majority&appName=graphRAG"
        self.db_name = "db_name"
        self.collection_name = "collection_name"
        self.client = None
        self.db = None
        self.collection = None
        self.connect()
    
    def connect(self):
        try:
            self.client = MongoClient(self.uri)
            self.client.admin.command('ping')
            print("Pinged your deployment. You successfully connected to MongoDB!")
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
        except Exception as e:
            print("Connection error:", e)
    
    def insert_document(self, document: dict):
        try:
            result = self.collection.insert_one(document)
            print("Document inserted with ID:", result.inserted_id)
            return result.inserted_id
        except Exception as e:
            print("An error occurred while inserting the document:", e)
    
    def insert_many_documents(self, documents: list):
        try:
            result = self.collection.insert_many(documents)
            print("Documents inserted with IDs:", result.inserted_ids)
            return result.inserted_ids
        except Exception as e:
            print("An error occurred while inserting multiple documents:", e)
    
    def get_first_document(self):
        try:
            document = self.collection.find_one()
            if document:
                st.write(document.raw_text)
            else:
                print("No documents found in the collection.")
            return document
        except Exception as e:
            print("An error occurred while retrieving the document:", e)
    
    def get_many_documents(self, limit: int = 10):
        try:
            documents = list(self.collection.find().limit(limit))
            if documents:
                print(f"Retrieved {len(documents)} documents.")
            else:
                print("No documents found in the collection.")
            return documents
        except Exception as e:
            print("An error occurred while retrieving multiple documents:", e)
