import streamlit as st
from mongoDB import DatabaseClient
from build_graph import KnowledgeGraphPipeline

# App Title
st.title("Cybersecurity Graph-based RAG")
mdb = DatabaseClient()

# Start Button
if st.button('fetch data'):
    st.write("fetching data from MOngoDB")
    text = mdb.get_first_document()
    pipeline = KnowledgeGraphPipeline(text)
    # Show second text
query = st.text_input("ask query here")
if query:
    res = pipeline.query_and_synthesize(query)
    st.write(res)