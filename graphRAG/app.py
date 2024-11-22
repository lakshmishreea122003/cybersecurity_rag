import streamlit as st
from mongoDB import DatabaseClient
from build_graph import KnowledgeGraphPipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# App Title
st.title("Cybersecurity Graph-based RAG")
mdb = DatabaseClient()
text=""
# Start Button
if st.button('fetch data'):
    st.write("fetching data from MOngoDB")
    # text = mdb.get_first_document()
    text = """nmap $target -p- -T4 -Pn --open --reason
Nmap scan report for 10.10.11.248
Host is up, received user-set (0.057s latency).
Not shown: 63049 closed tcp ports (conn-refused), 2481 filtered tcp ports (no-response)
Some closed ports may be reported as filtered due to --defeat-rst-ratelimit
PORT     STATE SERVICE REASON
22/tcp   open  ssh     syn-ack
80/tcp   open  http    syn-ack
389/tcp  open  ldap    syn-ack
443/tcp  open  https   syn-ack
5667/tcp open  unknown syn-ack
 
Nmap done: 1 IP address (1 host up) scanned in 33.02 seconds"""
    st.write(text)

# langGraph
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)


def rag(state: State):
    pipeline = KnowledgeGraphPipeline(text)
    res = pipeline.extract_knowledge_graph(state["messages"])
    return {"messages": [res["result"]]}

graph_builder.add_node("rag", rag)
graph_builder.add_edge(START, "rag")
graph_builder.add_edge("rag", END)
graph = graph_builder.compile()

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            st.write(value)


# Show response
query = st.text_input("ask query here")
if query:
    stream_graph_updates(query)
    


# app
