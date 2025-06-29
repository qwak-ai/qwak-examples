from typing import List, Dict
from qwak.model.base import QwakModel
import qwak
from qwak.model.adapters import JsonInputAdapter, JsonOutputAdapter

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from mlflow.langchain.chat_agent_langgraph import ChatAgentState
from langgraph.graph import START, StateGraph
from mlflow.types.agent import ChatAgentMessage

from .chroma_retriever import ChromaRetriever
from .jfml_local_chat_model import JFMLLocalChatModel
from .lang_graph_chat_agent import LangGraphChatAgent

import pandas as pd
import os
import yaml

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
INPUT_CSV = os.getenv("CONTEXT_INPUT_CSV", "main/dummy_data.csv")
CONFIG_FILE_PATH = 'main/rag_chain_config.yaml'

class JFrogChatAgent(QwakModel):

    def __init__(self):
        self.agent = None
        self.retriever_manager = None
        self.persist_directory = None
        
        # 1. Read the configuration using PyYAML
        print(f"\nReading configuration from {CONFIG_FILE_PATH}...")
        with open(CONFIG_FILE_PATH, 'r') as file:
            self.app_config = yaml.safe_load(file)


    def build(self):

        df = pd.read_csv(INPUT_CSV)

        # Convert DataFrame rows to LangChain Document objects
        documents = []
        for index, row in df.iterrows():
            page_content = row['Text']
            metadata = row.drop('Text').to_dict() # Use other columns as metadata
            documents.append(Document(page_content=page_content, metadata=metadata))

        print(f"Loaded {len(documents)} documents from the CSV data.")

        chroma_db_name = (
            self.app_config
                .get("retriever_config")
                .get("vector_search_index", "default_chroma_db")
        )

        self.persist_directory = os.path.join("./chroma_dbs", chroma_db_name) # Create a subfolder for dbs


        # 2. Instantiate the ChromaRetriever
        retriever_manager = ChromaRetriever(
            embedding_model_name = self.app_config.get("embedding_config").get("model_name"),
            persist_directory = self.persist_directory)
        

        # --- First Run: Create and Persist the DB ---
        print("\n--- FIRST RUN: Creating and Persisting DB ---")
        # Use overwrite=True to ensure a clean start if you run this multiple times for testing
        retriever_manager.create_and_persist_db(documents=documents, overwrite=True)


    def initialize_model(self):

        # 2. Instantiate the ChromaRetriever
        retriever_manager = ChromaRetriever(
            embedding_model_name = self.app_config.get("embedding_config").get("model_name"),
            persist_directory = self.persist_directory)
        

        # Load the existing database from disk
        retriever_manager.load_db()

        retriever = retriever_manager.get_retriever(
            search_kwargs = self.app_config.get("retriever_config").get("parameters", {"k": 3})
            )

        prompt = PromptTemplate(
            template = self.app_config.get("llm_config").get("llm_prompt_template"),
            input_variables = self.app_config.get("llm_config").get("llm_prompt_template_variables"),
        )

        # Initialize retriever, prompt and LLM model
        llm_model = JFMLLocalChatModel(**self.app_config.get("llm_config").get("llm_parameters"))

        # Define retrieve and generate functions
        def retrieve(state: ChatAgentState):
            last_message = state["messages"][-1]
            question = last_message['content']
            retrieved_docs = retriever.invoke(question)
            return {'context': retrieved_docs}

        def generate(state: ChatAgentState):
            last_message = state["messages"][-1]
            question = last_message['content']
            docs_content = "\n\n".join(doc.page_content for doc in state["context"])
            messages = prompt.invoke({"question": question, "context": docs_content})
            response = llm_model.invoke(messages)
            return {"messages": [response]}

        # Compile application and test
        graph_builder = StateGraph(ChatAgentState).add_sequence([retrieve, generate])
        graph_builder.add_edge(START, "retrieve")
        graph = graph_builder.compile()

        # Initialize LangGraphChatAgent
        self.agent = LangGraphChatAgent(graph)


    # API method for prediction
    # Decorated with frogml API to handle JSON input and output formats
    @qwak.api(input_adapter=JsonInputAdapter(), 
                output_adapter=JsonOutputAdapter())
    def predict(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:

        formatted_messages = [ChatAgentMessage(**msg) for msg in messages]
        response = self.agent.predict(formatted_messages)
        return [msg.model_dump() for msg in response.messages] # Convert ChatAgentMessage objects to dicts
