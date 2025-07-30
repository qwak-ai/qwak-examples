from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
import os
import shutil

class ChromaRetriever:
    """
    A class to manage ChromaDB initialization, embedding creation,
    persistence, loading, and provide a LangChain retriever.
    """
    def __init__(self, embedding_model_name: str, persist_directory: str):
        """
        Initializes the ChromaRetriever with an embedding model and a persistence directory.

        Args:
            embedding_model_name (str): The name of the HuggingFace model to use for embeddings
                                        (e.g., "sentence-transformers/all-MiniLM-L6-v2").
            persist_directory (str): The local path where the ChromaDB will be stored.
        """
        self.embedding_model_name = embedding_model_name
        self.persist_directory = persist_directory
        self.embeddings = None  # To store the HuggingFaceEmbeddings instance
        self.db = None          # To store the ChromaDB instance

        self._initialize_embeddings_model()

    def _initialize_embeddings_model(self):
        """
        Initializes the HuggingFaceEmbeddings model. This is a private helper method.
        """
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
            print(f"Initialized embedding model: {self.embedding_model_name}")
        except Exception as e:
            print(f"Error initializing embedding model: {e}")
            self.embeddings = None
            raise

    def _cleanup_db_directory(self):
        """Removes the existing ChromaDB directory for a fresh start."""
        if os.path.exists(self.persist_directory):
            try:
                shutil.rmtree(self.persist_directory)
                print(f"Cleaned up existing ChromaDB at {self.persist_directory}")
            except OSError as e:
                print(f"Error removing directory {self.persist_directory}: {e}")

    def create_and_persist_db(self, documents: list[Document], overwrite: bool = False):
        """
        Creates a new ChromaDB from a list of LangChain Documents and persists it to disk.
        If overwrite is True, any existing database at the persist_directory will be removed first.

        Args:
            documents (list[Document]): A list of LangChain Document objects to embed and store.
            overwrite (bool): If True, delete existing DB before creating a new one.
        """
        if self.embeddings is None:
            print("Embedding model not initialized. Cannot create DB.")
            return

        if overwrite:
            self._cleanup_db_directory()

        print(f"\nCreating ChromaDB and adding {len(documents)} documents...")
        print("(This will compute embeddings if not already present)")

        try:
            self.db = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            print(f"ChromaDB created and documents added to '{self.persist_directory}'.")
            print("Embeddings are now stored on disk.")
        except Exception as e:
            print(f"Error creating and persisting ChromaDB: {e}")
            self.db = None
            raise

    def load_db(self):
        """
        Loads an existing ChromaDB from the persistence directory.
        """
        if self.embeddings is None:
            print("Embedding model not initialized. Cannot load DB.")
            return

        if not os.path.exists(self.persist_directory):
            print(f"ChromaDB directory not found at {self.persist_directory}. Cannot load.")
            self.db = None
            return

        print(f"\nLoading ChromaDB from disk at {self.persist_directory}...")
        try:
            # It's crucial to provide the same embedding_function when loading
            self.db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            print("ChromaDB loaded from disk successfully.")
        except Exception as e:
            print(f"Error loading ChromaDB from disk: {e}")
            self.db = None
            raise

    def get_retriever(self, search_kwargs: dict = None) -> VectorStoreRetriever:
        """
        Returns a LangChain VectorStoreRetriever instance from the loaded ChromaDB.

        Args:
            search_kwargs (dict, optional): Keyword arguments for the retriever's search method
                                            (e.g., {"k": 3} for top 3 results). Defaults to None.

        Returns:
            VectorStoreRetriever: A LangChain retriever instance.

        Raises:
            ValueError: If the ChromaDB has not been created or loaded yet.
        """
        if self.db is None:
            raise ValueError("ChromaDB has not been created or loaded. Call create_and_persist_db() or load_db() first.")

        if search_kwargs is None:
            search_kwargs = {"k": 3} # Default to top 3 results

        retriever = self.db.as_retriever(search_kwargs=search_kwargs)
        print(f"ChromaDB configured as a LangChain retriever with search_kwargs: {search_kwargs}")
        return retriever
