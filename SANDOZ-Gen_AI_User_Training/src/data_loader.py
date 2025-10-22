from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from tqdm import tqdm


def load_pdf_documents(data_dir: str) -> List[Any]:
    """
    Load all pdf files from the data directory and convert to LangChain document structure.
    Shows a tqdm progress bar instead of printing each file path.
    """
    # Use project root data folder
    data_path = Path(data_dir).resolve()
    documents = []

    # Filter for only PDF files
    pdf_files = list(data_path.glob('**/*.pdf'))
    for pdf_file in tqdm(pdf_files, desc="Loading PDFs", unit="file"):
        try:
            loader = PyPDFLoader(str(pdf_file))
            loaded = loader.load()  # Convert loaded pdf document into a langchain documents list (one per page)
            documents.extend(loaded)  # Add loaded docs into the main list (concatenate langchain objects/docs)
        except Exception as e:
            tqdm.write(f"[ERROR] Failed to load PDF {pdf_file}: {e}")

    return documents