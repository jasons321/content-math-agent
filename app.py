import json
import os
from pathlib import Path
import streamlit as st
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

st.set_page_config(
    page_title="Computational Math 24h Study Agent", layout="wide"
)

st.title("🧠 24-Hour Exam & Coursework Revision Agent")
st.write(
    "Point this app to your folder containing your `.ipynb` and `.md` lecture"
    " files."
)

# Sidebar configuration for local folder path
st.sidebar.header("1. Lecture Source")
folder_path = st.sidebar.text_input(
    "Path to your lectures folder:", "./lectures"
)


# Safe custom loaders to handle heavy math, LaTeX, and special characters without crashing
def load_markdown_file(file_path):
  try:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
      text = f.read()
    return [
        Document(
            page_content=text, metadata={"source": str(file_path)}
        )
    ]
  except Exception as e:
    st.warning(f"Skipping markdown file {file_path.name} due to error: {e}")
    return []


def load_notebook_file(file_path):
  try:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
      nb = json.load(f)

    content_blocks = []
    for cell in nb.get("cells", []):
      cell_type = cell.get("cell_type")
      source = cell.get("source", "")
      if isinstance(source, list):
        source = "".join(source)

      if cell_type == "markdown":
        content_blocks.append(f"### [Markdown Note]\n{source}")
      elif cell_type == "code":
        content_blocks.append(f"### [Python Code Cell]\n```python\n{source}\n```")

    full_text = "\n\n".join(content_blocks)
    return [
        Document(
            page_content=full_text, metadata={"source": str(file_path)}
        )
    ]
  except Exception as e:
    st.warning(f"Skipping notebook {file_path.name} due to error: {e}")
    return []


# Load and index button
if st.sidebar.button("Index / Reload Lectures"):
  if not os.path.exists(folder_path):
    st.error(
        f"Directory '{folder_path}' does not exist. Please check the path."
    )
  else:
    with st.spinner(
        "Parsing lecture files safely (ignoring encoding/math glitches)..."
    ):
      docs = []
      path = Path(folder_path)

      # 1. Load Markdown files recursively
      for md_file in path.rglob("*.md"):
        docs.extend(load_markdown_file(md_file))

      # 2. Load Jupyter Notebook files recursively
      for ipynb_file in path.rglob("*.ipynb"):
        docs.extend(load_notebook_file(ipynb_file))

      if not docs:
        st.warning(
            "No valid .md or .ipynb files found in the specified directory."
        )
      else:
        # 3. Split content cleanly into bite-sized chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200, chunk_overlap=200
        )
        split_docs = text_splitter.split_documents(docs)

        # 4. Create Vector Store using Ollama Embeddings
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vectorstore = Chroma.from_documents(split_docs, embeddings)

        # Save retriever in session state
        st.session_state.retriever = vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )
        st.success(
            f"Successfully loaded and indexed {len(split_docs)} clean chunks"
            " from your lectures!"
        )

# Main Chat Interface
st.header("2. Ask Your Course Material")

if "retriever" not in st.session_state:
  st.info(
      "👈 Please enter your folder path and click 'Index / Reload Lectures' in"
      " the sidebar to start."
  )
else:
  # Initialize Ollama model with an expanded context window
  llm = OllamaLLM(model="qwen2.5:7b", options={"num_ctx": 8192})

  query = st.text_input(
      "Ask a question about your math notes, derivations, or code implementation:"
  )

  if query:
    with st.spinner("Searching notes and generating explanation..."):
      retrieved_docs = st.session_state.retriever.invoke(query)
      context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

      prompt_template = ChatPromptTemplate.from_messages([
          (
              "system",
              "You are an expert computational mathematics tutor. Answer the"
              " user's question using ONLY the provided context from their"
              " lecture notes and code blocks. Maintain all mathematical"
              " formulas and equations accurately. If you don't know, say you"
              " cannot find it in the notes.\n\nContext:\n{context}",
          ),
          ("user", "{input}"),
      ])

      prompt = prompt_template.invoke(
          {"context": context_text, "input": query}
      )
      response = llm.invoke(prompt)

      st.markdown("### Answer:")
      st.write(response)

      with st.expander("View Source Notes & References"):
        for doc in retrieved_docs:
          source_name = doc.metadata.get("source", "Unknown file")
          st.markdown(f"**Source: `{source_name}`**")
          st.text(doc.page_content[:400] + "...")
          st.divider()