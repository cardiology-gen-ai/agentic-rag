## How to run 

> [!IMPORTANT]
> To run the code the module of the repo cardiology-gen-ai MUST be installed e.g. via
> ```
> uv pip install -e ../cardiology-gen-ai
> ```
> (change `../cardiology-gen-ai` with the relative path to the repo in your machine).

> Other dependencies can be installed using `uv pip install .` and/or `uv pip install .[other]`

To run the interactive session with the agent do the following:

1. Activate the virtual environment
```
source .venv/bin/activate
```
 
2. Set the INDEX_ROOT environmental variable in the root directory from which you run the script (e.g. from agentic-rag)
```
export INDEX_ROOT="abs/path/to/agentic-rag"
```

3. Run the python script as a module
```
uv run python -m src.main
```

---

## System Architecture Overview

The project implements a three-phase pipeline for cardiology guideline processing and querying:

1. **Data ETL Pipeline** - Converts ESC cardiology protocols from PDF to searchable vectors
2. **Agentic RAG System** - Multi-agent framework for intelligent query processing
3. **User Interface** - Conversational web interface built with Chainlit

### Technical Stack:
- **Vectorstore** -> Qdrant / FAISS 
- **Database** -> PostgreSQL with SQLAlchemy
- **AI/ML Framework** -> LangChain/LangGraph for agent orchestration
- **Embedding Models** -> Sentence Transformers (all-MiniLM-L6-v2)
- **LLM** -> Ollama (llama3.2:1b) with local deployment
- **User Interface** -> Chainlit 2.6.5 with authentication
- **Containerization** -> Docker & Docker Compose
- **Additional Tools** -> PyMuPDF4LLM, HuggingFace Transformers

---

## data-etl (1st phase) 

The first phase implements an ETL pipeline for processing [ESC cardiology guidelines](https://www.escardio.org/Guidelines). The PDF files are stored in [Google Drive](https://drive.google.com/drive/folders/1rgaemZ4Jetyz98ivTw8fpLIndgZ2jczn?usp=sharing) for storage efficiency.

### Key Technical Decisions:

#### PDF to Markdown Conversion
- **Tool Selection**: [PyMuPDF4LLM](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/) with custom modifications for enhanced image extraction
- **Custom Image Algorithm**: Implemented rectangle merging algorithm with:
  - 200 DPI extraction for high-quality images
  - Tolerance-based rectangle merging (40.0 threshold)
  - Intelligent image positioning based on caption detection
  - Systematic naming: `FIG_[page]_[index].png`
- **Post-processing**: Unicode normalization, hyphenation fixing, and pattern cleanup

#### Document Chunking Strategy
- **Multi-stage Approach**: Hierarchical chunking preserving document structure
  - Markdown header splitting (4 levels)
  - Recursive character splitting (1000 tokens with 200 overlap)
  - Token-aware splitting with HuggingFace tokenizers
- **Semantic Preservation**: Maintains clinical context through header-aware segmentation

#### Vector Database Implementation
- **Dual Backend Support**:
  - **Qdrant** (production): Docker-deployed with dense + sparse vector support
  - **FAISS** (development): Local filesystem storage with cosine similarity
- **Embedding Choice**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
  - Rationale: Optimal balance of performance, speed, and medical text understanding
- **Storage Architecture**: Persistent Docker volumes with health monitoring

---

## agentic-rag (2nd phase)

The second phase implements a sophisticated agentic RAG architecture using LangGraph for state management and agent orchestration. The system evolved through multiple iterations to achieve the final optimized workflow.

![](./images/graph.png)

### Agent Graph Architecture

The system implements a **self-correcting multi-agent workflow** with the following specialized components:

#### Core Agent Nodes:
- **`contextualize_question`** -> Adds conversational context to user questions using chat history
- **`document_request_detector`** -> Intelligent routing between conversational and document-based queries
- **`retrieve`** -> Semantic search through cardiology protocols with configurable k-retrieval (default: 3 documents)
- **`retrieval_grader`** -> Quality assessment of retrieved documents using LLM-based relevance scoring
- **`generate`** -> Answer generation from relevant documents with grounding validation
- **`transform_query`** -> Query reformulation when initial retrieval fails relevance thresholds
- **`conversational_agent`** -> Handles greetings, social interaction, and non-medical queries

### Technical Architecture Decisions:

#### LLM Configuration:
- **Primary Model**: Ollama llama3.2:1b for local deployment and privacy
- **Role-specific Temperature Settings**:
  - Router: 0.5 (balanced creativity for classification)
  - Generator: 0.01 (high consistency for medical responses)
  - Grader: 0.2 (focused evaluation for quality assessment)

#### State Management:
- **LangGraph State Machine**: TypedDict-based state with comprehensive tracking:
  - Message history, document context, retrieval metadata
  - Generation attempts, validation scores, routing decisions
- **Memory Architecture**: Dual-layer memory system:
  - **Short-term**: LangGraph checkpointers with in-memory state
  - **Long-term**: PostgreSQL persistence for conversation history
  - **Medical Entity Extraction**: Preserves clinical context across sessions

#### Framework Rationale:
- **LangChain/LangGraph**: Selected for production-grade agent orchestration and state management
- **HuggingFace Integration**: Embedding models and transformer architecture for flexibility
- **Modular Design**: Pluggable components for different LLMs, vectorstores, and embedding models

For development a command-line demo has been developed, with further Chainlit integration for a more fluid interaction with a GUI. Also in-memory checkpointers have been used at this point. For **production** instead, PostgreSQL backend has been used, with async/sync dual interfaces for scalability. Chainlit here will be the main graphical interface. 

---

## user-interface (3rd phase) 

The third phase implements a production-ready conversational web interface using [Chainlit 2.6.5](https://docs.chainlit.io/get-started/overview), selected for its specialized conversational AI capabilities and seamless integration with the existing agentic RAG system.

### Architecture Integration Decisions:

Chainlit allows for different benefits:
- **Authentication and user management** -> using PostgreSQL as backend and `CHAINLIT_AUTH_SECRET` for session encryption. Users are persistent and this along with authentication allows for chat history and human feedback for later processing
- **Fluid testing and experience** -> rich message types (text, images, files) and real-time updates
- **Scalability** -> ready for production with async support and multi-user handling
- **Monitoring** -> easy tracking and analytics integration using LangSmith or LangFuse