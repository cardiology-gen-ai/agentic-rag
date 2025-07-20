# Agentic RAG Pipeline for Cardiology Protocols

An intelligent multi-agent system for querying European Society of Cardiology (ESC) guidelines using advanced RAG (Retrieval-Augmented Generation) techniques with self-reflection and conversational capabilities.

![Cardiology Agent Architecture](images/cardiology-agent-transparent.drawio.png)

## 🏥 Overview

This project implements a sophisticated agentic RAG system specifically designed for medical professionals and students working with cardiology protocols. The system intelligently routes queries, retrieves relevant ESC guidelines, and provides accurate, evidence-based responses while maintaining conversational context.

### Key Features

- **🎯 Intelligent Query Routing**: Automatically classifies queries as conversational or medical
- **🔍 Self-RAG Architecture**: Self-reflecting retrieval with quality assessment and regeneration
- **💬 Conversational Agent**: Handles greetings, farewells, and general conversation naturally
- **🧠 Memory Management**: Maintains conversation context while managing token limits
- **📚 ESC Guidelines Integration**: Direct access to European Society of Cardiology protocols
- **🔄 State Management**: Robust state tracking across the entire conversation flow
- **💾 PostgreSQL Integration**: Persistent conversation storage and state management

## 📁 Project Structure

```
agentic-rag/
├── Makefile                    # Build automation and shortcuts
├── README.md                   # This documentation
├── requirements.txt            # Python dependencies
├── docker-compose.yml          # PostgreSQL database setup
├── init.sql                   # Database initialization script
├── .env.example               # Environment configuration template
├── test_database.py           # Database integration tests
├── agent/                      # Core agent implementations
│   ├── router.py              # Query classification agent
│   ├── self_rag.py           # Self-reflecting RAG agent
│   ├── conversational_agent.py # Social interaction agent
│   ├── memory.py             # Memory management agent
│   ├── database_manager.py   # PostgreSQL database manager
│   ├── state.py              # Shared state definitions
│   ├── orchestrator.py       # Main system orchestrator
│   └── chainlit.md           # Chainlit configuration
└── images/                    # Architecture diagrams
    ├── cardiology-agent-dark.drawio.png
    ├── cardiology-agent-light.drawio.png
    └── cardiology-agent-transparent.drawio.png
```

## 🏗️ Architecture

The system consists of several specialized agents working together:

### Agent Details

- **Router Agent** (`router.py`): Classifies incoming queries using both LLM and rule-based approaches
- **Self-RAG Agent** (`self_rag.py`): Handles medical queries with retrieval, generation, and self-assessment
- **Conversational Agent** (`conversational_agent.py`): Manages social interactions and system information
- **Memory Manager** (`memory.py`): Maintains conversation context and manages token limits
- **State Manager** (`state.py`): Defines the shared state structure across all agents

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Ollama with required models
- Qdrant vector database (from data-etl pipeline)
- ESC protocols data (processed through data-etl)
- PostgreSQL database (for conversation persistence)
- Docker (optional, for easy PostgreSQL setup)

### Database Setup

The agent uses PostgreSQL for persistent conversation storage. You can set it up using Docker:

```bash
# Start PostgreSQL database
docker-compose up -d

# Verify the database is running
docker-compose ps
```

Or use an existing PostgreSQL instance and set the `DATABASE_URL` environment variable:

```bash
# Copy the environment template
cp .env.example .env

# Edit .env with your database connection string
export DATABASE_URL="postgresql://username:password@localhost:5432/database"
```

### One-Command Setup

```bash
# Clone and setup everything
git clone https://github.com/your-org/agentic-rag.git
cd agentic-rag

# Start database (if using Docker)
docker-compose up -d

# Install dependencies and run tests
make install
python test_database.py  # Test database integration

# Run the agent
make run
```

This will:
1. Start PostgreSQL database
2. Install all dependencies
3. Test database connectivity
4. Run the agent in interactive mode

**Important**: remember to start the vectorstore separately for better control

## 🔍 Agent Functionality

### Router Agent

**Purpose**: Intelligent query classification
- **LLM Classification**: Uses Llama models for semantic understanding
- **Rule-based Fallback**: Medical keyword detection and pattern matching
- **Categories**: `conversational` vs `document_based`

```python
# Example usage
router = Router()
query_type = router.classify_query("What is the protocol for heart failure?")
# Returns: "document_based"
```

### Self-RAG Agent

**Purpose**: Medical query processing with self-reflection
- **Retrieval Loop**: Multiple attempts with query reformulation
- **Generation Loop**: Self-assessment and regeneration
- **Quality Gates**: Relevance, hallucination, and adequacy checking

```python
# Example usage
self_rag = SelfRAG(vectorstore)
state = {"messages": [HumanMessage(content="ESC heart failure guidelines?")]}
result = self_rag.selfRAG_node(state)
```

### Conversational Agent

**Purpose**: Natural conversation handling
- **Template Responses**: Pre-defined responses for common patterns
- **Contextual Generation**: LLM-based responses for complex conversations
- **Edge Case Handling**: Graceful handling of unclear or short inputs

```python
# Example usage
conv_agent = ConversationalAgent()
state = {"messages": [HumanMessage(content="Hello!")]}
result = conv_agent.conversational_agent_node(state)
```

### Memory Manager

**Purpose**: Context preservation and token management
- **Medical Entity Extraction**: Preserves important clinical information
- **Conversation Summarization**: LLM-based summarization of longer conversations
- **Token Estimation**: Automatic cleanup when approaching limits
- **Database Integration**: Automatic saving and restoration of conversation state

```python
# Example usage
memory = Memory(max_tokens=2000, session_id="user_123")
updated_state = memory.memory_management_node(state)
```

### Database Manager

**Purpose**: Persistent conversation storage and state management
- **Conversation Persistence**: Saves messages, summaries, and medical context
- **Session Management**: Tracks conversations across multiple interactions
- **Agent State Storage**: Persists internal agent states for continuity
- **Async & Sync Support**: Both synchronous and asynchronous database operations

```python
# Example usage
from database_manager import DatabaseManager, ConversationState

# Async usage
db_manager = DatabaseManager()
await db_manager.initialize()
conversation_id = await db_manager.save_conversation_state(conversation_state)

# Sync usage
sync_db = SyncDatabaseManager()
sync_db.initialize()
state_id = sync_db.save_simple_state(session_id, state_data)
```

## 🎯 State Management

The system uses a centralized state object that flows between agents:

```python
class State(TypedDict):
    # Core conversation
    messages: List[BaseMessage]
    
    # Routing information  
    query_type: Optional[str]
    
    # Response generation
    response: Optional[str]
    
    # Memory management
    context: Optional[Dict[str, Any]]
    conversation_summary: Optional[str]
    
    # RAG-specific
    documents: Optional[List[Document]]
    
    # Metadata and tracking
    metadata: Optional[Dict[str, Any]]
    retrieval_attempts: Optional[int]
    generation_attempts: Optional[int]
```

## 🧪 Testing

Each agent includes comprehensive testing capabilities:

```bash
# Run all tests
make test

# Test individual components
make test-router      # Router classification tests
make test-selfrag     # Self-RAG pipeline tests  
make test-conv        # Conversation tests
make test-memory      # Memory management tests
```

**System Health Check**
```bash
# Complete system status
make status

# Clean up if needed
make clean
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

