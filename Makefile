# Makefile for Agentic RAG Cardiology Pipeline
.PHONY: help setup install test run start stop clean check-deps check-models check-vectorstore status dev

# Default target
help:
	@echo "🤖 Agentic RAG Cardiology Pipeline"
	@echo ""
	@echo "Available commands:"
	@echo "  install       - Install Python dependencies"
	@echo "  test          - Run all agent tests"
	@echo "  test-router   - Test router agent only"
	@echo "  test-selfrag  - Test self-RAG agent only"
	@echo "  test-conv     - Test conversational agent only"
	@echo "  test-memory   - Test memory manager only"
	@echo "  run           - Run the main orchestrator"
	@echo "  status        - Show system status"
	@echo "  clean         - Clean up cache and temporary files"

install:
	@echo "📦 Installing Python dependencies..."
	@pip install -r requirements.txt
	@echo "✅ Dependencies installed"

# Testing
test: test-router test-selfrag test-conv test-memory
	@echo "✅ All tests completed"

test-router:
	@echo "🎯 Testing Router Agent..."
	@cd agent && python3 router.py

test-selfrag:
	@echo "🔍 Testing Self-RAG Agent..."
	@cd agent && python3 self_rag.py

test-conv:
	@echo "💬 Testing Conversational Agent..."
	@cd agent && python3 conversational_agent.py

test-memory:
	@echo "🧠 Testing Memory Manager..."
	@cd agent && python3 memory.py

# Running the system
run: check-deps
	@echo "🚀 Starting Agentic RAG Pipeline..."
	@cd agent && python3 orchestrator.py

# Status and monitoring
status:
	@echo "📊 System Status"
	@echo "================"
	@echo ""
	@echo "🐍 Python Dependencies:"
	@python3 -c "import sys; print(f'Python: {sys.version}')"
	@pip list | grep -E "(langchain|ollama|qdrant)" || echo "Some packages may be missing"
	@echo ""
	@echo "🤖 Ollama Models:"
	@if command -v ollama >/dev/null 2>&1; then \
		ollama list | grep -E "(llama3\.|Name)" || echo "No models found"; \
	else \
		echo "Ollama not installed"; \
	fi
	@echo ""
	@echo "🗄️  Qdrant Status:"
	@if curl -f http://localhost:6333/healthz >/dev/null 2>&1; then \
		echo "✅ Qdrant running"; \
		echo "Collections:"; \
		curl -s http://localhost:6333/collections | python3 -m json.tool 2>/dev/null || echo "Could not parse collections"; \
	else \
		echo "❌ Qdrant not running"; \
	fi

# Cleanup
clean:
	@echo "🧹 Cleaning up..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .coverage htmlcov/ 2>/dev/null || true
	@echo "✅ Cleanup complete"

install-ollama:
	@echo "🤖 Installing Ollama models..."
	@if command -v ollama >/dev/null 2>&1; then \
		echo "Installing required models..."; \
		ollama pull llama3.2:1b; \
		ollama pull llama3.1:latest; \
		echo "✅ Models installed"; \
	else \
		echo "❌ Ollama not found. Please install Ollama first:"; \
		echo "Visit: https://ollama.ai/download"; \
	fi

# Quick start for new users
quickstart:
	@echo "🚀 Quick Start Guide"
	@echo "===================="
	@echo ""
	@echo "1. Setting up the system..."
	@make install
	@echo ""
	@echo "2. Running tests..."
	@make test
	@echo ""
	@echo "3. Starting the pipeline..."
	@make run
