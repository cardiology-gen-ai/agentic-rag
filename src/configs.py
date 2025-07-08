
###################
##### General #####
###################

# Debugging
DEBUG = True
LLM_MODEL = "llama3.2:1b"

###################
####### RAG #######
###################

# LLM settings
LLM_TEMPERATURE = 0.7
LLM_GENERATION_TEMPERATURE = 0.7
LLM_CLASSIFICATION_TEMPERATURE = 0.0

# Retrieval Configuration
RETRIEVAL_K = 5
SEARCH_TYPE = 'similarity'
RETRIEVAL_SCORE_THRESHOLD = 0.5
MAX_RETRIEVAL_ATTEMPTS = 2
MAX_GENERATION_ATTEMPTS = 2

# Context in generation
MAX_CONTEXT_LENGTH = 1000
MAX_DOCS_TO_USE = 3

####################
###### Router ######
####################

ROUTER_LLM_TEMPERATURE = 0.0

##################
##### Memory #####
##################

MEMORY_LLM_TEMPERATURE = 0.1

##############################
##### Conversational LLM #####
##############################

CONVERSATIONAL_LLM_TEMPERATURE = 0.7

########################
##### Orchestrator #####
########################

# Langfuse Configuration
LANGFUSE_PUBLIC_KEY = "pk-lf-2e9fb95c-ae8e-4d0c-b102-e878d6b6738f"
LANGFUSE_SECRET_KEY = "sk-lf-81521946-8bfc-4822-b518-09031931753c"
LANGFUSE_HOST = "https://cloud.langfuse.com"

# Vectorstore Configuration
VECTORSTORE_TYPE = "qdrant"
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION_NAME = "cardio_protocols"
FAISS_INDEX_PATH = "./faiss_index"
