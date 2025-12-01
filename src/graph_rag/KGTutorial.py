import os
from dotenv import load_dotenv
load_dotenv()

import asyncio

from neo4j import GraphDatabase
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

from neo4j_graphrag.generation.prompts import ERExtractionTemplate

neo4j_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)
neo4j_driver.verify_connectivity()

# When testing more files gpt-4o-mini is better
llm = OpenAILLM(
    model_name="gpt-4o",
    model_params={
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
)

embedder = OpenAIEmbeddings(
    model="text-embedding-ada-002"
)

#Example propmpt
domain_instructions = (
    "Only extract entities that are related to the medical domain."
    "These include diseases, symptoms, conditions, treatments, medications, and medical procedures."
    "\n"
)

prompt_template = ERExtractionTemplate(
    template = domain_instructions + ERExtractionTemplate.DEFAULT_TEMPLATE 
)

kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=neo4j_driver, 
    neo4j_database=os.getenv("NEO4J_DATABASE"), 
    embedder=embedder, 
    from_pdf=True,
    prompt_template=prompt_template,
)

# Modify if you need to process multiple files
pdf_file = "test_data/Test_file.pdf"
result = asyncio.run(kg_builder.run_async(file_path=pdf_file))
print(result.result)