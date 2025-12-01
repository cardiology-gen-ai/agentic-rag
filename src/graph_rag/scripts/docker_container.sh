docker run -d \
  --name neo4j-local \
  -p 7474:7474 -p 7687:7687 \
  -v $PWD/neo4j_data:/data \
  -e NEO4J_AUTH=neo4j/pippo123 \ 
  -e NEO4J_PLUGINS='["graph-data-science", "apoc"]' \
  -e NEO4J_dbms_security_procedures_unrestricted=apoc.* \
  neo4j:latest

# Choose password (ex. pippo123) and account name (ex. neo4j), these will then be used to access neo4j in docker
# Also load the chosen name and password into the .env file 