from agentic_rag.agent import output

node_output_schemas = {
    "question_contextualizer": None,
    "conversational_agent": None,
    "error_handler": None,
    "generator": None,
    "default_response_generator": None,
    "document_response_generator": None,
    "question_rewriter": None,
    "language_detector": output.DetectedLanguage,
    "document_request_detector": output.DocumentRequest,
    "router": output.RouteQuery,
    "answer_grader": output.GradeAnswer,
    "groundedness_grader": output.GradeGrounding,
    "retrieval_grader": output.GradeDocuments,
    "multi_query_generator": output.MultipleQueries,
    "question_ambiguity_detector": output.QueryAmbiguity,
    "question_clarifier": output.MultipleQueries,
}
