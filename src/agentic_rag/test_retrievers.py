import json
import re
import logging
import pathlib
from datetime import datetime
from fuzzywuzzy import fuzz
from dotenv import load_dotenv

# Load environment
dotenv_path = pathlib.Path(__file__).resolve().parents[2] / ".env"
print(f"Loading .env from: {dotenv_path}")
load_dotenv(dotenv_path=dotenv_path)

from agentic_rag.managers.search_manager import SearchManager
from agentic_rag.config.manager import AgentConfigManager


class RetrieverTester:
    def __init__(self, test_file: str, app_id: str = "cardiology_protocols", results_dir: str = "src/agentic_rag/retriever_tests"):
        self.test_file = pathlib.Path(test_file)
        self.results_dir = pathlib.Path(results_dir)
        self.logger = logging.getLogger("RetrieverTester")

        # Load the agent configuration (same one used by your RAG system)
        config = AgentConfigManager(app_id=app_id).config

        # Initialize the unified search manager
        self.search_manager = SearchManager(
            index_config=config.indexing,
            search_config=config.search,
            embeddings=config.embeddings
        )
        self.model_name = getattr(config.embeddings, "deployment", None) or getattr(config.embeddings, "model_name", "unknown")
        self.model_name = self.model_name.split("/")[-1]

    def load_questions(self):
        with open(self.test_file, "r", encoding="utf-8") as f:
            self.test_data = json.load(f)
        self.logger.info(f"Loaded {self.test_data['metadata']['total_questions']} questions.")

    def run_tests(self, top_k: int = 5):
        self.logger.info(f"Running retrieval tests with top_k={top_k}...")
        self.results = {
            "metadata": {
                "embedding_model": self.model_name,
                "index_name": self.search_manager.index_config.name,
                "test_date": datetime.now().isoformat(),
                "top_k": top_k,
            },
            "retrieval_results": []
        }

        for guideline_key, guideline_data in self.test_data["guidelines"].items():
            self.logger.info(f"\nTesting guideline: {guideline_data['guideline_name']}")
            for q in guideline_data["questions"]:
                docs = self.search_manager.search(query=q["question"]) or []
                docs = docs[:top_k]  # limit to top_k

                # Serialize retrieved documents
                serialized_docs = []
                for d in docs:
                    if hasattr(d, "page_content"):
                        md = dict(getattr(d, "metadata", {}) or {})
                        serialized_docs.append({
                            "page_content": d.page_content,
                            "metadata": md,
                            "headers": md.get("headers", {})  # flatten headers for easier eval
                        })
                    elif isinstance(d, dict):
                        serialized_docs.append(d)
                    else:
                        self.logger.warning(f"Unknown doc type: {type(d)}")

                self.results["retrieval_results"].append({
                    "guideline": guideline_key,
                    "guideline_name": guideline_data["guideline_name"],
                    "question_id": q["id"],
                    "question": q["question"],
                    "sections": q.get("sections", []),
                    "retrieved_documents": serialized_docs
                })

    def evaluate_retrieval(self, top_k: int = 5):
        self.logger.info("Evaluating retrieval results...")
        total_questions = 0
        correct_hits = 0
        precisions, recalls = [], []

        for result in self.results["retrieval_results"]:
            total_questions += 1
            expected_sections = [
                re.sub(r"\s+", " ", re.sub(r"[/#*()]", "", s.lower())).strip()
                for s in result.get("sections", [])
            ]

            retrieved_chunks = result["retrieved_documents"][:top_k]
            hit = 0

            for chunk in retrieved_chunks:
                # Safe extraction of headers
                if isinstance(chunk, dict):
                    headers = chunk.get("headers", {})
                else:
                    headers = getattr(chunk, "metadata", {}).get("headers", {})

                processed_headers = [
                    re.sub(r"\s+", " ", re.sub(r"[/#*()]", "", h.lower())).strip()
                    for h in headers.values()
                ]
                header_text = " ".join(processed_headers) if headers else ""
               

                # Fuzzy matching

                for h in processed_headers:
                    if any(fuzz.partial_ratio(h, sec) > 85 for sec in expected_sections):
                        hit += 1
                        break

            if hit > 0:
                correct_hits += 1

            precision = hit / len(retrieved_chunks) if retrieved_chunks else 0
            recall = hit / len(expected_sections) if expected_sections else 0
            precisions.append(precision)
            recalls.append(min(recall, 1.0))

        accuracy = correct_hits / total_questions if total_questions else 0
        avg_precision = sum(precisions) / len(precisions) if precisions else 0
        avg_recall = sum(recalls) / len(recalls) if recalls else 0
        f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall + 1e-9)

        metrics = {
            "accuracy": round(accuracy, 3),
            f"precision@{top_k}": round(avg_precision, 3),
            f"recall@{top_k}": round(avg_recall, 3),
            f"f1@{top_k}": round(f1, 3),
        }

        self.results["metrics"] = metrics
        self.logger.info("\n Evaluation Metrics:")
        for k, v in metrics.items():
            self.logger.info(f"{k:15s}: {v:.3f}")
        return metrics

    def save_results(self):
        self.results_dir.mkdir(parents=True, exist_ok=True)
        safe_model_name = re.sub(r'[^a-zA-Z0-9._]', '_', self.model_name)
        timestamp = datetime.now().strftime("%H_%M_%S__%d_%m_%Y")
        output_file = self.results_dir / f"{safe_model_name}_{timestamp}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Results saved to {output_file}")
        return output_file


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    tester = RetrieverTester(
        test_file="src/agentic_rag/retriever_tests/subset_test_questions.json",
        app_id="cardiology_protocols",
        results_dir="src/agentic_rag/retriever_tests/",
    )
    tester.load_questions()
    tester.run_tests(top_k=5)
    tester.evaluate_retrieval(top_k=5)
    tester.save_results()
