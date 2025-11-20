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

        # Load the agent configuration 
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
                "test_tag": getattr(self, "test_tag", "default"),
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

    def _normalize_text(self, s: str) -> str:
        """
        Helper to normalize section/header strings:
        - lowercase
        - remove common punctuation
        - collapse whitespace
        """
        s = s.lower()
        s = re.sub(r"[/#*()]", " ", s)
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    def evaluate_retrieval(self, top_k: int = 5, fuzzy_threshold: int = 85):
        """
        Section-level evaluation.

        For each question:
        - expected_sections = list of section titles from the dataset
        - retrieved_chunks = top_k retrieved documents

        A retrieved chunk is considered CORRECT if any of its headers
        fuzzy-matches any expected section above fuzzy_threshold.

        Metrics:
        - accuracy: fraction of questions with at least one correct chunk in top_k
        - precision@k: average over questions of (correct_chunks / retrieved_chunks)
        - recall@k: average over questions of
                    (#expected_sections_hit / #expected_sections)
        - f1@k: harmonic mean of avg precision and avg recall
        """
        self.logger.info("Evaluating retrieval results...")

        total_questions = 0
        questions_with_hit = 0
        per_q_precisions = []
        per_q_recalls = []

        for result in self.results["retrieval_results"]:

            # Extract and normalize expected section labels
            raw_expected_sections = result.get("sections", []) or []
            expected_sections = [
                self._normalize_text(s) 
                for s in raw_expected_sections 
                if isinstance(s, str) and s.strip()
            ]

            # If the question has no valid expected sections → SKIP from evaluation
            if not expected_sections:
                self.logger.warning(
                    f"No expected sections for question id={result.get('question_id')}, "
                    "skipping from evaluation."
                )
                continue

            # Count only evaluated questions
            total_questions += 1

            retrieved_chunks = result.get("retrieved_documents", [])[:top_k]

            correct_chunk_count = 0
            hit_any_section = False
            matched_sections = set()  # track which expected sections were hit

            # Evaluate each retrieved chunk
            for chunk in retrieved_chunks:
                # Extract headers robustly
                if isinstance(chunk, dict):
                    headers = chunk.get("headers", {})
                else:
                    headers = getattr(chunk, "metadata", {}).get("headers", {})

                # Convert headers into a list of strings
                header_values = []
                if isinstance(headers, dict):
                    header_values = list(headers.values())
                elif isinstance(headers, (list, tuple)):
                    header_values = list(headers)
                elif isinstance(headers, str):
                    header_values = [headers]

                # Normalize header texts
                header_values = [
                    h for h in header_values 
                    if isinstance(h, str) and h.strip()
                ]
                header_norms = [self._normalize_text(h) for h in header_values]

                # Fuzzy match
                chunk_hits_section = False
                for h_norm in header_norms:
                    for sec_norm in expected_sections:
                        score = fuzz.partial_ratio(h_norm, sec_norm)
                        if score >= fuzzy_threshold:
                            chunk_hits_section = True
                            hit_any_section = True
                            matched_sections.add(sec_norm)
                            break
                    if chunk_hits_section:
                        break

                if chunk_hits_section:
                    correct_chunk_count += 1

            # Accuracy
            if hit_any_section:
                questions_with_hit += 1

            # Per-question precision
            q_precision = (
                correct_chunk_count / len(retrieved_chunks)
                if retrieved_chunks else 0.0
            )

            # Per-question recall
            q_recall = (
                len(matched_sections) / len(expected_sections)
                if expected_sections else 0.0
            )
            q_recall = min(q_recall, 1.0)

            per_q_precisions.append(q_precision)
            per_q_recalls.append(q_recall)

        # Aggregate metrics
        if total_questions == 0:
            self.logger.warning("No questions evaluated (total_questions == 0).")
            return {}

        accuracy = questions_with_hit / total_questions
        avg_precision = sum(per_q_precisions) / len(per_q_precisions)
        avg_recall = sum(per_q_recalls) / len(per_q_recalls)
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
        test_tag = getattr(self, "test_tag", "test")
        output_file = self.results_dir / f"{safe_model_name}_{test_tag}_{timestamp}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Results saved to {output_file}")
        return output_file


if __name__ == "__main__":
# Logging Setup
    log_dir = pathlib.Path("logs")
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    log_file = log_dir / f"retriever_test_{timestamp}.log"

    # Configure console logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create file handler 
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S"
    ))

    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    print(f"[LOG] Writing detailed logs to: {log_file}")

    test_files = {
    "en":  "src/agentic_rag/retriever_tests/test_questions.json",
    "it":  "src/agentic_rag/retriever_tests/translated_questions.json",
    }

    for test_tag, test_file in test_files.items():
        logging.info(f"\n=== Evaluating test file ({test_tag}): {test_file} ===")

        tester = RetrieverTester(
            test_file=test_file,
            app_id="cardiology_protocols",
            results_dir="logs/",
        )
        tester.test_tag = test_tag 
        tester.load_questions()
        tester.run_tests(top_k=5)
        tester.evaluate_retrieval(top_k=5, fuzzy_threshold=90)
        tester.save_results()




