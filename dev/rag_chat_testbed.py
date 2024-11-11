"""
RAG Chat System with OpenAI and Pinecone

This script implements a Retrieval-Augmented Generation (RAG) chat system using OpenAI's
embedding models and Pinecone's vector database. It processes user queries by generating
embeddings, retrieving relevant document chunks, handling multiple matches within the same
document, fetching contextual neighbors, merging content, and sending prompts to OpenAI's
completion endpoint to generate responses.

Key Components:
- VectorDatabaseHandler: Manages embeddings creation, querying Pinecone, and fetching items.
- ContextFetcher: Retrieves and merges neighboring document chunks based on matched results.
- QueryProcessor: Orchestrates the query processing workflow, including filtering, scoring, and generating responses.

Configuration:
- Requires OpenAI and Pinecone API keys set in a `.env` file.
- Configurable parameters include `score_threshold`, `neighbour_range`, and `gap_threshold`.

Usage:
1. Define a list of questions and namespaces.
2. Initialize VectorDatabaseHandler and QueryProcessor with desired settings.
3. Run the script to process queries, retrieve contexts, and generate responses using OpenAI's Completion API.

Notes:
- This code serves as a testbed to determine optimal settings for chunk size and overlap.
- It integrates sending merged contexts to OpenAI's Completion API for answer generation.
- Future enhancements can include better error handling, dynamic system prompt loading, and more sophisticated prompt engineering.

Ensure all dependencies are installed and the Pinecone index is properly set up before execution.
"""

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, List, Dict, Tuple

from pinecone import Pinecone, Index
from openai import OpenAI
from dotenv import load_dotenv

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# Load environment variables from .env file
load_dotenv()

# Constants and Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "wd-rag-dev"
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 1024

SYSTEM_PROMPT_FILE = "chat_system_prompt.txt"  # Path to your system prompt file
USER_PROMPT_FILE = "chat_user_prompt.txt"  # Path to your user prompt file

# Ensure all required environment variables are set
if not all([OPENAI_API_KEY, PINECONE_API_KEY]):
    raise ValueError(
        "Please set OPENAI_API_KEY and PINECONE_API_KEY in your .env file."
    )

# Initialize OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)


@dataclass
class QueryResult:
    score: float
    metadata: Dict[str, Any]


class VectorDatabaseHandler:
    def __init__(self, index: Index):
        self.index = index

    def create_embeddings(
        self, texts: List[str], model: str = EMBEDDING_MODEL
    ) -> List[List[float]]:
        response = client.embeddings.create(
            input=texts, model=model, dimensions=EMBEDDING_DIMENSIONS
        )
        embeddings = [record.embedding for record in response.data]
        logging.info(f"Generated embeddings for {len(embeddings)} texts.")
        return embeddings

    def query(
        self, embedding: List[float], namespace: str, top_k: int
    ) -> List[QueryResult]:
        response = self.index.query(
            vector=embedding,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True,
            include_values=False,
        )
        matches = response.get("matches", [])
        results = [
            QueryResult(score=match["score"], metadata=match["metadata"])
            for match in matches
        ]
        return results

    def fetch_items(self, ids: List[str], namespace: str) -> List[Dict[str, Any]]:
        response = self.index.fetch(ids=ids, namespace=namespace)
        items = response.get("vectors", {})
        return [item["metadata"] for item in items.values()]

    def list_ids_by_document(self, document_id: str, namespace: str) -> List[str]:
        """
        Efficiently list all IDs for a specific document using the prefix filter.
        Assumes IDs are in the format 'document_id#chunk_id'.
        """
        prefix = f"{document_id}#"
        all_ids = []
        try:
            # Pinecone's list method returns a generator of IDs matching the prefix
            ids_generator = self.index.list(namespace=namespace, prefix=prefix)

            # Determine if the generator yields individual IDs or lists of IDs
            for item in ids_generator:
                if isinstance(item, list):
                    # If a list is yielded, extend the all_ids list
                    all_ids.extend(item)
                else:
                    # If a single ID string is yielded, append it
                    all_ids.append(item)

            logging.debug(
                f"Found {len(all_ids)} chunks for document '{document_id}' in namespace '{namespace}'."
            )
        except Exception as e:
            logging.error(
                f"Error listing IDs for document '{document_id}' in namespace '{namespace}': {e}"
            )
        return all_ids


class ContextFetcher:
    def __init__(self, db_handler: VectorDatabaseHandler, overlap_size: int):
        self.db_handler = db_handler
        self.overlap_size = overlap_size

    def get_document_id_and_chunk_id(self, id_str: str) -> Tuple[str, int]:
        """Assumes ID format is 'document_id#chunk_id' where chunk_id is integer."""
        try:
            doc_id, chunk_id_str = id_str.split("#", 1)
            chunk_id = int(chunk_id_str)
        except ValueError:
            doc_id = id_str
            chunk_id = 0  # Default or handle differently if not integer
        return doc_id, chunk_id

    def merge_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """Merge chunks based on their order and overlap size."""
        if not chunks:
            return ""

        # Sort chunks by chunk_id
        chunks_sorted = sorted(
            chunks,
            key=lambda x: (
                int(x["id"].split("#")[1])
                if "#" in x["id"] and x["id"].split("#")[1].isdigit()
                else 0
            ),
        )

        merged_content = ""
        for chunk in chunks_sorted:
            text = chunk.get("text", "")

            if not merged_content:
                merged_content = text
            else:
                # overlap_size is the number of characters
                overlap_size = self.overlap_size
                max_deviation = (
                    self.overlap_size // 5
                )  # Allowable deviation from overlap_size

                # Define the range for possible overlap sizes
                start_overlap = max(overlap_size - max_deviation, 1)
                end_overlap = overlap_size + max_deviation

                # Find the maximum overlapping substring within the range
                overlap_length = 0
                for i in range(end_overlap, start_overlap - 1, -1):
                    check = merged_content[-i:]
                    if text.startswith(check):
                        overlap_length = i
                        break

                if overlap_length > 0:
                    # Append without the overlapping part
                    trimmed = text[overlap_length:]
                    merged_content += trimmed
                else:
                    # If no sufficient overlap, append with a delimiter
                    merged_content += f"...\n{'-'*32}\n" + text

        merged_content = merged_content.replace(
            "Previous NextDown",
            "\n",
        ).strip()  # Remove pagination link texts, trim whitespace

        return merged_content


class QueryProcessor:
    def __init__(
        self,
        db_handler: VectorDatabaseHandler,
        namespaces: List[str],
        score_threshold: float = 0.5,  # Default score threshold
        neighbour_range: int = 2,  # Number of chunks before and after to fetch as neighbours
        gap_threshold: int = 1,  # Maximum allowed gap between neighbour ranges to merge them
        top_k: int = 20,  # Default top_k for each Pinecone query
    ):
        self.db_handler = db_handler
        self.namespaces = namespaces
        self.score_threshold = score_threshold
        self.neighbour_range = neighbour_range
        self.gap_threshold = gap_threshold
        self.top_k = top_k

        # Load the system prompt
        self.system_prompt = self.load_text(
            os.path.join(os.getcwd(), "dev", SYSTEM_PROMPT_FILE)
        )

        self.user_prompt = self.load_text(
            os.path.join(os.getcwd(), "dev", USER_PROMPT_FILE)
        )

    def load_text(self, filepath: str) -> str:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"file not found: {filepath}")
        with open(filepath, "r", encoding="utf-8") as file:
            prompt = file.read()
        logging.info(f"Loaded text from {filepath}.")
        return prompt

    def process_queries(self, questions: List[str]):
        # Create embeddings for all questions
        embeddings = self.db_handler.create_embeddings(questions)

        outputs = {}

        # Iterate over each question and its embedding
        for namespace in self.namespaces:
            logging.info(f"\n--- Namespace: {namespace} ---")
            outputs[namespace] = []

            for question, embedding in zip(questions, embeddings):
                logging.info(f"\nProcessing Query: \"{question}\"\n{'='*50}")

                # Parse namespace to get overlap size
                match = re.match(r"web-\d+-(\d+)", namespace)
                overlap_size = int(match.group(1)) if match else 100  # Default overlap
                context_fetcher = ContextFetcher(self.db_handler, overlap_size)

                # Query Pinecone
                top_k = self.top_k  # Increase top_k for more matches
                results = self.db_handler.query(embedding, namespace, top_k)

                logging.info(f"Found {len(results)} matching chunks")

                # Preprocess results: filter by score and group by document_id
                filtered_results = [
                    result for result in results if result.score >= self.score_threshold
                ]

                logging.info(
                    f"Found {len(filtered_results)} matches after applying score threshold of {self.score_threshold}."
                )

                # Group results by document_id
                grouped_results: Dict[str, List[QueryResult]] = {}
                for result in filtered_results:
                    document_id, _ = context_fetcher.get_document_id_and_chunk_id(
                        result.metadata["id"]
                    )
                    if document_id not in grouped_results:
                        grouped_results[document_id] = []
                    grouped_results[document_id].append(result)

                logging.info(f"Processing {len(grouped_results)} unique documents.")

                all_contexts = []

                for document_id, doc_results in grouped_results.items():
                    logging.debug(f"\n--- Document ID: {document_id} ---")
                    # Collect all chunk_ids for matches in this document
                    matched_chunk_ids = [
                        context_fetcher.get_document_id_and_chunk_id(r.metadata["id"])[
                            1
                        ]
                        for r in doc_results
                    ]

                    # Define neighbour ranges for each matched chunk
                    neighbour_ranges = [
                        (
                            max(chunk_id - self.neighbour_range, 0),
                            chunk_id + self.neighbour_range,
                        )
                        for chunk_id in matched_chunk_ids
                    ]

                    logging.debug(
                        f"Initial neighbor ranges for document '{document_id}': {neighbour_ranges}"
                    )

                    # Merge overlapping neighbour ranges
                    merged_ranges = self.merge_neighbour_ranges(
                        neighbour_ranges, gap_threshold=self.gap_threshold
                    )

                    # Collect all neighbour chunk_ids to fetch
                    neighbour_ids_set = set()
                    all_ids = self.db_handler.list_ids_by_document(
                        document_id, namespace
                    )
                    all_ids_sorted = sorted(
                        all_ids,
                        key=lambda x: context_fetcher.get_document_id_and_chunk_id(x)[
                            1
                        ],
                    )
                    for start, end in merged_ranges:
                        # Fetch chunk_ids within the range
                        for chunk_id in range(start, end + 1):
                            # Find the corresponding chunk_id in all_ids_sorted
                            for id_str in all_ids_sorted:
                                _, cid = context_fetcher.get_document_id_and_chunk_id(
                                    id_str
                                )
                                if cid == chunk_id:
                                    neighbour_ids_set.add(id_str)
                                    break  # Move to next chunk_id

                    neighbour_ids = sorted(
                        neighbour_ids_set,
                        key=lambda x: context_fetcher.get_document_id_and_chunk_id(x)[
                            1
                        ],
                    )

                    logging.debug(f"Fetching neighbours: {neighbour_ids}")

                    # Fetch the neighbour chunks
                    fetched_chunks = self.db_handler.fetch_items(
                        neighbour_ids, namespace
                    )

                    # Merge content
                    merged_content = context_fetcher.merge_chunks(fetched_chunks)

                    # Aggregate scores: take the highest score among matches
                    highest_score = max(result.score for result in doc_results)
                    mean_score = sum(result.score for result in doc_results) / len(
                        doc_results
                    )
                    combined_score = (highest_score * 3 + mean_score) / 4.0
                    logging.debug(f"Highest Score: {highest_score:.4f}")
                    logging.debug(f"Mean Score: {mean_score:.4f}")
                    logging.debug(f"Combined Score: {combined_score:.4f}")

                    # Calculate weighted score based on the latest last_modified
                    last_modified = doc_results[0].metadata.get("lastModified", 0)
                    weighted_score = self.weight_score(
                        combined_score,  # Use a combined score that favours towards the highest scoring chunk but also considers the mean
                        last_modified,
                    )

                    # Display or process the merged content
                    logging.debug(f"[Weighted Score: {weighted_score:.4f}]")
                    logging.debug(
                        f"Last Modified: {self.format_timestamp(last_modified)}"
                    )

                    all_contexts.append(
                        {
                            "url": doc_results[0].metadata.get("source", ""),
                            "title": doc_results[0].metadata.get("title", ""),
                            "date": f"{self.format_timestamp(last_modified)}",
                            "relevance": f"{weighted_score:.4f}",
                            "content": merged_content,
                        }
                    )

                # sort all_contexts by relevance, descending
                all_contexts = sorted(
                    all_contexts, key=lambda x: float(x["relevance"]), reverse=True
                )

                #  Generate and send prompt to OpenAI's Completion API
                combined_context = json.dumps(all_contexts)

                response_text = self.generate_openai_response(
                    question, combined_context
                )

                outputs[namespace].append(
                    {
                        "question": question,
                        "response": response_text,
                        "contexts": all_contexts,
                    }
                )

                logging.info(f"Question:\n{question}\n{'='*40}")
                logging.info(f"Response:\n{response_text}\n{'='*40}")

                # For demonstration, just process one question and continue
                # break

            # save outputs as json to file
            with open(
                os.path.join(os.getcwd(), "dev", "outputs", f"{namespace}.json"), "w"
            ) as f:
                json.dump(outputs[namespace], f, indent=4)

            # break

    def generate_openai_response(self, question: str, context: str) -> str:
        """Generate a response from OpenAI's Completion API using the system prompt."""
        # Prepare the system prompt by replacing placeholders
        current_date_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        system_prompt = self.system_prompt.replace("{currentDate}", current_date_utc)
        user_prompt = self.user_prompt.format(question=question, context=context)

        # logging.info(f"Sending prompt to OpenAI: {system_prompt}")
        # logging.info(f"User prompt: {user_prompt}")

        try:
            # Send the prompt to OpenAI's Completion API
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                max_tokens=4096,
                temperature=0.0,
                frequency_penalty=0.5,
                presence_penalty=0.5,
            )
            # Extract and return the generated text
            generated_text = response.choices[0].message.content.strip()
            return generated_text

        except Exception as e:
            logging.error(f"Error generating response from OpenAI: {e}")
            return "I'm sorry, but I couldn't generate a response at this time."

    def merge_neighbour_ranges(
        self, ranges: List[Tuple[int, int]], gap_threshold: int = 1
    ) -> List[Tuple[int, int]]:
        """
        Merge overlapping or adjacent neighbour ranges.

        Args:
            ranges: List of (start, end) tuples.
            gap_threshold: Maximum allowed gap between ranges to consider merging.

        Returns:
            List of merged (start, end) tuples.
        """
        if not ranges:
            return []

        # Sort ranges by start
        sorted_ranges = sorted(ranges, key=lambda x: x[0])
        merged = [sorted_ranges[0]]

        for current in sorted_ranges[1:]:
            last = merged[-1]
            # If current range overlaps or is within the gap_threshold, merge them
            if current[0] <= last[1] + gap_threshold:
                merged[-1] = (last[0], max(last[1], current[1]))
                logging.debug(f"Merged range {last} with {current} to {merged[-1]}")
            else:
                merged.append(current)
                logging.debug(f"Added new range {current} to merged ranges")

        logging.debug(f"Merged neighbour ranges: {merged}")
        return merged

    @staticmethod
    def weight_score(score: float, last_modified: float) -> float:
        """Combine similarity score with recency. Newer content gets higher weight."""
        # Normalize last_modified to a score between 0 and 1
        current_time_ms = datetime.now(timezone.utc).timestamp() * 1000
        age_ms = current_time_ms - last_modified

        # Assuming that content older than 3 years gets the lowest weight
        max_age = 3 * 365 * 24 * 60 * 60 * 1000
        recency_score = max(0, 1 - age_ms / max_age)

        # Weight factors can be adjusted
        similarity_weight = 0.6
        recency_weight = 0.4

        return similarity_weight * score + recency_weight * recency_score

    @staticmethod
    def format_timestamp(timestamp_ms: float) -> str:
        """Convert UTC timestamp in milliseconds to readable format."""
        try:
            dt = datetime.fromtimestamp(timestamp_ms / 1000, timezone.utc)
            return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        except Exception:
            return "Unknown"


def main():

    question_dataset = json.load(
        open(os.path.join(os.getcwd(), "dev", "questions.json"))
    )

    db_handler = VectorDatabaseHandler(index)
    query_processor = QueryProcessor(
        db_handler,
        question_dataset["namespaces"],
        score_threshold=0.4,  # Minimum score threshold for matches from Pinecone
        neighbour_range=3,  # Number of chunks before and after to fetch as neighbours
        gap_threshold=2,  # Maximum allowed gap between neighbour ranges to merge them
        top_k=32,  # top_k for the primary Pinecone chunk search
    )
    query_processor.process_queries(question_dataset["questions"])


if __name__ == "__main__":
    main()
