"""
RAG Chat System with OpenAI and Pinecone

This script implements a Retrieval-Augmented Generation (RAG) chat system using OpenAI's
embedding models and Pinecone's vector database. It processes user queries by generating
embeddings, retrieving relevant document chunks, handling multiple matches within the same
document, fetching contextual neighbors, and merging content to provide comprehensive responses.

Key Components:
- VectorDatabaseHandler: Manages embeddings creation, querying Pinecone, and fetching items.
- ContextFetcher: Retrieves and merges neighboring document chunks based on matched results.
- QueryProcessor: Orchestrates the query processing workflow, including filtering and scoring.

Configuration:
- Requires OpenAI and Pinecone API keys set in a `.env` file.
- Configurable parameters include `num_neighbours`, `score_threshold`, `neighbor_range`, and `gap_threshold`.

Usage:
1. Define a list of questions and namespaces.
2. Initialize VectorDatabaseHandler and QueryProcessor with desired settings.
3. Run the script to process queries and obtain merged contextual responses.

Notes:
- This code serves as a testbed to determine optimal settings for chunk size and overlap.
- It is a work in progress and currently lacks features to feed merged contexts to an LLM for answer generation.
- Future enhancements include integrating an LLM to generate and verify answers based on the retrieved context.


Ensure all dependencies are installed and the Pinecone index is properly set up before execution.
"""

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

            logging.info(
                f"Found {len(all_ids)} chunks for document '{document_id}' in namespace '{namespace}'."
            )
        except Exception as e:
            logging.info(
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

    def fetch_neighbours(
        self,
        match: QueryResult,
        namespace: str,
        num_neighbours: Optional[int] = None,  # None or 'all' implies 'all'
    ) -> List[Dict[str, Any]]:
        document_id, chunk_id = self.get_document_id_and_chunk_id(match.metadata["id"])
        all_ids = self.db_handler.list_ids_by_document(document_id, namespace)
        all_ids_sorted = sorted(
            all_ids, key=lambda x: self.get_document_id_and_chunk_id(x)[1]
        )
        try:
            current_index = all_ids_sorted.index(match.metadata["id"])
        except ValueError:
            current_index = 0  # If not found, default to first chunk

        if num_neighbours == "all" or num_neighbours is None:
            neighbour_ids = all_ids_sorted
        else:
            half = num_neighbours // 2
            start = max(current_index - half, 0)
            end = min(current_index + half + 1, len(all_ids_sorted))
            neighbour_ids = all_ids_sorted[start:end]

        logging.info(
            f"Matched on chunk {chunk_id}, fetching neighbours {neighbour_ids}."
        )

        fetched_chunks = self.db_handler.fetch_items(neighbour_ids, namespace)

        # Sort fetched chunks by chunk_id
        fetched_chunks = sorted(
            fetched_chunks,
            key=lambda x: (
                int(x["id"].split("#")[1])
                if "#" in x["id"] and x["id"].split("#")[1].isdigit()
                else 0
            ),
        )

        return fetched_chunks

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
                # Assuming overlap_size is the number of words
                overlap_words = self.overlap_size
                merged_content_words = merged_content.split()
                new_text_words = text.split()

                # Find overlapping part
                overlap_candidate = new_text_words[:overlap_words]
                merged_overlap = merged_content_words[-overlap_words:]

                # Check if the end of merged_content matches the start of new text
                if merged_overlap == overlap_candidate:
                    # Append without overlapping part
                    merged_content += " " + " ".join(new_text_words[overlap_words:])
                else:
                    # If no exact overlap, attempt partial overlap or append entirely
                    # Here, we simply append with a space
                    merged_content += " " + text

        return merged_content


class QueryProcessor:
    def __init__(
        self,
        db_handler: VectorDatabaseHandler,
        namespaces: List[str],
        num_neighbours: Optional[int] = 2,  # Default number of neighbours
        score_threshold: float = 0.5,  # Default score threshold
        neighbour_range: int = 2,  # Number of chunks before and after to fetch as neighbours
        gap_threshold: int = 1,  # Maximum allowed gap between neighbour ranges to merge them
    ):
        self.db_handler = db_handler
        self.namespaces = namespaces
        self.num_neighbours = num_neighbours
        self.score_threshold = score_threshold
        self.neighbour_range = neighbour_range
        self.gap_threshold = gap_threshold

    def process_queries(self, questions: List[str]):
        # Create embeddings for all questions
        embeddings = self.db_handler.create_embeddings(questions)

        # Iterate over each question and its embedding
        for question, embedding in zip(questions, embeddings):
            logging.info(f"\nProcessing Query: \"{question}\"\n{'='*50}")

            for namespace in self.namespaces:
                logging.info(f"\n--- Namespace: {namespace} ---")
                # Parse namespace to get overlap size
                match = re.match(r"web-\d+-(\d+)", namespace)
                overlap_size = int(match.group(1)) if match else 100  # Default overlap
                context_fetcher = ContextFetcher(self.db_handler, overlap_size)

                # Query Pinecone
                top_k = 20  # Increased top_k for more matches
                results = self.db_handler.query(embedding, namespace, top_k)

                logging.info(f"Found {len(results)} matching chunks")

                # Preprocess results: filter by score and group by document_id
                filtered_results = [
                    result for result in results if result.score >= self.score_threshold
                ]

                # Group results by document_id
                grouped_results: Dict[str, List[QueryResult]] = {}
                for result in filtered_results:
                    document_id, _ = context_fetcher.get_document_id_and_chunk_id(
                        result.metadata["id"]
                    )
                    if document_id not in grouped_results:
                        grouped_results[document_id] = []
                    grouped_results[document_id].append(result)

                logging.info(
                    f"Found {len(filtered_results)} matches after applying score threshold of {self.score_threshold}."
                )

                logging.info(f"Processing {len(grouped_results)} unique documents.")

                for document_id, doc_results in grouped_results.items():
                    logging.info(f"\n--- Document ID: {document_id} ---")
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

                    logging.info(
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

                    logging.info(f"Fetching neighbours: {neighbour_ids}")

                    # Fetch the neighbour chunks
                    fetched_chunks = self.db_handler.fetch_items(
                        neighbour_ids, namespace
                    )

                    # Merge content
                    merged_content = context_fetcher.merge_chunks(fetched_chunks)

                    # Aggregate scores: take the highest score among matches
                    highest_score = max(result.score for result in doc_results)
                    # Calculate weighted score based on the latest last_modified
                    latest_last_modified = max(
                        result.metadata.get("lastModified", 0) for result in doc_results
                    )
                    weighted_score = self.weight_score(
                        highest_score, latest_last_modified
                    )

                    # Display or process the merged content
                    logging.info(
                        f"[Highest Score: {highest_score:.4f}, Weighted Score: {weighted_score:.4f}]"
                    )
                    logging.info(
                        f"Last Modified: {self.format_timestamp(latest_last_modified)}"
                    )
                    logging.info(
                        f"Merged Content:\n{merged_content[0:140]}...\n{'-'*40}"
                    )

                # For demonstration, just process one namespace and exit
                break

            # For demonstration, process one question and exit
            break

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
                logging.info(f"Merged range {last} with {current} to {merged[-1]}")
            else:
                merged.append(current)
                logging.info(f"Added new range {current} to merged ranges")

        logging.info(f"Merged neighbour ranges: {merged}")
        return merged

    @staticmethod
    def weight_score(score: float, last_modified: float) -> float:
        """Combine similarity score with recency. Newer content gets higher weight."""
        # Normalize last_modified to a score between 0 and 1
        current_time_ms = datetime.now(timezone.utc).timestamp() * 1000
        age_ms = current_time_ms - last_modified
        # Assuming that content older than 5 years gets the lowest weight
        max_age = 5 * 365 * 24 * 60 * 60 * 1000
        recency_score = max(0, 1 - age_ms / max_age)
        # Weight factors can be adjusted
        similarity_weight = 0.7
        recency_weight = 0.3
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
    questions = [
        "What projects have we done involving swimming pools?",  # needs to pull recent projects
        "What 50m swimming pools have we made?",  # we need to mention the 50m pool at Winchester Sports Park completed in 2022
        "landscape work",  # can it focus on more recent projects like the Broadmarsh centre
        "How many passivhaus projects has Willmott Dixon completed?",  # there’s loads of Passivhaus projects on this page https://www.willmottdixon.co.uk/expertise/passivhaus I’ve also updated the intro to include mention of 9 projects
        "What bluelight projects have you done?",  # Can we refer this question to projects featured in our blue light sector and showcases projects for the police force and emergency services
        "Projects in Liverpool?",  # All these projects are complete to how can we train it to reflect that. Most have case studies too on the completion. Education response is spot on. Police headquarters were finished two years ago, so not in preconstruction. No mention either of Kings Dock Car Park in Liverpool
    ]

    namespaces = [
        "web-600-100",
        "web-600-150",
        "web-700-100",
        "web-700-150",
        "web-800-100",
        "web-800-150",
        "web-1000-200",  # default
    ]

    db_handler = VectorDatabaseHandler(index)
    query_processor = QueryProcessor(
        db_handler,
        namespaces,
        num_neighbours=2,
        score_threshold=0.5,
        neighbour_range=2,  # Number of chunks before and after to fetch as neighbours
        gap_threshold=1,  # Maximum allowed gap between neighbour ranges to merge them
    )
    # Adjust num_neighbours, score_threshold, neighbour_range, and gap_threshold as needed
    query_processor.process_queries(questions)


if __name__ == "__main__":
    main()
