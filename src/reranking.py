from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from typing import List, Dict


class Reranker:
    def __init__(self, model_name="castorini/monot5-base-msmarco"):
        """
        Initialize the monoT5 reranker with the specified model.
        """
        print(f"Loading reranker model: {model_name}")

        # Use the slow tokenizer to avoid conversion issues
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def format_to_t5_query(self, query: str, candidate: str) -> str:
        """
        Format query-candidate pair for monoT5 input.
        """
        return f"Query: {query} Document: {candidate} Relevant:"

    def rerank(self, query: str, candidates: List[Dict[str, str]]) -> List[Dict[str, float]]:
        """
        Rerank the candidates based on relevance to the query.
        :param query: The query text.
        :param candidates: A list of candidate documents (chunks).
        :return: A list of candidates with relevance scores.
        """
        reranked_results = []

        for candidate in candidates:
            # Format input for monoT5
            t5_input = self.format_to_t5_query(query, candidate["text"])
            inputs = self.tokenizer(t5_input, return_tensors="pt", max_length=512, truncation=True)

            # Generate relevance score (use max_new_tokens to avoid max_length issues)
            with torch.no_grad():  # Disable gradient computation for inference
                outputs = self.model.generate(**inputs, max_new_tokens=5)  # Generate up to 5 new tokens
                relevance_score = float(self.tokenizer.decode(outputs[0], skip_special_tokens=True))

            reranked_results.append({**candidate, "relevance_score": relevance_score})

        # Sort candidates by relevance score
        reranked_results = sorted(reranked_results, key=lambda x: x["relevance_score"], reverse=True)
        return reranked_results
