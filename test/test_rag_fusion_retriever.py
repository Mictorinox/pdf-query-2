# python -m unittest test/test_rag_fusion_retriever.py

import unittest
from unittest.mock import MagicMock
from retrievers.rag_fusion_retriever import generate_queries, reciprocal_rank_fusion, RagFusionRetriever

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_community.vectorstores import Chroma

class TestRagFusionRetriever(unittest.TestCase):

    def test_generate_queries(self):
        # Mock LLM for testing generate_queries
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_llm.invoke.return_value = AIMessage(content="query1\nquery2\nquery3")
        
        
        original_query = "What is the capital of France?"
        generated_queries = generate_queries(original_query, mock_llm)
        
        self.assertIsInstance(generated_queries, list)
        self.assertGreater(len(generated_queries), 0)
        self.assertIn("query1", generated_queries)
        self.assertIn("query2", generated_queries)
        self.assertIn("query3", generated_queries)

    def test_reciprocal_rank_fusion(self):
        # Example search results for RRF
        results = [
            [MagicMock(page_content="doc_A"), MagicMock(page_content="doc_B"), MagicMock(page_content="doc_C")],
            [MagicMock(page_content="doc_B"), MagicMock(page_content="doc_D"), MagicMock(page_content="doc_A")]
        ]
        
        fused_results = reciprocal_rank_fusion(results)
        
        self.assertIsInstance(fused_results, list)
        self.assertGreater(len(fused_results), 0)
        # doc_B should have a higher score (1/1 + 1/1 = 2) than doc_A (1/1 + 1/3 = 1.33)
        # and doc_C (1/3) and doc_D (1/2)
        self.assertEqual(fused_results[0].page_content, "doc_B")
        self.assertEqual(fused_results[1].page_content, "doc_A")
        self.assertEqual(fused_results[2].page_content, "doc_D")
        self.assertEqual(fused_results[3].page_content, "doc_C")

    def test_rag_fusion_retriever(self):
        # Mock LLM and vector store for RagFusionRetriever
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_llm.invoke.return_value = AIMessage(content="query1\nquery2")
        
        mock_vectorstore = MagicMock(spec=Chroma)
        mock_vectorstore.similarity_search.side_effect = [
            [MagicMock(page_content="doc_A"), MagicMock(page_content="doc_B")], # Results for original query
            [MagicMock(page_content="doc_B"), MagicMock(page_content="doc_C")], # Results for "query1"
            [MagicMock(page_content="doc_A"), MagicMock(page_content="doc_D")]  # Results for "query2"
        ]

        retriever = RagFusionRetriever(llm=mock_llm, vector_store=mock_vectorstore)
        query = "Test query"
        mock_run_manager = MagicMock(spec=CallbackManagerForRetrieverRun)
        retrieved_docs = retriever._get_relevant_documents(query, run_manager=mock_run_manager)
        
        self.assertIsInstance(retrieved_docs, list)
        self.assertGreater(len(retrieved_docs), 0)
        # Verify that similarity_search was called for each generated query
        self.assertEqual(mock_vectorstore.similarity_search.call_count, 3) # Original query + 2 generated queries
        # Verify the order of fused results based on the mock data
        self.assertEqual(retrieved_docs[0].page_content, "doc_A")
        self.assertEqual(retrieved_docs[1].page_content, "doc_B")
        self.assertIn(retrieved_docs[2].page_content, ["doc_C", "doc_D"])
        self.assertIn(retrieved_docs[3].page_content, ["doc_C", "doc_D"])

if __name__ == '__main__':
    unittest.main()