"""
Unit tests for processing results data models.

Tests DocumentProcessingResult and BatchProcessingResult consistency validation
and method behavior.
Requirements: 6.5
"""

import pytest

from rag_loader.models.results import BatchProcessingResult, DocumentProcessingResult


class TestDocumentProcessingResultConsistency:
    """Test DocumentProcessingResult consistency validation."""

    def test_valid_successful_result(self):
        """Test creating a valid successful processing result."""
        result = DocumentProcessingResult(
            success=True,
            chunks_processed=10,
            embeddings_generated=10,
            storage_success=True,
            processing_time=5.5,
            errors=[],
            source_document="test.pdf",
        )

        assert result.success is True
        assert result.chunks_processed == 10
        assert result.embeddings_generated == 10
        assert result.storage_success is True
        assert result.processing_time == 5.5
        assert result.errors == []
        assert result.source_document == "test.pdf"

    def test_valid_failed_result(self):
        """Test creating a valid failed processing result."""
        result = DocumentProcessingResult(
            success=False,
            chunks_processed=5,
            embeddings_generated=3,
            storage_success=False,
            processing_time=2.1,
            errors=["Embedding generation failed for 2 chunks"],
            source_document="failed.pdf",
        )

        assert result.success is False
        assert result.chunks_processed == 5
        assert result.embeddings_generated == 3
        assert result.storage_success is False
        assert result.errors == ["Embedding generation failed for 2 chunks"]

    def test_success_true_with_errors_invalid(self):
        """Test that success=True with errors raises ValueError."""
        with pytest.raises(ValueError, match="Success cannot be True when errors are present"):
            DocumentProcessingResult(
                success=True,
                chunks_processed=10,
                embeddings_generated=10,
                storage_success=True,
                processing_time=5.0,
                errors=["Some error"],
                source_document="test.pdf",
            )

    def test_success_false_without_errors_invalid(self):
        """Test that success=False without errors raises ValueError."""
        with pytest.raises(ValueError, match="Success cannot be False when no errors are present"):
            DocumentProcessingResult(
                success=False,
                chunks_processed=5,
                embeddings_generated=3,
                storage_success=False,
                processing_time=2.0,
                errors=[],
                source_document="test.pdf",
            )

    def test_embeddings_exceed_chunks_invalid(self):
        """Test that embeddings_generated > chunks_processed raises ValueError."""
        with pytest.raises(ValueError, match="Embeddings generated cannot exceed chunks processed"):
            DocumentProcessingResult(
                success=True,
                chunks_processed=5,
                embeddings_generated=10,  # More than chunks processed
                storage_success=True,
                processing_time=3.0,
                errors=[],
                source_document="test.pdf",
            )

    def test_embeddings_without_chunks_invalid(self):
        """Test that embeddings_generated > 0 with chunks_processed = 0 raises ValueError."""
        with pytest.raises(ValueError, match="Embeddings generated cannot exceed chunks processed"):
            DocumentProcessingResult(
                success=True,
                chunks_processed=0,
                embeddings_generated=5,  # Embeddings without chunks
                storage_success=True,
                processing_time=1.0,
                errors=[],
                source_document="test.pdf",
            )

    def test_success_with_storage_failure_invalid(self):
        """success=True with storage failure and embeddings > 0 must raise ValueError."""
        with pytest.raises(
            ValueError,
            match="Overall success cannot be True if storage failed with embeddings generated",
        ):
            DocumentProcessingResult(
                success=True,
                chunks_processed=10,
                embeddings_generated=10,
                storage_success=False,  # Storage failed but success=True
                processing_time=5.0,
                errors=[],
                source_document="test.pdf",
            )

    def test_negative_processing_time_invalid(self):
        """Test that negative processing_time raises ValueError."""
        with pytest.raises(ValueError, match="Processing time cannot be negative"):
            DocumentProcessingResult(
                success=True,
                chunks_processed=10,
                embeddings_generated=10,
                storage_success=True,
                processing_time=-1.0,  # Negative time
                errors=[],
                source_document="test.pdf",
            )

    def test_negative_chunks_processed_invalid(self):
        """Test that negative chunks_processed raises ValueError."""
        with pytest.raises(ValueError, match="Embeddings generated cannot exceed chunks processed"):
            DocumentProcessingResult(
                success=False,
                chunks_processed=-5,  # Negative count
                embeddings_generated=0,
                storage_success=False,
                processing_time=1.0,
                errors=["Error occurred"],
                source_document="test.pdf",
            )

    def test_negative_embeddings_generated_invalid(self):
        """Test that negative embeddings_generated raises ValueError."""
        with pytest.raises(ValueError, match="Embeddings generated count cannot be negative"):
            DocumentProcessingResult(
                success=False,
                chunks_processed=5,
                embeddings_generated=-3,  # Negative count
                storage_success=False,
                processing_time=1.0,
                errors=["Error occurred"],
                source_document="test.pdf",
            )


class TestDocumentProcessingResultMethods:
    """Test DocumentProcessingResult methods."""

    def test_add_error_method(self):
        """Test add_error method updates errors and success status."""
        result = DocumentProcessingResult(
            success=True,
            chunks_processed=10,
            embeddings_generated=10,
            storage_success=True,
            processing_time=5.0,
            errors=[],
            source_document="test.pdf",
        )

        result.add_error("New error occurred")

        assert result.success is False
        assert "New error occurred" in result.errors
        assert len(result.errors) == 1

    def test_get_success_rate_full_success(self):
        """Test get_success_rate with full success."""
        result = DocumentProcessingResult(
            success=True,
            chunks_processed=10,
            embeddings_generated=10,
            storage_success=True,
            processing_time=5.0,
            errors=[],
            source_document="test.pdf",
        )

        assert result.get_success_rate() == 1.0

    def test_get_success_rate_partial_success(self):
        """Test get_success_rate with partial success."""
        result = DocumentProcessingResult(
            success=False,
            chunks_processed=10,
            embeddings_generated=7,
            storage_success=False,
            processing_time=5.0,
            errors=["Some chunks failed"],
            source_document="test.pdf",
        )

        assert result.get_success_rate() == 0.7

    def test_get_success_rate_no_chunks(self):
        """Test get_success_rate with no chunks processed."""
        result = DocumentProcessingResult(
            success=False,
            chunks_processed=0,
            embeddings_generated=0,
            storage_success=False,
            processing_time=1.0,
            errors=["No chunks to process"],
            source_document="empty.pdf",
        )

        assert result.get_success_rate() == 0.0

    def test_is_partial_success_true(self):
        """Test is_partial_success returns True for partial success."""
        result = DocumentProcessingResult(
            success=False,
            chunks_processed=10,
            embeddings_generated=7,
            storage_success=False,
            processing_time=5.0,
            errors=["Some chunks failed"],
            source_document="test.pdf",
        )

        assert result.is_partial_success() is True

    def test_is_partial_success_false_full_success(self):
        """Test is_partial_success returns False for full success."""
        result = DocumentProcessingResult(
            success=True,
            chunks_processed=10,
            embeddings_generated=10,
            storage_success=True,
            processing_time=5.0,
            errors=[],
            source_document="test.pdf",
        )

        assert result.is_partial_success() is False

    def test_is_partial_success_false_complete_failure(self):
        """Test is_partial_success returns False for complete failure."""
        result = DocumentProcessingResult(
            success=False,
            chunks_processed=10,
            embeddings_generated=0,
            storage_success=False,
            processing_time=5.0,
            errors=["All chunks failed"],
            source_document="test.pdf",
        )

        assert result.is_partial_success() is False


class TestBatchProcessingResultBasic:
    """Test BatchProcessingResult basic functionality."""

    def test_empty_batch_result_creation(self):
        """Test creating empty BatchProcessingResult."""
        batch = BatchProcessingResult()

        assert batch.total_documents == 0
        assert batch.successful_documents == 0
        assert batch.failed_documents == 0
        assert batch.total_chunks_processed == 0
        assert batch.total_embeddings_generated == 0
        assert batch.total_processing_time == 0.0
        assert batch.document_results == []
        assert batch.batch_errors == []
        assert batch.started_at is None
        assert batch.completed_at is None

    def test_add_document_result_successful(self):
        """Test adding successful document result to batch."""
        batch = BatchProcessingResult()
        doc_result = DocumentProcessingResult(
            success=True,
            chunks_processed=10,
            embeddings_generated=10,
            storage_success=True,
            processing_time=5.0,
            errors=[],
            source_document="test.pdf",
        )

        batch.add_document_result(doc_result)

        assert batch.total_documents == 1
        assert batch.successful_documents == 1
        assert batch.failed_documents == 0
        assert batch.total_chunks_processed == 10
        assert batch.total_embeddings_generated == 10
        assert batch.total_processing_time == 5.0
        assert len(batch.document_results) == 1

    def test_add_document_result_failed(self):
        """Test adding failed document result to batch."""
        batch = BatchProcessingResult()
        doc_result = DocumentProcessingResult(
            success=False,
            chunks_processed=5,
            embeddings_generated=3,
            storage_success=False,
            processing_time=2.0,
            errors=["Processing failed"],
            source_document="failed.pdf",
        )

        batch.add_document_result(doc_result)

        assert batch.total_documents == 1
        assert batch.successful_documents == 0
        assert batch.failed_documents == 1
        assert batch.total_chunks_processed == 5
        assert batch.total_embeddings_generated == 3
        assert batch.total_processing_time == 2.0

    def test_combine_results_method(self):
        """Test combine_results method with multiple document results."""
        batch = BatchProcessingResult()

        results = [
            DocumentProcessingResult(
                success=True,
                chunks_processed=10,
                embeddings_generated=10,
                storage_success=True,
                processing_time=5.0,
                errors=[],
                source_document="doc1.pdf",
            ),
            DocumentProcessingResult(
                success=False,
                chunks_processed=5,
                embeddings_generated=3,
                storage_success=False,
                processing_time=2.0,
                errors=["Error"],
                source_document="doc2.pdf",
            ),
        ]

        batch.combine_results(results)

        assert batch.total_documents == 2
        assert batch.successful_documents == 1
        assert batch.failed_documents == 1
        assert batch.total_chunks_processed == 15
        assert batch.total_embeddings_generated == 13
        assert batch.total_processing_time == 7.0


class TestBatchProcessingResultMethods:
    """Test BatchProcessingResult methods."""

    def test_get_success_rate_mixed_results(self):
        """Test get_success_rate with mixed successful and failed results."""
        batch = BatchProcessingResult()

        # Add 3 successful and 2 failed results
        for i in range(3):
            batch.add_document_result(
                DocumentProcessingResult(
                    success=True,
                    chunks_processed=10,
                    embeddings_generated=10,
                    storage_success=True,
                    processing_time=5.0,
                    errors=[],
                    source_document=f"success{i}.pdf",
                )
            )

        for i in range(2):
            batch.add_document_result(
                DocumentProcessingResult(
                    success=False,
                    chunks_processed=5,
                    embeddings_generated=0,
                    storage_success=False,
                    processing_time=2.0,
                    errors=["Error"],
                    source_document=f"failed{i}.pdf",
                )
            )

        assert batch.get_success_rate() == 0.6  # 3/5

    def test_get_success_rate_empty_batch(self):
        """Test get_success_rate with empty batch."""
        batch = BatchProcessingResult()
        assert batch.get_success_rate() == 0.0

    def test_get_embedding_success_rate(self):
        """Test get_embedding_success_rate calculation."""
        batch = BatchProcessingResult()

        batch.add_document_result(
            DocumentProcessingResult(
                success=True,
                chunks_processed=10,
                embeddings_generated=8,
                storage_success=True,
                processing_time=5.0,
                errors=[],
                source_document="doc1.pdf",
            )
        )

        batch.add_document_result(
            DocumentProcessingResult(
                success=False,
                chunks_processed=10,
                embeddings_generated=2,
                storage_success=False,
                processing_time=3.0,
                errors=["Error"],
                source_document="doc2.pdf",
            )
        )

        assert batch.get_embedding_success_rate() == 0.5  # 10/20

    def test_get_failed_documents(self):
        """Test get_failed_documents method."""
        batch = BatchProcessingResult()

        success_result = DocumentProcessingResult(
            success=True,
            chunks_processed=10,
            embeddings_generated=10,
            storage_success=True,
            processing_time=5.0,
            errors=[],
            source_document="success.pdf",
        )

        failed_result = DocumentProcessingResult(
            success=False,
            chunks_processed=5,
            embeddings_generated=0,
            storage_success=False,
            processing_time=2.0,
            errors=["Error"],
            source_document="failed.pdf",
        )

        batch.add_document_result(success_result)
        batch.add_document_result(failed_result)

        failed_docs = batch.get_failed_documents()
        assert len(failed_docs) == 1
        assert failed_docs[0].source_document == "failed.pdf"

    def test_get_successful_documents(self):
        """Test get_successful_documents method."""
        batch = BatchProcessingResult()

        success_result = DocumentProcessingResult(
            success=True,
            chunks_processed=10,
            embeddings_generated=10,
            storage_success=True,
            processing_time=5.0,
            errors=[],
            source_document="success.pdf",
        )

        failed_result = DocumentProcessingResult(
            success=False,
            chunks_processed=5,
            embeddings_generated=0,
            storage_success=False,
            processing_time=2.0,
            errors=["Error"],
            source_document="failed.pdf",
        )

        batch.add_document_result(success_result)
        batch.add_document_result(failed_result)

        successful_docs = batch.get_successful_documents()
        assert len(successful_docs) == 1
        assert successful_docs[0].source_document == "success.pdf"

    def test_get_all_errors(self):
        """Test get_all_errors method."""
        batch = BatchProcessingResult()
        batch.add_batch_error("Batch initialization error")

        batch.add_document_result(
            DocumentProcessingResult(
                success=False,
                chunks_processed=5,
                embeddings_generated=0,
                storage_success=False,
                processing_time=2.0,
                errors=["Doc error 1", "Doc error 2"],
                source_document="failed.pdf",
            )
        )

        all_errors = batch.get_all_errors()

        assert "Batch initialization error" in all_errors["batch_errors"]
        assert "failed.pdf: Doc error 1" in all_errors["document_errors"]
        assert "failed.pdf: Doc error 2" in all_errors["document_errors"]

    def test_add_batch_error(self):
        """Test add_batch_error method."""
        batch = BatchProcessingResult()
        batch.add_batch_error("Configuration error")

        assert "Configuration error" in batch.batch_errors
        assert len(batch.batch_errors) == 1

    def test_is_complete_success(self):
        """Test is_complete_success method."""
        batch = BatchProcessingResult()

        # Empty batch should not be complete success
        assert batch.is_complete_success() is False

        # Add successful document
        batch.add_document_result(
            DocumentProcessingResult(
                success=True,
                chunks_processed=10,
                embeddings_generated=10,
                storage_success=True,
                processing_time=5.0,
                errors=[],
                source_document="success.pdf",
            )
        )

        assert batch.is_complete_success() is True

        # Add batch error
        batch.add_batch_error("Some error")
        assert batch.is_complete_success() is False

    def test_is_complete_failure(self):
        """Test is_complete_failure method."""
        batch = BatchProcessingResult()

        # Empty batch should not be complete failure
        assert batch.is_complete_failure() is False

        # Add failed document
        batch.add_document_result(
            DocumentProcessingResult(
                success=False,
                chunks_processed=5,
                embeddings_generated=0,
                storage_success=False,
                processing_time=2.0,
                errors=["Error"],
                source_document="failed.pdf",
            )
        )

        assert batch.is_complete_failure() is True

        # Add successful document
        batch.add_document_result(
            DocumentProcessingResult(
                success=True,
                chunks_processed=10,
                embeddings_generated=10,
                storage_success=True,
                processing_time=5.0,
                errors=[],
                source_document="success.pdf",
            )
        )

        assert batch.is_complete_failure() is False

    def test_is_partial_success(self):
        """Test is_partial_success method."""
        batch = BatchProcessingResult()

        # Add both successful and failed documents
        batch.add_document_result(
            DocumentProcessingResult(
                success=True,
                chunks_processed=10,
                embeddings_generated=10,
                storage_success=True,
                processing_time=5.0,
                errors=[],
                source_document="success.pdf",
            )
        )

        batch.add_document_result(
            DocumentProcessingResult(
                success=False,
                chunks_processed=5,
                embeddings_generated=0,
                storage_success=False,
                processing_time=2.0,
                errors=["Error"],
                source_document="failed.pdf",
            )
        )

        assert batch.is_partial_success() is True

        # Add batch error should make it not partial success
        batch.add_batch_error("Batch error")
        assert batch.is_partial_success() is False


class TestBatchProcessingResultEdgeCases:
    """Test BatchProcessingResult edge cases."""

    def test_get_summary_statistics_empty_batch(self):
        """Test get_summary_statistics with empty batch."""
        batch = BatchProcessingResult()
        stats = batch.get_summary_statistics()

        assert stats["total_documents"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["average_processing_time_per_document"] == 0.0
        assert stats["batch_errors_count"] == 0
        assert stats["total_document_errors_count"] == 0

    def test_get_summary_statistics_with_data(self):
        """Test get_summary_statistics with actual data."""
        batch = BatchProcessingResult()
        batch.started_at = "2024-01-01T10:00:00"
        batch.completed_at = "2024-01-01T10:05:00"

        batch.add_document_result(
            DocumentProcessingResult(
                success=True,
                chunks_processed=10,
                embeddings_generated=10,
                storage_success=True,
                processing_time=3.0,
                errors=[],
                source_document="doc1.pdf",
            )
        )

        batch.add_document_result(
            DocumentProcessingResult(
                success=False,
                chunks_processed=5,
                embeddings_generated=2,
                storage_success=False,
                processing_time=2.0,
                errors=["Error1", "Error2"],
                source_document="doc2.pdf",
            )
        )

        batch.add_batch_error("Batch error")

        stats = batch.get_summary_statistics()

        assert stats["total_documents"] == 2
        assert stats["successful_documents"] == 1
        assert stats["failed_documents"] == 1
        assert stats["success_rate"] == 0.5
        assert stats["total_chunks_processed"] == 15
        assert stats["total_embeddings_generated"] == 12
        assert stats["embedding_success_rate"] == 0.8  # 12/15
        assert stats["total_processing_time"] == 5.0
        assert stats["average_processing_time_per_document"] == 2.5
        assert stats["batch_errors_count"] == 1
        assert stats["total_document_errors_count"] == 2
        assert stats["started_at"] == "2024-01-01T10:00:00"
        assert stats["completed_at"] == "2024-01-01T10:05:00"
