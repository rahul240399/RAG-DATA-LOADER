"""
Processing Results Data Models

This module defines data models for tracking and reporting the results
of document processing operations in the RAG indexing pipeline.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentProcessingResult:
    """
    Represents the result of processing a single document through the RAG pipeline.

    This class encapsulates all relevant information about the success or failure
    of processing a document, including performance metrics, error details,
    and processing statistics.

    Attributes:
        success (bool): Whether the document was processed successfully
        chunks_processed (int): Number of text chunks created from the document
        embeddings_generated (int): Number of embeddings successfully generated
        storage_success (bool): Whether embeddings were successfully stored
        processing_time (float): Total processing time in seconds
        errors (List[str]): List of error messages encountered during processing
        source_document (str): Path or identifier of the source document
    """

    success: bool
    chunks_processed: int
    embeddings_generated: int
    storage_success: bool
    processing_time: float
    errors: list[str]
    source_document: str

    def __post_init__(self) -> None:
        """
        Post-initialization method that validates logical consistency
        of the processing result attributes.
        """
        self._validate_consistency()

    def _validate_consistency(self) -> None:
        """
        Validate logical consistency between different result attributes.

        This method ensures that the result attributes are logically consistent
        with each other. For example, if success is True, there should be no errors,
        and if embeddings_generated > 0, chunks_processed should also be > 0.

        Raises:
            ValueError: If logical inconsistencies are detected
        """
        # If success is True, there should be no errors
        if self.success and len(self.errors) > 0:
            raise ValueError("Success cannot be True when errors are present")

        # If success is False, there should be at least one error
        if not self.success and len(self.errors) == 0:
            raise ValueError("Success cannot be False when no errors are present")

        # Embeddings generated cannot exceed chunks processed
        if self.embeddings_generated > self.chunks_processed:
            raise ValueError("Embeddings generated cannot exceed chunks processed")

        # If embeddings were generated, chunks must have been processed
        if self.embeddings_generated > 0 and self.chunks_processed == 0:
            raise ValueError("Cannot generate embeddings without processing chunks")

        # Storage success should align with overall success for complete processing
        if self.success and self.embeddings_generated > 0 and not self.storage_success:
            raise ValueError(
                "Overall success cannot be True if storage failed with embeddings generated"
            )

        # Processing time should be non-negative
        if self.processing_time < 0:
            raise ValueError("Processing time cannot be negative")

        # Counts should be non-negative
        if self.chunks_processed < 0:
            raise ValueError("Chunks processed count cannot be negative")

        if self.embeddings_generated < 0:
            raise ValueError("Embeddings generated count cannot be negative")

    def add_error(self, error_message: str) -> None:
        """
        Add an error message to the result and update success status.

        Args:
            error_message (str): The error message to add
        """
        self.errors.append(error_message)
        self.success = False

    def get_success_rate(self) -> float:
        """
        Calculate the success rate of embedding generation.

        Returns:
            float: Ratio of embeddings generated to chunks processed (0.0 to 1.0)
                   Returns 0.0 if no chunks were processed
        """
        if self.chunks_processed == 0:
            return 0.0
        return self.embeddings_generated / self.chunks_processed

    def is_partial_success(self) -> bool:
        """
        Check if the processing was partially successful.

        Returns:
            bool: True if some but not all chunks were successfully processed
        """
        return (
            self.chunks_processed > 0
            and self.embeddings_generated > 0
            and self.embeddings_generated < self.chunks_processed
        )


@dataclass
class BatchProcessingResult:
    """
    Represents the result of processing multiple documents through the RAG pipeline.

    This class aggregates individual DocumentProcessingResult instances and provides
    batch-level statistics and reporting capabilities. It supports continuing processing
    after individual document failures and provides comprehensive summary statistics.

    Attributes:
        total_documents (int): Total number of documents in the batch
        successful_documents (int): Number of documents processed successfully
        failed_documents (int): Number of documents that failed processing
        total_chunks_processed (int): Total chunks processed across all documents
        total_embeddings_generated (int): Total embeddings generated across all documents
        total_processing_time (float): Total processing time for the entire batch
        document_results (List[DocumentProcessingResult]): Individual document results
        batch_errors (List[str]): Batch-level errors (not document-specific)
        started_at (Optional[str]): Timestamp when batch processing started
        completed_at (Optional[str]): Timestamp when batch processing completed
    """

    total_documents: int = 0
    successful_documents: int = 0
    failed_documents: int = 0
    total_chunks_processed: int = 0
    total_embeddings_generated: int = 0
    total_processing_time: float = 0.0
    document_results: list[DocumentProcessingResult] = field(default_factory=list)
    batch_errors: list[str] = field(default_factory=list)
    started_at: str | None = None
    completed_at: str | None = None

    def add_document_result(self, result: DocumentProcessingResult) -> None:
        """
        Add a DocumentProcessingResult to the batch and update aggregate statistics.

        Args:
            result (DocumentProcessingResult): The document processing result to add
        """
        self.document_results.append(result)
        self.total_documents += 1

        if result.success:
            self.successful_documents += 1
        else:
            self.failed_documents += 1

        self.total_chunks_processed += result.chunks_processed
        self.total_embeddings_generated += result.embeddings_generated
        self.total_processing_time += result.processing_time

    def combine_results(self, results: list[DocumentProcessingResult]) -> None:
        """
        Combine multiple DocumentProcessingResult instances into this batch result.

        Args:
            results (List[DocumentProcessingResult]): List of document results to combine
        """
        for result in results:
            self.add_document_result(result)

    def get_success_rate(self) -> float:
        """
        Calculate the success rate of the batch processing.

        Returns:
            float: Ratio of successful documents to total documents (0.0 to 1.0)
                   Returns 0.0 if no documents were processed
        """
        if self.total_documents == 0:
            return 0.0
        return self.successful_documents / self.total_documents

    def get_embedding_success_rate(self) -> float:
        """
        Calculate the success rate of embedding generation across all chunks.

        Returns:
            float: Ratio of embeddings generated to chunks processed (0.0 to 1.0)
                   Returns 0.0 if no chunks were processed
        """
        if self.total_chunks_processed == 0:
            return 0.0
        return self.total_embeddings_generated / self.total_chunks_processed

    def get_failed_documents(self) -> list[DocumentProcessingResult]:
        """
        Get all failed document processing results.

        Returns:
            List[DocumentProcessingResult]: List of failed document results
        """
        return [result for result in self.document_results if not result.success]

    def get_successful_documents(self) -> list[DocumentProcessingResult]:
        """
        Get all successful document processing results.

        Returns:
            List[DocumentProcessingResult]: List of successful document results
        """
        return [result for result in self.document_results if result.success]

    def get_all_errors(self) -> dict[str, list[str]]:
        """
        Get all errors from both batch-level and document-level processing.

        Returns:
            Dict[str, List[str]]: Dictionary with 'batch_errors' and 'document_errors' keys
        """
        document_errors = []
        for result in self.document_results:
            if result.errors:
                document_errors.extend(
                    [f"{result.source_document}: {error}" for error in result.errors]
                )

        return {"batch_errors": self.batch_errors.copy(), "document_errors": document_errors}

    def add_batch_error(self, error_message: str) -> None:
        """
        Add a batch-level error message.

        Args:
            error_message (str): The batch-level error message to add
        """
        self.batch_errors.append(error_message)

    def get_summary_statistics(self) -> dict[str, Any]:
        """
        Get comprehensive summary statistics for the batch processing.

        Returns:
            dict[str, Any]: Dictionary containing detailed batch statistics
        """
        return {
            "total_documents": self.total_documents,
            "successful_documents": self.successful_documents,
            "failed_documents": self.failed_documents,
            "success_rate": self.get_success_rate(),
            "total_chunks_processed": self.total_chunks_processed,
            "total_embeddings_generated": self.total_embeddings_generated,
            "embedding_success_rate": self.get_embedding_success_rate(),
            "total_processing_time": self.total_processing_time,
            "average_processing_time_per_document": (
                self.total_processing_time / self.total_documents
                if self.total_documents > 0
                else 0.0
            ),
            "batch_errors_count": len(self.batch_errors),
            "total_document_errors_count": sum(
                len(result.errors) for result in self.document_results
            ),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    def is_complete_success(self) -> bool:
        """
        Check if all documents in the batch were processed successfully.

        Returns:
            bool: True if all documents succeeded and no batch errors occurred
        """
        return (
            self.total_documents > 0 and self.failed_documents == 0 and len(self.batch_errors) == 0
        )

    def is_complete_failure(self) -> bool:
        """
        Check if all documents in the batch failed processing.

        Returns:
            bool: True if all documents failed or batch-level errors occurred
        """
        return (self.total_documents > 0 and self.successful_documents == 0) or len(
            self.batch_errors
        ) > 0

    def is_partial_success(self) -> bool:
        """
        Check if the batch processing was partially successful.

        Returns:
            bool: True if some but not all documents were successfully processed
        """
        return (
            self.total_documents > 0
            and self.successful_documents > 0
            and self.failed_documents > 0
            and len(self.batch_errors) == 0
        )
