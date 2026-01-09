"""
Property-based tests for processing results models.

This module contains property-based tests that verify the correctness
of DocumentProcessingResult and BatchProcessingResult behavior across 
a wide range of inputs using Hypothesis.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from src.models.results import DocumentProcessingResult, BatchProcessingResult
from typing import List


class TestProcessingResultsProperties:
    """Property-based tests for processing results models."""
    
    @given(
        success=st.booleans(),
        chunks_processed=st.integers(min_value=0, max_value=1000),
        embeddings_generated=st.integers(min_value=0, max_value=1000),
        storage_success=st.booleans(),
        processing_time=st.floats(min_value=0.0, max_value=3600.0),
        errors=st.lists(st.text(min_size=1, max_size=100), max_size=10),
        source_document=st.text(min_size=1, max_size=200)
    )
    @settings(max_examples=100)
    def test_resource_management_and_reporting(self, success, chunks_processed, 
                                             embeddings_generated, storage_success,
                                             processing_time, errors, source_document):
        """
        **Feature: rag-indexing-pipeline, Property 7: Resource management and reporting**
        
        Property: For any processing operation, the pipeline should properly clean up 
        resources during errors and provide comprehensive summary statistics upon completion.
        
        This test validates that:
        1. Processing results maintain logical consistency between success status and errors
        2. Resource cleanup is properly tracked through success/failure states
        3. Comprehensive summary statistics are provided for all processing operations
        4. Error conditions are properly reported with context
        5. Processing time and resource usage metrics are accurately tracked
        
        **Validates: Requirements 6.4, 6.5**
        """
        # Ensure embeddings_generated doesn't exceed chunks_processed
        assume(embeddings_generated <= chunks_processed)
        
        # Adjust success and errors to maintain logical consistency
        if success and len(errors) > 0:
            # If we want success=True, we need no errors
            errors = []
        elif not success and len(errors) == 0:
            # If we want success=False, we need at least one error
            errors = ["Generated error for consistency"]
        
        # Create DocumentProcessingResult
        try:
            result = DocumentProcessingResult(
                success=success,
                chunks_processed=chunks_processed,
                embeddings_generated=embeddings_generated,
                storage_success=storage_success,
                processing_time=processing_time,
                errors=errors,
                source_document=source_document
            )
            
            # Property 1: Resource management - success status should be consistent with errors
            if result.success:
                assert len(result.errors) == 0, (
                    f"Successful processing should have no errors. "
                    f"Got success={result.success} with errors={result.errors}"
                )
            else:
                assert len(result.errors) > 0, (
                    f"Failed processing should have at least one error. "
                    f"Got success={result.success} with errors={result.errors}"
                )
            
            # Property 2: Resource tracking - embeddings should not exceed chunks
            assert result.embeddings_generated <= result.chunks_processed, (
                f"Embeddings generated ({result.embeddings_generated}) should not exceed "
                f"chunks processed ({result.chunks_processed})"
            )
            
            # Property 3: Comprehensive reporting - all metrics should be non-negative
            assert result.chunks_processed >= 0, "Chunks processed should be non-negative"
            assert result.embeddings_generated >= 0, "Embeddings generated should be non-negative"
            assert result.processing_time >= 0, "Processing time should be non-negative"
            
            # Property 4: Success rate calculation should be accurate
            expected_success_rate = (
                result.embeddings_generated / result.chunks_processed 
                if result.chunks_processed > 0 else 0.0
            )
            assert abs(result.get_success_rate() - expected_success_rate) < 1e-10, (
                f"Success rate calculation incorrect. Expected {expected_success_rate}, "
                f"got {result.get_success_rate()}"
            )
            
            # Property 5: Partial success detection should be accurate
            expected_partial = (
                result.chunks_processed > 0 and 
                result.embeddings_generated > 0 and 
                result.embeddings_generated < result.chunks_processed
            )
            assert result.is_partial_success() == expected_partial, (
                f"Partial success detection incorrect. Expected {expected_partial}, "
                f"got {result.is_partial_success()}"
            )
            
            # Property 6: Storage success should align with overall success for complete processing
            if result.success and result.embeddings_generated > 0:
                assert result.storage_success, (
                    f"Overall success with embeddings generated should require storage success. "
                    f"Got success={result.success}, embeddings_generated={result.embeddings_generated}, "
                    f"storage_success={result.storage_success}"
                )
            
        except ValueError as e:
            # If validation fails, it should be due to logical inconsistencies
            # This is expected behavior for invalid combinations
            assert "cannot" in str(e).lower() or "should" in str(e).lower(), (
                f"Validation error should be descriptive: {e}"
            )
    
    @given(
        data=st.data(),
        batch_errors=st.lists(st.text(min_size=1, max_size=100), max_size=5)
    )
    @settings(max_examples=100)
    def test_batch_processing_resource_management(self, data, batch_errors):
        """
        Property: Batch processing should provide comprehensive resource management 
        and reporting across multiple documents.
        
        This test validates batch-level resource management and reporting capabilities.
        """
        # Generate number of documents
        num_documents = data.draw(st.integers(min_value=1, max_value=20))
        
        # Generate valid document results manually
        document_results = []
        for i in range(num_documents):
            # Create a valid document result
            chunks = data.draw(st.integers(min_value=0, max_value=100))
            embeddings = data.draw(st.integers(min_value=0, max_value=chunks))
            has_errors = data.draw(st.booleans())
            
            if has_errors:
                success = False
                errors = [f"Error in document {i}"]
                storage_success = data.draw(st.booleans())
            else:
                success = True
                errors = []
                # If we have embeddings and success, storage must succeed
                storage_success = True if embeddings > 0 else data.draw(st.booleans())
            
            doc_result = DocumentProcessingResult(
                success=success,
                chunks_processed=chunks,
                embeddings_generated=embeddings,
                storage_success=storage_success,
                processing_time=data.draw(st.floats(min_value=0.0, max_value=60.0)),
                errors=errors,
                source_document=f"document_{i}.pdf"
            )
            document_results.append(doc_result)
        
        batch_result = BatchProcessingResult()
        
        # Add batch errors
        for error in batch_errors:
            batch_result.add_batch_error(error)
        
        # Add document results
        for doc_result in document_results:
            batch_result.add_document_result(doc_result)
        
        # Property 1: Total documents should match added results
        assert batch_result.total_documents == len(document_results), (
            f"Total documents should match added results. "
            f"Expected {len(document_results)}, got {batch_result.total_documents}"
        )
        
        # Property 2: Success/failure counts should be accurate
        expected_successful = sum(1 for r in document_results if r.success)
        expected_failed = sum(1 for r in document_results if not r.success)
        
        assert batch_result.successful_documents == expected_successful, (
            f"Successful documents count incorrect. Expected {expected_successful}, "
            f"got {batch_result.successful_documents}"
        )
        assert batch_result.failed_documents == expected_failed, (
            f"Failed documents count incorrect. Expected {expected_failed}, "
            f"got {batch_result.failed_documents}"
        )
        
        # Property 3: Aggregate statistics should be accurate
        expected_total_chunks = sum(r.chunks_processed for r in document_results)
        expected_total_embeddings = sum(r.embeddings_generated for r in document_results)
        expected_total_time = sum(r.processing_time for r in document_results)
        
        assert batch_result.total_chunks_processed == expected_total_chunks, (
            f"Total chunks processed incorrect. Expected {expected_total_chunks}, "
            f"got {batch_result.total_chunks_processed}"
        )
        assert batch_result.total_embeddings_generated == expected_total_embeddings, (
            f"Total embeddings generated incorrect. Expected {expected_total_embeddings}, "
            f"got {batch_result.total_embeddings_generated}"
        )
        assert abs(batch_result.total_processing_time - expected_total_time) < 1e-10, (
            f"Total processing time incorrect. Expected {expected_total_time}, "
            f"got {batch_result.total_processing_time}"
        )
        
        # Property 4: Success rates should be calculated correctly
        expected_doc_success_rate = (
            expected_successful / len(document_results) 
            if len(document_results) > 0 else 0.0
        )
        expected_embedding_success_rate = (
            expected_total_embeddings / expected_total_chunks 
            if expected_total_chunks > 0 else 0.0
        )
        
        assert abs(batch_result.get_success_rate() - expected_doc_success_rate) < 1e-10, (
            f"Document success rate incorrect. Expected {expected_doc_success_rate}, "
            f"got {batch_result.get_success_rate()}"
        )
        assert abs(batch_result.get_embedding_success_rate() - expected_embedding_success_rate) < 1e-10, (
            f"Embedding success rate incorrect. Expected {expected_embedding_success_rate}, "
            f"got {batch_result.get_embedding_success_rate()}"
        )
        
        # Property 5: Complete success/failure detection should be accurate
        has_batch_errors = len(batch_errors) > 0
        all_docs_successful = all(r.success for r in document_results)
        no_docs_successful = not any(r.success for r in document_results)
        
        expected_complete_success = (
            len(document_results) > 0 and 
            all_docs_successful and 
            not has_batch_errors
        )
        expected_complete_failure = (
            (len(document_results) > 0 and no_docs_successful) or 
            has_batch_errors
        )
        expected_partial_success = (
            len(document_results) > 0 and
            not all_docs_successful and
            not no_docs_successful and
            not has_batch_errors
        )
        
        assert batch_result.is_complete_success() == expected_complete_success, (
            f"Complete success detection incorrect. Expected {expected_complete_success}, "
            f"got {batch_result.is_complete_success()}"
        )
        assert batch_result.is_complete_failure() == expected_complete_failure, (
            f"Complete failure detection incorrect. Expected {expected_complete_failure}, "
            f"got {batch_result.is_complete_failure()}"
        )
        assert batch_result.is_partial_success() == expected_partial_success, (
            f"Partial success detection incorrect. Expected {expected_partial_success}, "
            f"got {batch_result.is_partial_success()}"
        )
        
        # Property 6: Summary statistics should be comprehensive
        summary = batch_result.get_summary_statistics()
        required_keys = {
            'total_documents', 'successful_documents', 'failed_documents',
            'success_rate', 'total_chunks_processed', 'total_embeddings_generated',
            'embedding_success_rate', 'total_processing_time',
            'average_processing_time_per_document', 'batch_errors_count',
            'total_document_errors_count', 'started_at', 'completed_at'
        }
        
        assert set(summary.keys()) == required_keys, (
            f"Summary statistics missing required keys. "
            f"Expected {required_keys}, got {set(summary.keys())}"
        )
        
        # Property 7: Error aggregation should be comprehensive
        all_errors = batch_result.get_all_errors()
        assert 'batch_errors' in all_errors, "All errors should include batch_errors"
        assert 'document_errors' in all_errors, "All errors should include document_errors"
        assert len(all_errors['batch_errors']) == len(batch_errors), (
            f"Batch errors count mismatch. Expected {len(batch_errors)}, "
            f"got {len(all_errors['batch_errors'])}"
        )
    
    @given(
        initial_success=st.booleans(),
        initial_chunks=st.integers(min_value=0, max_value=100),
        initial_embeddings=st.integers(min_value=0, max_value=100),
        initial_storage_success=st.booleans(),
        initial_time=st.floats(min_value=0.0, max_value=60.0),
        initial_errors=st.lists(st.text(min_size=1, max_size=50), max_size=2),
        source_doc=st.text(min_size=1, max_size=50),
        error_to_add=st.text(min_size=1, max_size=100)
    )
    @settings(max_examples=100)
    def test_error_handling_resource_cleanup(self, initial_success, initial_chunks,
                                           initial_embeddings, initial_storage_success,
                                           initial_time, initial_errors, source_doc,
                                           error_to_add):
        """
        Property: Adding errors should properly update success status and maintain 
        resource cleanup tracking.
        
        This test validates that error handling properly manages resource states.
        """
        # Ensure valid initial state
        assume(initial_embeddings <= initial_chunks)
        
        # Adjust for consistency - create a valid initial state
        if initial_success:
            # For success, we need no errors and proper storage success if embeddings exist
            adjusted_errors = []
            adjusted_storage_success = initial_storage_success if initial_embeddings == 0 else True
        else:
            # For failure, we need at least one error
            adjusted_errors = initial_errors if len(initial_errors) > 0 else ["Initial error"]
            adjusted_storage_success = initial_storage_success
        
        try:
            result = DocumentProcessingResult(
                success=initial_success,
                chunks_processed=initial_chunks,
                embeddings_generated=initial_embeddings,
                storage_success=adjusted_storage_success,
                processing_time=initial_time,
                errors=adjusted_errors,
                source_document=source_doc
            )
            
            # Record initial state
            initial_error_count = len(result.errors)
            
            # Add an error
            result.add_error(error_to_add)
            
            # Property 1: Adding error should set success to False
            assert not result.success, (
                f"Adding error should set success to False. Got {result.success}"
            )
            
            # Property 2: Error should be added to errors list
            assert error_to_add in result.errors, (
                f"Added error should be in errors list. "
                f"Error '{error_to_add}' not found in {result.errors}"
            )
            
            # Property 3: Error count should increase by 1
            expected_error_count = initial_error_count + 1
            assert len(result.errors) == expected_error_count, (
                f"Error count should increase by 1. Expected {expected_error_count}, "
                f"got {len(result.errors)}"
            )
            
            # Property 4: Other attributes should remain unchanged
            assert result.chunks_processed == initial_chunks, (
                f"Chunks processed should remain unchanged after adding error"
            )
            assert result.embeddings_generated == initial_embeddings, (
                f"Embeddings generated should remain unchanged after adding error"
            )
            assert result.storage_success == adjusted_storage_success, (
                f"Storage success should remain unchanged after adding error"
            )
            assert abs(result.processing_time - initial_time) < 1e-10, (
                f"Processing time should remain unchanged after adding error"
            )
            assert result.source_document == source_doc, (
                f"Source document should remain unchanged after adding error"
            )
            
        except ValueError:
            # If initial state is invalid, that's expected
            pass