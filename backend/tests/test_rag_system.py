import sys
import os
import pytest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock config before importing RAGSystem
import config
mock_config = config.Config(GEMINI_API_KEY="test_key")

with mock.patch('rag_system.document_processor.DocumentProcessor'):
    with mock.patch('rag_system.vector_store.VectorStore'):
        with mock.patch('rag_system.ai_generator.AIGenerator'):
            with mock.patch('rag_system.session_manager.SessionManager'):
                from rag_system import RAGSystem


@pytest.fixture
def rag_system():
    """Fixture to create a RAGSystem with mocked components."""
    with mock.patch('rag_system.document_processor.DocumentProcessor') as mock_dp, \
         mock.patch('rag_system.vector_store.VectorStore') as mock_vs, \
         mock.patch('rag_system.ai_generator.AIGenerator') as mock_aig, \
         mock.patch('rag_system.session_manager.SessionManager') as mock_sm:
        
        system = RAGSystem(mock_config)
        # Attach mocks to the instance for easy access in tests
        system.document_processor = mock_dp
        system.vector_store = mock_vs
        system.ai_generator = mock_aig
        system.session_manager = mock_sm
        
        # Also mock the tool manager and its methods
        system.tool_manager = mock.MagicMock()
        
        yield system


def test_query_with_session_id(rag_system):
    """Test the query method with a session_id."""
    query = "test query"
    session_id = "test_session"
    expected_answer = "This is a test answer."
    expected_sources = [{"source": "source1", "link": "link1"}]

    # Mock the return values of the components
    rag_system.session_manager.get_conversation_history.return_value = "Previous conversation"
    rag_system.ai_generator.generate_response.return_value = expected_answer
    rag_system.tool_manager.get_last_sources.return_value = expected_sources

    answer, sources = rag_system.query(query, session_id)

    assert answer == expected_answer
    assert sources == expected_sources

    # Verify that the correct methods were called
    rag_system.session_manager.get_conversation_history.assert_called_once_with(session_id)
    rag_system.ai_generator.generate_response.assert_called_once()
    rag_system.tool_manager.get_last_sources.assert_called_once()
    rag_system.session_manager.add_exchange.assert_called_once_with(session_id, query, expected_answer)

def test_query_without_session_id(rag_system):
    """Test the query method without a session_id."""
    query = "test query"
    expected_answer = "This is a test answer."
    expected_sources = []

    # Mock the return values
    rag_system.ai_generator.generate_response.return_value = expected_answer
    rag_system.tool_manager.get_last_sources.return_value = expected_sources

    answer, sources = rag_system.query(query)

    assert answer == expected_answer
    assert sources == expected_sources

    # Verify that session history was not requested, but a new session was not created here.
    # In the actual app, the session is created in the endpoint.
    rag_system.session_manager.get_conversation_history.assert_not_called()
    
    # Verify that add_exchange was not called since there is no session_id
    rag_system.session_manager.add_exchange.assert_not_called()

def test_add_course_document(rag_system):
    """Test adding a single course document."""
    file_path = "path/to/doc.txt"
    mock_course = mock.MagicMock()
    mock_chunks = [mock.MagicMock(), mock.MagicMock()]

    rag_system.document_processor.process_course_document.return_value = (mock_course, mock_chunks)

    course, num_chunks = rag_system.add_course_document(file_path)

    assert course == mock_course
    assert num_chunks == len(mock_chunks)
    rag_system.document_processor.process_course_document.assert_called_once_with(file_path)
    rag_system.vector_store.add_course_metadata.assert_called_once_with(mock_course)
    rag_system.vector_store.add_course_content.assert_called_once_with(mock_chunks)
