import sys
import os
import pytest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ai_generator

@pytest.fixture
def mock_genai_client():
    """Fixture to create a mock genai.Client."""
    return mock.MagicMock()

@pytest.fixture
def mock_tool_manager():
    """Fixture to create a mock ToolManager."""
    return mock.MagicMock()

@pytest.fixture
def ai_generator_instance(mock_genai_client):
    """Fixture to create an AIGenerator with a mock genai.Client."""
    with mock.patch('google.genai.Client', return_value=mock_genai_client):
        generator = ai_generator.AIGenerator(api_key="test_key", model="test_model")
    return generator

def test_generate_response_direct_answer(ai_generator_instance, mock_genai_client):
    """Test that generate_response returns a direct text answer when no tool is called."""
    query = "What is the capital of France?"
    
    # Mock the response from the Gemini API
    mock_response = mock.MagicMock()
    mock_response.text = "Paris"
    mock_response.candidates[0].content.parts = []
    mock_genai_client.models.generate_content.return_value = mock_response

    response = ai_generator_instance.generate_response(query)

    assert response == "Paris"
    mock_genai_client.models.generate_content.assert_called_once()

def test_generate_response_calls_tool(ai_generator_instance, mock_genai_client, mock_tool_manager):
    """Test that generate_response calls the tool manager when a tool call is in the response."""
    query = "Search for 'machine learning' in the 'AI' course"

    # Mock the initial response to include a function call
    mock_initial_response = mock.MagicMock()
    mock_function_call = mock.MagicMock()
    mock_function_call.name = "search_course_content"
    mock_function_call.args = {"query": "machine learning", "course_name": "AI"}
    
    mock_part = mock.MagicMock(function_call=mock_function_call)
    mock_initial_response.candidates[0].content.parts = [mock_part]
    
    # Mock the final response after tool execution
    mock_final_response = mock.MagicMock()
    mock_final_response.text = ("Here are the results for 'machine learning' in "
                                "'AI'.")
    
    # Set up the mock client to return the two responses in order
    mock_genai_client.models.generate_content.side_effect = [mock_initial_response, mock_final_response]

    # Mock the tool manager's execution result
    mock_tool_manager.execute_tool.return_value = "Search results content"

    ai_generator_instance.generate_response(query=query, tools=["search_course_content"], tool_manager=mock_tool_manager)

    # Assert that the tool manager was called correctly
    mock_tool_manager.execute_tool.assert_called_once_with(
        "search_course_content",
        query="machine learning",
        course_name="AI"
    )
    
    # Assert that generate_content was called twice
    assert mock_genai_client.models.generate_content.call_count == 2
