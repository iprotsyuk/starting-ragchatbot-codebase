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

def test_generate_response_single_tool_call(ai_generator_instance, mock_genai_client, mock_tool_manager):
    """Test that generate_response calls a tool and returns the final answer."""
    query = "Search for 'machine learning' in the 'AI' course"

    # Mock the initial response to include a function call
    mock_initial_response = mock.MagicMock()
    mock_function_call = mock.MagicMock()
    mock_function_call.name = "search_course_content"
    mock_function_call.args = {"query": "machine learning", "course_name": "AI"}
    
    mock_part = mock.MagicMock()
    mock_part.function_call = mock_function_call
    mock_initial_response.candidates[0].content.parts = [mock_part]
    
    # Mock the final response after tool execution
    mock_final_response = mock.MagicMock()
    mock_final_response.text = ("Here are the results for 'machine learning' in "
                                "'AI'.")
    mock_final_response.candidates[0].content.parts = []
    
    # Set up the mock client to return the two responses in order
    mock_genai_client.models.generate_content.side_effect = [mock_initial_response, mock_final_response]

    # Mock the tool manager's execution result
    mock_tool_manager.execute_tool.return_value = "Search results content"

    response = ai_generator_instance.generate_response(query=query, tools=["search_course_content"], tool_manager=mock_tool_manager)

    # Assert that the tool manager was called correctly
    mock_tool_manager.execute_tool.assert_called_once_with(
        "search_course_content",
        query="machine learning",
        course_name="AI"
    )
    
    # Assert that generate_content was called twice
    assert mock_genai_client.models.generate_content.call_count == 2
    assert response == "Here are the results for 'machine learning' in 'AI'."

def test_generate_response_two_sequential_tool_calls(ai_generator_instance, mock_genai_client, mock_tool_manager):
    """Test that generate_response can handle two sequential tool calls."""
    query = "Compare 'neural networks' in 'AI' and 'Deep Learning' courses."

    # Mock first response with a tool call
    mock_response_1 = mock.MagicMock()
    mock_function_call_1 = mock.MagicMock()
    mock_function_call_1.name = "search_course_content"
    mock_function_call_1.args = {"query": "neural networks", "course_name": "AI"}
    mock_part_1 = mock.MagicMock()
    mock_part_1.function_call = mock_function_call_1
    mock_response_1.candidates[0].content.parts = [mock_part_1]
    mock_response_1.candidates[0].content.role = "assistant"

    # Mock second response with another tool call
    mock_response_2 = mock.MagicMock()
    mock_function_call_2 = mock.MagicMock()
    mock_function_call_2.name = "search_course_content"
    mock_function_call_2.args = {"query": "neural networks", "course_name": "Deep Learning"}
    mock_part_2 = mock.MagicMock()
    mock_part_2.function_call = mock_function_call_2
    mock_response_2.candidates[0].content.parts = [mock_part_2]
    mock_response_2.candidates[0].content.role = "assistant"

    # Mock final response with text
    mock_final_response = mock.MagicMock()
    mock_final_response.text = "Comparison of neural networks..."
    mock_final_response.candidates[0].content.parts = []

    mock_genai_client.models.generate_content.side_effect = [mock_response_1, mock_response_2, mock_final_response]
    mock_tool_manager.execute_tool.side_effect = ["AI course content", "Deep Learning course content"]

    response = ai_generator_instance.generate_response(query=query, tools=["search_course_content"], tool_manager=mock_tool_manager)

    assert mock_tool_manager.execute_tool.call_count == 2
    assert mock_genai_client.models.generate_content.call_count == 3
    assert response == "Comparison of neural networks..."

def test_generate_response_tool_failure(ai_generator_instance, mock_genai_client, mock_tool_manager):
    """Test that generate_response handles tool execution failure gracefully."""
    query = "Search for something that will fail."

    # Mock response with a tool call
    mock_response_with_tool = mock.MagicMock()
    mock_function_call = mock.MagicMock()
    mock_function_call.name = "search_course_content"
    mock_function_call.args = {"query": "fail"}
    mock_part = mock.MagicMock()
    mock_part.function_call = mock_function_call
    mock_response_with_tool.candidates[0].content.parts = [mock_part]

    mock_genai_client.models.generate_content.return_value = mock_response_with_tool
    mock_tool_manager.execute_tool.side_effect = Exception("Tool failed!")

    response = ai_generator_instance.generate_response(query=query, tools=["search_course_content"], tool_manager=mock_tool_manager)

    assert response == "An error occurred while executing the tool."
    mock_tool_manager.execute_tool.assert_called_once()
    mock_genai_client.models.generate_content.assert_called_once()
