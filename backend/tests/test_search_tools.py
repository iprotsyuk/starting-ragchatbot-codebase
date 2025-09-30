import sys
import os
import pytest
from unittest import mock

# Add backend to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import search_tools
import vector_store

@pytest.fixture
def mock_vector_store():
    """Fixture to create a mock VectorStore."""
    return mock.MagicMock()

@pytest.fixture
def course_search_tool(mock_vector_store):
    """Fixture to create a CourseSearchTool with a mock VectorStore."""
    return search_tools.CourseSearchTool(mock_vector_store)

def test_execute_with_query_only(course_search_tool, mock_vector_store):
    """Test that execute calls vector_store.search with only a query."""
    query = "test query"
    
    mock_vector_store.search.return_value = vector_store.SearchResults(
        documents=["doc1"],
        metadata=[{"course_title": "course1", "lesson_number": 1, "lesson_link": "link1"}],
        distances=[0.1]
    )

    result = course_search_tool.execute(query=query)

    mock_vector_store.search.assert_called_once_with(query, None, None)
    assert "course1" in result
    assert "Lesson 1" in result
    assert "doc1" in result
    assert len(course_search_tool.last_sources) == 1
    assert course_search_tool.last_sources[0]["source"] == "course1 - Lesson 1"

def test_execute_with_course_name(course_search_tool, mock_vector_store):
    """Test that execute calls vector_store.search with a query and course_name."""
    query = "test query"
    course_name = "course1"
    
    mock_vector_store.search.return_value = vector_store.SearchResults(
        documents=["doc1"],
        metadata=[{"course_title": "course1", "lesson_number": 1, "lesson_link": "link1"}],
        distances=[0.1]
    )

    course_search_tool.execute(query=query, course_name=course_name)

    mock_vector_store.search.assert_called_once_with(query, course_name, None)

def test_execute_with_lesson_number(course_search_tool, mock_vector_store):
    """Test that execute calls vector_store.search with a query and lesson_number."""
    query = "test query"
    lesson_number = 1

    mock_vector_store.search.return_value = vector_store.SearchResults(
        documents=["doc1"],
        metadata=[{"course_title": "course1", "lesson_number": 1, "lesson_link": "link1"}],
        distances=[0.1]
    )

    course_search_tool.execute(query=query, lesson_number=lesson_number)

    mock_vector_store.search.assert_called_once_with(query, None, lesson_number)

def test_format_results(course_search_tool):
    """Test the _format_results method directly."""
    results = vector_store.SearchResults(
        documents=["doc1", "doc2"],
        metadata=[
            {"course_title": "course1", "lesson_number": 1, "lesson_link": "link1"},
            {"course_title": "course2", "lesson_number": 2, "lesson_link": "link2"}
        ],
        distances=[0.1, 0.2]
    )

    formatted_string = course_search_tool._format_results(results)
    
    expected_string = "[course1 - Lesson 1]\ndoc1\n\n[course2 - Lesson 2]\ndoc2"
    assert formatted_string == expected_string
    
    assert len(course_search_tool.last_sources) == 2
    assert course_search_tool.last_sources[0] == {"source": "course1 - Lesson 1", "link": "link1"}
    assert course_search_tool.last_sources[1] == {"source": "course2 - Lesson 2", "link": "link2"}

def test_execute_with_no_results(course_search_tool, mock_vector_store):
    """Test execute when the vector store returns no results."""
    query = "non-existent query"
    
    mock_vector_store.search.return_value = vector_store.SearchResults.empty("No results found")

    result = course_search_tool.execute(query=query)

    assert result == ""
    assert len(course_search_tool.last_sources) == 0
