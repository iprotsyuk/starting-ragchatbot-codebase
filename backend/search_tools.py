import abc
import logging
from typing import Optional

from google import genai
import vector_store

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Tool(abc.ABC):
    """Abstract base class for all tools"""
    
    @abc.abstractmethod
    def get_tool_definition(self) -> genai.types.Tool:
        """Return Gemini tool definition for this tool"""
        pass
    
    @abc.abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters"""
        pass


class CourseSearchTool(Tool):
    """Tool for searching course content with semantic course name matching"""
    
    def __init__(self, vector_store: vector_store.VectorStore):
        self.vector_store = vector_store
        self.last_sources = []

    def get_tool_definition(self) -> genai.types.Tool:
        """Return Gemini tool definition for this tool"""
        return genai.types.Tool(
            function_declarations=[{
                "name": "search_course_content",
                "description": "Search course materials with smart course name matching and lesson filtering",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string", 
                            "description": "What to search for in the course content"
                        },
                        "course_name": {
                            "type": "string",
                            "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')"
                        },
                        "lesson_number": {
                            "type": "integer",
                            "description": "Specific lesson number to search within (e.g. 1, 2, 3)"
                        }
                    },
                    "required": ["query"]
                }
            }]
        )
    
    def execute(self, query: str, course_name: Optional[str] = None, lesson_number: Optional[int] = None) -> str:
        """
        Execute the course search with the given query and optional filters.
        
        Args:
            query: The search query string.
            course_name: Optional course name to filter results.
            lesson_number: Optional lesson number to filter results.
            
        Returns:
            A formatted string of search results.
        """
        logger.info(f"Executing CourseSearchTool with query: '{query}', course_name: '{course_name}', lesson_number: '{lesson_number}'")
        results = self.vector_store.search(query, course_name, lesson_number)
        return self._format_results(results)

    def _format_results(self, results: vector_store.SearchResults) -> str:
        """Format search results with course and lesson context"""
        formatted = []
        sources = []  # Track sources for the UI
        
        for doc, meta in zip(results.documents, results.metadata):
            course_title = meta.get('course_title', 'unknown')
            lesson_num = meta.get('lesson_number')
            lesson_link = meta.get('lesson_link')
            
            # Build context header
            header = f"[{course_title}"
            if lesson_num is not None:
                header += f" - Lesson {lesson_num}"
            header += "]"
            
            # Track source for the UI
            source_text = course_title
            if lesson_num is not None:
                source_text += f" - Lesson {lesson_num}"
            
            sources.append({
                "source": source_text,
                "link": lesson_link
            })
            
            formatted.append(f"{header}\n{doc}")
        
        # Store sources for retrieval
        self.last_sources = sources
        
        return "\n\n".join(formatted)


class CourseOutlineTool(Tool):
    """Tool for fetching the outline of a course."""

    def __init__(self, vector_store: vector_store.VectorStore):
        self.vector_store = vector_store

    def get_tool_definition(self) -> genai.types.Tool:
        """Return Gemini tool definition for this tool"""
        return genai.types.Tool(
            function_declarations=[{
                "name": "get_course_outline",
                "description": "Get the outline of a course, including title, link, and all lesson titles.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "course_name": {
                            "type": "string",
                            "description": "The title of the course to get the outline for (e.g. 'MCP', 'Introduction')"
                        }
                    },
                    "required": ["course_name"]
                }
            }]
        )

    def execute(self, course_name: str) -> str:
        """
        Execute the course outline search.

        Args:
            course_name: The course name to get the outline for.

        Returns:
            A formatted string of the course outline.
        """
        import json
        logger.info(f"Executing CourseOutlineTool with course_name: '{course_name}'")

        # First, resolve the course name to get the exact title
        exact_course_title = self.vector_store._resolve_course_name(course_name)
        if not exact_course_title:
            return f"Could not find a course named '{course_name}'."

        # Get the course metadata from the course_catalog collection
        try:
            course_meta_results = self.vector_store.course_catalog.get(ids=[exact_course_title])
            if not course_meta_results or not course_meta_results['metadatas']:
                return f"Could not retrieve metadata for course '{exact_course_title}'."
        except Exception as e:
            logger.error(f"Error fetching course metadata: {e}")
            return "An error occurred while fetching course details."

        # Extract and format the outline
        metadata = course_meta_results['metadatas'][0]
        course_title = metadata.get('title', 'Unknown Course')
        course_link = metadata.get('course_link', 'Unknown Link')
        lessons_json = metadata.get('lessons_json')

        if not lessons_json:
            return f"No lessons found for course '{course_title}'."

        try:
            lessons = json.loads(lessons_json)
            if not lessons:
                return f"No lessons listed for course '{course_title}'."
        except json.JSONDecodeError:
            return "Error parsing lesson data."

        # Sort lessons by lesson number
        sorted_lessons = sorted(lessons, key=lambda x: x.get('lesson_number', 0))

        formatted = [
            f"Course: {course_title}",
            f"Link: {course_link}",
            "Lessons:"
        ]
        for lesson in sorted_lessons:
            lesson_num = lesson.get('lesson_number')
            lesson_title = lesson.get('lesson_title')
            if lesson_num is not None and lesson_title:
                formatted.append(f"  - Lesson {lesson_num}: {lesson_title}")

        return "\n".join(formatted)

    def _format_outline_from_results(self, results: vector_store.SearchResults) -> str:
        """Format search results into a course outline."""
        if not results.metadata:
            return "Could not find an outline for the specified course."

        # We can get course title and link from the first result's metadata
        first_meta = results.metadata[0]
        course_title = first_meta.get('course_title', 'Unknown Course')
        course_link = first_meta.get('course_link', 'Unknown Link')

        lessons = {}
        for meta in results.metadata:
            lesson_num = meta.get('lesson_number')
            lesson_title = meta.get('lesson_title')
            if lesson_num is not None and lesson_title is not None:
                # Use a dict to store unique lessons
                lessons[lesson_num] = lesson_title

        if not lessons:
            return f"No lessons found for course '{course_title}'."

        # Sort lessons by lesson number
        sorted_lessons = sorted(lessons.items())

        formatted = [
            f"Course: {course_title}",
            f"Link: {course_link}",
            "Lessons:"
        ]
        for lesson_num, lesson_title in sorted_lessons:
            formatted.append(f"  - Lesson {lesson_num}: {lesson_title}")

        return "\n".join(formatted)


class ToolManager:
    """Manages available tools for the AI"""
    
    def __init__(self):
        self.tools = {}
    
    def register_tool(self, tool: Tool):
        """Register any tool that implements the Tool interface"""
        tool_def = tool.get_tool_definition()
        tool_name = tool_def.function_declarations[0].name
        if not tool_name:
            raise ValueError("Tool must have a 'name' in its definition")
        self.tools[tool_name] = tool

    
    def get_tool_definitions(self) -> list:
        """Get all tool definitions for Gemini tool calling"""
        return [tool.get_tool_definition() for tool in self.tools.values()]
    
    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name with given parameters"""
        if tool_name not in self.tools:
            return f"Tool '{tool_name}' not found"
        
        return self.tools[tool_name].execute(**kwargs)
    
    def get_last_sources(self) -> list:
        """Get sources from the last search operation"""
        # Check all tools for last_sources attribute
        for tool in self.tools.values():
            if hasattr(tool, 'last_sources') and tool.last_sources:
                return tool.last_sources
        return []

    def reset_sources(self):
        """Reset sources from all tools that track sources"""
        for tool in self.tools.values():
            if hasattr(tool, 'last_sources'):
                tool.last_sources = []