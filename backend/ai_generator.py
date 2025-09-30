import logging
from typing import List, Optional

from google import genai

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AIGenerator:
    """Handles interactions with Gemini API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool `search_course_content` **only** for questions about specific course content or detailed educational materials.
- For queries about a course outline, use the `get_course_outline` tool. When you do, return the course title, link, and the number and title of each lesson.
- **One search per query maximum**
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = genai.Client(api_key=api_key)
        self.model = model
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Prepare API call parameters efficiently
        config = {
            "temperature": 0,
            "max_output_tokens": 800
        }
        if tools:
            config["tools"] = tools
        
        # Get response from Gemini
        history = []
        if conversation_history:
            history.append({"role": "user", "parts": [{"text": conversation_history}]})
        history.append({"role": "user", "parts": [{"text": self.SYSTEM_PROMPT}]})
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=history + [{"role": "user", "parts": [{"text": query}]}],
            config=config
        )

        # Handle tool execution if needed
        if (
            response.candidates[0].content.parts
            and response.candidates[0].content.parts[0].function_call
            and tool_manager
        ):
            return self._handle_tool_execution(query, response, tool_manager)
        
        # Return direct response
        return response.text
    
    def _handle_tool_execution(self, user_query, initial_response, tool_manager):
        """
        Handle execution of tool calls and get follow-up response.
        
        Args:
            user_query: The original user query string
            initial_response: The response containing tool use requests
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        # Prepare API call parameters efficiently
        config = {
            "temperature": 0,
            "max_output_tokens": 800
        }
        # Build contents for follow-up call: user -> assistant (function_call) -> function (tool result)
        initial_parts = initial_response.candidates[0].content.parts
        contents = [
            {"role": "user", "parts": [{"text": user_query}]},
            {"role": "assistant", "parts": initial_parts}
        ]

        # Execute all tool calls and collect results
        for part in initial_parts:
            if part.function_call:
                tool_response = tool_manager.execute_tool(
                    part.function_call.name,
                    **part.function_call.args
                )
                contents.append({
                    "role": "function",
                    "parts": [{
                        "function_response": {
                            "name": part.function_call.name,
                            "response": {"result": tool_response}
                        }
                    }]
                })

        # Get final response
        final_response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config
        )
        return final_response.text
