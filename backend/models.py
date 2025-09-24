from typing import List, Optional
import pydantic


class Lesson(pydantic.BaseModel):
    """Represents a lesson within a course"""
    lesson_number: int  # Sequential lesson number (1, 2, 3, etc.)
    title: str         # Lesson title
    lesson_link: Optional[str] = None  # URL link to the lesson


class Course(pydantic.BaseModel):
    """Represents a complete course with its lessons"""
    title: str                 # Full course title (used as unique identifier)
    course_link: Optional[str] = None  # URL link to the course
    instructor: Optional[str] = None  # Course instructor name (optional metadata)
    lessons: List[Lesson] = [] # List of lessons in this course


class CourseChunk(pydantic.BaseModel):
    """Represents a text chunk from a course for vector storage"""
    content: str                        # The actual text content
    course_title: str                   # Which course this chunk belongs to
    lesson_number: Optional[int] = None # Which lesson this chunk is from
    chunk_index: int                    # Position of this chunk in the document
