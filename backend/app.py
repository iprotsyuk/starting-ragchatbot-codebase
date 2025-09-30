import contextlib
import logging
import os
from typing import List, Optional
import warnings

import fastapi
from fastapi.middleware import cors
from fastapi import staticfiles
from fastapi import responses
from fastapi.middleware import trustedhost
import pydantic

from config import config
from rag_system import RAGSystem

warnings.filterwarnings("ignore", message="resource_tracker: There appear to be.*")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@contextlib.asynccontextmanager
async def lifespan(_: fastapi.FastAPI):
    """Load initial documents on startup"""
    docs_path = "../docs"
    if os.path.exists(docs_path):
        logger.info("Loading initial documents...")
        try:
            courses, chunks = rag_system.add_course_folder(docs_path, clear_existing=False)
            logger.info(f"Loaded {courses} courses with {chunks} chunks")
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
    yield


# Initialize FastAPI app
app = fastapi.FastAPI(
    title="Course Materials RAG System",
    root_path="",
    lifespan=lifespan,
)

# Add trusted host middleware for proxy
app.add_middleware(
    trustedhost.TrustedHostMiddleware,
    allowed_hosts=["*"]
)

# Enable CORS with proper settings for proxy
app.add_middleware(
    cors.CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Initialize RAG system
rag_system = RAGSystem(config)

# Pydantic models for request/response
class QueryRequest(pydantic.BaseModel):
    """Request model for course queries"""
    query: str
    session_id: Optional[str] = None


class QueryResponse(pydantic.BaseModel):
    """Response model for course queries"""
    answer: str
    sources: List[dict]
    session_id: str


class CourseStats(pydantic.BaseModel):
    """Response model for course statistics"""
    total_courses: int
    course_titles: List[str]

# API Endpoints

@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Process a query and return response with sources"""
    try:
        # Create session if not provided
        session_id = request.session_id
        if not session_id:
            session_id = rag_system.session_manager.create_session()
        
        # Process query using RAG system
        answer, sources = rag_system.query(request.query, session_id)
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            session_id=session_id
        )
    except Exception as e:
        raise fastapi.HTTPException(status_code=500, detail=str(e))


@app.get("/api/courses", response_model=CourseStats)
async def get_course_stats():
    """Get course analytics and statistics"""
    try:
        analytics = rag_system.get_course_analytics()
        return CourseStats(
            total_courses=analytics["total_courses"],
            course_titles=analytics["course_titles"]
        )
    except Exception as e:
        raise fastapi.HTTPException(status_code=500, detail=str(e))


class DevStaticFiles(staticfiles.StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if isinstance(response, responses.FileResponse):
            # Add no-cache headers for development
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response
    
    
# Serve static files for the frontend
app.mount(
    "/",
    staticfiles.StaticFiles(directory="../frontend", html=True),
    name="static",
)
