import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import OpenAI
import docx
from io import BytesIO
from typing import Optional
import asyncio
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="CodeWeaver API")

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ai-code-doc-helper.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not DEEPSEEK_API_KEY:
    raise ValueError("❌ DEEPSEEK_API_KEY not found. Make sure it's in your .env file or environment variables.")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found. Make sure it's in your .env file or environment variables.")
# Initialize OpenAI clients with custom base URLs
deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY ,
    base_url="https://api.deepseek.com"
)

groq_client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)


def extract_text_from_docx(file_content: bytes) -> str:
    """Extract text from .docx file"""
    try:
        doc = docx.Document(BytesIO(file_content))
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    except Exception as e:
        return f"Error reading file: {str(e)}"


async def stream_code_generation(description: str, file_content: Optional[str] = None):
    """Stream Python code generation using DeepSeek"""
    
    # Combine description and file content
    full_prompt = description
    if file_content:
        full_prompt = f"File content:\n{file_content}\n\nAdditional description:\n{description}"
    
    system_prompt = """You are an expert Python developer. Generate clean, well-structured, 
    production-ready Python code based on the user's requirements. Include:
    - Proper imports
    - Type hints
    - Docstrings
    - Error handling
    - Main execution block
    
    Only output the Python code, no explanations."""
    
    try:
        stream = deepseek_client.chat.completions.create(
            model="deepseek-chat",  # or "deepseek-coder"
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt}
            ],
            stream=True,
            temperature=0.7,
            max_tokens=4000
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                await asyncio.sleep(0.01)  # Small delay for smoother streaming
                
    except Exception as e:
        yield f"# Error generating code: {str(e)}\n"


async def stream_docs_generation(description: str, file_content: Optional[str] = None):
    """Stream documentation generation using Groq"""
    
    full_prompt = description
    if file_content:
        full_prompt = f"Project details from file:\n{file_content}\n\nAdditional info:\n{description}"
    
    system_prompt = """You are a technical documentation expert. Create comprehensive, 
    well-structured documentation in Markdown format. Include:
    - Project overview
    - Features
    - Installation instructions
    - Usage examples
    - API reference (if applicable)
    - Architecture overview
    - Contributing guidelines
    
    Use proper Markdown formatting with headers, code blocks, lists, and emphasis."""
    
    try:
        stream = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # or "mixtral-8x7b-32768"
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt}
            ],
            stream=True,
            temperature=0.7,
            max_tokens=8000
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                await asyncio.sleep(0.01)
                
    except Exception as e:
        yield f"# Error generating documentation\n\n{str(e)}\n"


@app.post("/api/generate-code")
async def generate_code(
    file: Optional[UploadFile] = File(None),
    description: str = Form(...)
):
    """Generate Python code using DeepSeek"""
    
    file_content = None
    if file:
        content = await file.read()
        if file.filename.endswith('.docx'):
            file_content = extract_text_from_docx(content)
        else:
            file_content = content.decode('utf-8', errors='ignore')
    
    return StreamingResponse(
        stream_code_generation(description, file_content),
        media_type="text/plain"
    )


@app.post("/api/generate-docs")
async def generate_docs(
    file: Optional[UploadFile] = File(None),
    description: str = Form(...)
):
    """Generate documentation using Groq"""
    
    file_content = None
    if file:
        content = await file.read()
        if file.filename.endswith('.docx'):
            file_content = extract_text_from_docx(content)
        else:
            file_content = content.decode('utf-8', errors='ignore')
    
    return StreamingResponse(
        stream_docs_generation(description, file_content),
        media_type="text/plain"
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "deepseek_configured": bool(os.getenv("DEEPSEEK_API_KEY")),
        "groq_configured": bool(os.getenv("GROQ_API_KEY"))
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)