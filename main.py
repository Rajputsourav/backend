# Run this file with: uvicorn main:app --reload 
# Run tis file if 1st command give error : python -m uvicorn main:app --reload


import json
import re
from pptx import Presentation
from fastapi.responses import FileResponse

from pathlib import Path
                                                                                                    
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


# =============================
# Environment
# =============================

load_dotenv()


# =============================
# Paths
# =============================

BASE_DIR = Path(__file__).resolve().parent.parent

KNOWLEDGE_PATH = BASE_DIR / "knowledge_base"

CHROMA_PATH = str(BASE_DIR / "chroma_db")

# =============================
# Embeddings
# =============================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)



# =============================
# Load ChromaDB
# =============================

db = Chroma(
    collection_name="rag_documents",
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH
)

retriever = db.as_retriever(
    search_kwargs={"k": 5}
)



# =============================
# Gemini
# =============================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0.3,
    max_retries=3
)

# =============================
# FastAPI
# =============================

app = FastAPI(
    title="SIH 26154 AI Content Transformation Platform"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# =============================
# Home
# =============================

@app.get("/")
def home():

    return {
        "message": "SIH 26154 AI Content Transformation API"
    }


# =============================
# Normal RAG Query
# =============================

class Query(BaseModel):

    text: str


@app.post("/query")
def query_rag(query: Query):

    documents = retriever.invoke(
        query.text
    )

    context = "\n\n".join(
        doc.page_content
        for doc in documents
    )

    prompt = f"""
You are a helpful document assistant.

Answer the user's question ONLY using the provided context.

If the context does not contain the answer, say:

"I don't know based on the provided document."

SOURCE CONTEXT:
{context}

USER QUESTION:
{query.text}
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


# =============================
# Document Upload
# =============================

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...)
):

    allowed_extensions = [
        ".pdf",
        ".txt"
    ]

    file_extension = Path(
        file.filename
    ).suffix.lower()

    # Check file type
    if file_extension not in allowed_extensions:

        return {
            "success": False,
            "message": "Only PDF and TXT files are currently supported."
        }

    # Create knowledge base folder
    KNOWLEDGE_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save uploaded file
    file_path = KNOWLEDGE_PATH / file.filename

    contents = await file.read()

    with open(file_path, "wb") as f:

        f.write(contents)

    # =============================
    # Load document
    # =============================

    if file_extension == ".pdf":

        loader = PyPDFLoader(
            str(file_path)
        )

    else:

        loader = TextLoader(
            str(file_path),
            encoding="utf-8"
        )

    documents = loader.load()

    # =============================
    # Split document
    # =============================

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(
        documents
    )

    # =============================
    # Add Documents to ChromaDB
    # =============================

    global retriever

    db.add_documents(chunks)

    retriever = db.as_retriever(
        search_kwargs={"k": 5}
    )

    # =============================
    # Return result
    # =============================

    return {
        "success": True,
        "filename": file.filename,
        "pages_or_sections": len(documents),
        "chunks": len(chunks),
        "message": "Document uploaded and ChromaDB updated successfully."
    }

# =============================
# Content Generation
# =============================

class GenerateRequest(BaseModel):

    instruction: str

    output_type: str = "Executive Summary"

    audience: str = "Executive"

    tone: str = "Professional"

    language: str = "English"

    length: str = "Medium"


@app.post("/generate")
def generate_content(
    request: GenerateRequest
):

    # Retrieve relevant document information
    documents = retriever.invoke(
        request.instruction
    )

    context = "\n\n".join(
        doc.page_content
        for doc in documents
    )

    # =============================
    # Output-specific instructions
    # =============================

    if request.output_type == "Executive Summary":

        output_instruction = """
Create an executive summary.

Include:

- Main topic
- Important findings
- Important facts
- Key conclusions
- Important recommendations if present in the source
"""


    elif request.output_type == "Social Media Post":
        output_instruction = """
    Create social media content based ONLY on the source document.

Create:
- LinkedIn Post
- X/Twitter Post
- Instagram Caption

Keep the information accurate and source-grounded.
Do not invent facts, statistics, names, dates, or claims.

Clearly separate the three formats.
"""


    elif request.output_type == "Infographic Content":
        output_instruction = """
Create structured infographic content based ONLY on the source document.

Use exactly this structure:

TITLE:
A short and engaging title.

KEY MESSAGE:
The main message of the document in 1-2 sentences.

KEY FACTS:
- Important fact 1
- Important fact 2
- Important fact 3

MAIN POINTS:
- Important point 1
- Important point 2
- Important point 3
- Important point 4

IMPACT:
Explain the important impact or significance mentioned in the source.

RECOMMENDATIONS:
List recommendations only if they are present in the source document.

CALL TO ACTION:
Provide a short action-oriented statement based only on the source.

IMPORTANT RULES:
- Use ONLY information from the source document.
- Do not invent statistics, facts, names, dates, or recommendations.
- Keep the content concise and suitable for an infographic.
- Clearly separate each section.
- Follow the requested audience, tone, language, and length.
"""


    elif request.output_type == "Video Script / Storyboard":
        output_instruction = """
Create a professional video script and storyboard based ONLY
on the source document.

Use exactly this structure:

TITLE:
A clear and engaging video title.

OBJECTIVE:
Explain the main objective of the video in 1-2 sentences.

TARGET AUDIENCE:
Describe the intended audience.

SCENE 1:
VISUAL:
Describe what should appear on screen.

VOICE-OVER:
Write the narration for this scene.

ON-SCREEN TEXT:
Write the important text that should appear on screen.

SCENE 2:
VISUAL:
Describe what should appear on screen.

VOICE-OVER:
Write the narration for this scene.

ON-SCREEN TEXT:
Write the important text that should appear on screen.

SCENE 3:
VISUAL:
Describe what should appear on screen.

VOICE-OVER:
Write the narration for this scene.

ON-SCREEN TEXT:
Write the important text that should appear on screen.

SCENE 4:
VISUAL:
Describe what should appear on screen.

VOICE-OVER:
Write the narration for this scene.

ON-SCREEN TEXT:
Write the important text that should appear on screen.

SCENE 5:
VISUAL:
Describe what should appear on screen.

VOICE-OVER:
Write the narration for this scene.

ON-SCREEN TEXT:
Write the important text that should appear on screen.

CALL TO ACTION:
Provide a short closing action based ONLY on the source document.

IMPORTANT RULES:
- Use ONLY information from the source document.
- Do not invent facts, statistics, names, dates, or claims.
- Do not add information from your own knowledge.
- Keep the storyboard practical and easy to convert into a video.
- Follow the requested audience, tone, language, and length.
"""




    elif request.output_type == "Security Advisory":

        output_instruction = """
Create a professional security advisory.

Use the following structure:

1. Title
2. Overview
3. Security Issue
4. Affected Systems or Components
5. Risk / Impact
6. Recommended Actions
7. Important Notes

Only include information that is actually available
in the source document.

If a particular detail is not available,
do not invent it.
"""

    else:

        output_instruction = f"""
Create the requested content type:

{request.output_type}
"""


    # =============================
    # Main AI Prompt
    # =============================

    prompt = f"""
You are an AI content transformation assistant.

Your job is to transform the SOURCE DOCUMENT
into the requested communication format.

IMPORTANT RULES:

1. Use ONLY information from the SOURCE DOCUMENT.
2. Never invent facts, numbers, names, dates,
   vulnerabilities, products or recommendations.
3. Preserve important facts from the source.
4. Do not add information from your own knowledge.
5. If information is missing, do not make it up.
6. The generated content must be clear and professional.
7. Do not mention that you are an AI.
8. Follow the requested audience, tone, language
   and length.

OUTPUT TYPE:
{request.output_type}

AUDIENCE:
{request.audience}

TONE:
{request.tone}

LANGUAGE:
{request.language}

LENGTH:
{request.length}

OUTPUT REQUIREMENTS:
{output_instruction}

SOURCE DOCUMENT:
{context}

USER INSTRUCTION:
{request.instruction}

Now generate the final content.
"""

    response = llm.invoke(
        prompt
    )

    return {
        "type": request.output_type,
        "audience": request.audience,
        "tone": request.tone,
        "language": request.language,
        "length": request.length,
        "content": response.content
    }





    # =============================
# PowerPoint Generation
# =============================

class PPTRequest(BaseModel):

    audience: str = "Executive"
    tone: str = "Professional"
    language: str = "English"
    number_of_slides: int = 6


@app.post("/generate-ppt")
def generate_ppt(request: PPTRequest):

    documents = retriever.invoke(
        "Create a presentation covering all important topics in the source document."
    )

    context = "\n\n".join(
        doc.page_content
        for doc in documents
    )

    prompt = f"""
You are an AI presentation generation assistant.

Create a PowerPoint presentation using ONLY
the information contained in the SOURCE DOCUMENT.

IMPORTANT RULES:

1. Do not invent facts.
2. Do not add information from your own knowledge.
3. Preserve important facts and concepts.
4. Create exactly {request.number_of_slides} slides.
5. Target audience: {request.audience}
6. Tone: {request.tone}
7. Language: {request.language}

Return ONLY valid JSON.

Use exactly this structure:

{{
    "title": "Presentation title",
    "slides": [
        {{
            "title": "Slide title",
            "bullets": [
                "Bullet point 1",
                "Bullet point 2",
                "Bullet point 3"
            ]
        }}
    ]
}}

SOURCE DOCUMENT:
{context}
"""

    response = llm.invoke(prompt)

    raw_content = response.content

    if isinstance(raw_content, list):

        text_parts = []

        for item in raw_content:

            if isinstance(item, dict):

                if item.get("type") == "text":

                    text_parts.append(
                        item.get("text", "")
                    )

        raw_content = "\n".join(text_parts)

    raw_content = raw_content.strip()

    raw_content = re.sub(
        r"^```json\s*",
        "",
        raw_content
    )

    raw_content = re.sub(
        r"\s*```$",
        "",
        raw_content
    )

    presentation_data = json.loads(
        raw_content
    )

    prs = Presentation()

    title_slide = prs.slides.add_slide(
        prs.slide_layouts[0]
    )

    title_slide.shapes.title.text = (
        presentation_data["title"]
    )

    title_slide.placeholders[1].text = (
        "AI Generated Presentation"
    )

    for slide_data in presentation_data["slides"]:

        slide = prs.slides.add_slide(
            prs.slide_layouts[1]
        )

        slide.shapes.title.text = (
            slide_data["title"]
        )

        text_frame = (
            slide.placeholders[1].text_frame
        )

        text_frame.clear()

        for i, bullet in enumerate(
            slide_data["bullets"]
        ):

            if i == 0:
                paragraph = text_frame.paragraphs[0]
            else:
                paragraph = text_frame.add_paragraph()

            paragraph.text = bullet
            paragraph.level = 0

    output_path = (
        BASE_DIR / "generated_presentation.pptx"
    )

    prs.save(output_path)

    return FileResponse(
        path=output_path,
        filename="SIH_26154_Presentation.pptx",
        media_type=(
            "application/vnd.openxmlformats-officedocument."
            "presentationml.presentation"
        )
    )





    # =============================
# Generate All Outputs
# =============================

@app.post("/generate-all")
def generate_all_outputs():

    documents = retriever.invoke(
        "Summarize and identify all important information from the source document."
    )

    context = "\n\n".join(
        doc.page_content
        for doc in documents
    )

    prompt = f"""
You are an AI content transformation assistant.

Transform the SOURCE DOCUMENT into multiple communication
formats.

Use ONLY information from the SOURCE DOCUMENT.

DO NOT:
- invent facts
- invent statistics
- invent names
- invent dates
- add outside knowledge
- create unsupported recommendations

Create the following six outputs:

1. EXECUTIVE SUMMARY

2. SECURITY ADVISORY

3. SOCIAL MEDIA POSTS
   - LinkedIn
   - X/Twitter
   - Instagram

4. INFOGRAPHIC CONTENT

5. VIDEO SCRIPT / STORYBOARD
   - Title
   - Objective
   - Target Audience
   - Scene 1 to Scene 5
   - Visual
   - Voice-over
   - On-screen text
   - Call to Action

Return ONLY valid JSON.

Use exactly this structure:

{{
    "executive_summary": "...",

    "security_advisory": "...",

    "social_media": {{
        "linkedin": "...",
        "twitter": "...",
        "instagram": "..."
    }},

    "infographic": "...",

    "video_storyboard": "..."
}}

SOURCE DOCUMENT:
{context}
"""

    response = llm.invoke(prompt)

    raw_content = response.content

    if isinstance(raw_content, list):
        text_parts = []

        for item in raw_content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))

        raw_content = "\n".join(text_parts)

    raw_content = raw_content.strip()

    raw_content = re.sub(
        r"^```json\s*",
        "",
        raw_content
    )

    raw_content = re.sub(
        r"\s*```$",
        "",
        raw_content
    )

    result = json.loads(raw_content)

    return result



 # =============================
# Complete Communication Package
# =============================

@app.post("/generate-complete")
def generate_complete_package():

    # Generate all text-based outputs
    documents = retriever.invoke(
        "Identify all important information from the source document."
    )

    context = "\n\n".join(
        doc.page_content
        for doc in documents
    )

    prompt = f"""
You are an AI content transformation assistant.

Transform the SOURCE DOCUMENT into a complete
communication package.

Use ONLY information from the SOURCE DOCUMENT.

Do not invent facts, statistics, names, dates,
recommendations, or other information.

Return ONLY valid JSON.

Use exactly this structure:

{{
    "executive_summary": "...",
    "security_advisory": "...",

    "social_media": {{
        "linkedin": "...",
        "twitter": "...",
        "instagram": "..."
    }},

    "infographic": "...",

    "video_storyboard": "..."
}}

SOURCE DOCUMENT:
{context}
"""

    response = llm.invoke(prompt)

    raw_content = response.content

    if isinstance(raw_content, list):
        text_parts = []

        for item in raw_content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))

        raw_content = "\n".join(text_parts)

    raw_content = raw_content.strip()

    raw_content = re.sub(
        r"^```json\s*",
        "",
        raw_content
    )

    raw_content = re.sub(
        r"\s*```$",
        "",
        raw_content
    )

    result = json.loads(raw_content)

    # =============================
    # Create PowerPoint
    # =============================

    ppt_prompt = f"""
Create a PowerPoint presentation from the SOURCE DOCUMENT.

Create exactly 6 slides.

Use ONLY information from the SOURCE DOCUMENT.

Return ONLY valid JSON.

Structure:

{{
    "title": "Presentation title",
    "slides": [
        {{
            "title": "Slide title",
            "bullets": [
                "Bullet 1",
                "Bullet 2",
                "Bullet 3"
            ]
        }}
    ]
}}

SOURCE DOCUMENT:
{context}
"""

    ppt_response = llm.invoke(ppt_prompt)

    ppt_content = ppt_response.content

    if isinstance(ppt_content, list):
        text_parts = []

        for item in ppt_content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))

        ppt_content = "\n".join(text_parts)

    ppt_content = ppt_content.strip()

    ppt_content = re.sub(
        r"^```json\s*",
        "",
        ppt_content
    )

    ppt_content = re.sub(
        r"\s*```$",
        "",
        ppt_content
    )

    presentation_data = json.loads(ppt_content)

    prs = Presentation()

    # Title slide
    title_slide = prs.slides.add_slide(
        prs.slide_layouts[0]
    )

    title_slide.shapes.title.text = (
        presentation_data["title"]
    )

    title_slide.placeholders[1].text = (
        "SIH 26154 AI Generated Presentation"
    )

    # Content slides
    for slide_data in presentation_data["slides"]:

        slide = prs.slides.add_slide(
            prs.slide_layouts[1]
        )

        slide.shapes.title.text = (
            slide_data["title"]
        )

        text_frame = slide.placeholders[1].text_frame
        text_frame.clear()

        for i, bullet in enumerate(
            slide_data["bullets"]
        ):

            if i == 0:
                paragraph = text_frame.paragraphs[0]
            else:
                paragraph = text_frame.add_paragraph()

            paragraph.text = bullet
            paragraph.level = 0

    output_path = BASE_DIR / "complete_communication_package.pptx"

    prs.save(output_path)

    return {
        "outputs": result,
        "presentation": "/download-presentation"
    }


@app.get("/download-presentation")
def download_presentation():

    output_path = BASE_DIR / "complete_communication_package.pptx"

    return FileResponse(
        path=output_path,
        filename="SIH_26154_Complete_Presentation.pptx",
        media_type=(
            "application/vnd.openxmlformats-officedocument."
            "presentationml.presentation"
        )
    )   