import os
import io
import sys
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv  # Used to load the .env file
from PIL import Image
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi import Request
from google.protobuf.json_format import MessageToDict 
from base64 import b64decode
import json
from datetime import datetime
from bson import ObjectId
from db import history_collection

# =========================================================
# 1. INITIALIZATION & API KEY CONFIGURATION
# =========================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load environment variables from the .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Error: GEMINI_API_KEY not found in .env file.")
    sys.exit("API Key not found. Please create a .env file and add your key.")

# Configure the Gemini API client
try:
    genai.configure(api_key=api_key)
    print("Gemini API configured successfully.")
except Exception as e:
    print(f"Error configuring API: {e}")
    sys.exit("Configuration failed.")

# =========================================================
# 2. CREATE THE GEMINI MODEL & FASTAPI APP
# =========================================================

# Create the FastAPI application (already created above)
app.title = "Gemini AI Image Analysis API"
app.description = "An API to analyze uploaded images using the Gemini model."

# Initialize the generative model
# Keep a single model instance for reuse. Adjust model_name as needed.
try:
    gemini_model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",  # adjust as required
        generation_config={"temperature": 0.4},
    )
    print("Gemini model (gemini-2.5-flash) initialized.")
except Exception as e:
    print(f"Error creating model: {e}")
    sys.exit("Model initialization failed.")

# =========================================================
# 3. DEFINE THE API ENDPOINT
# =========================================================


# @app.post("/analyze-image/")
# async def analyze_image_endpoint(
#     file: UploadFile = File(..., description="The image file to analyze."),
#     prompt: str = Form(
#         "Analyze this pediatric chest X-ray. Describe the key findings in the lung fields, heart, and mediastinum.",
#         description="The specific prompt for the AI.",
#     ),
# ):
#     """
#     Analyzes an uploaded image file with a given text prompt.
#     """
#     print(f"Received request for image: {file.filename}")

#     # --- 1. Validate Image ---
#     # Check if the uploaded file is an image
#     if not file.content_type.startswith("image/"):
#         raise HTTPException(status_code=400, detail="File provided is not an image.")

#     try:
#         # --- 2. Process Image ---
#         # Read the file contents into bytes
#         image_bytes = await file.read()

#         # Open the image using Pillow from the in-memory bytes
#         img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

#         # --- 3. Prepare Content for Gemini ---
#         # The model can accept a list containing text and image objects
#         contents = [prompt, img]

#     except Exception as e:
#         print(f"Error processing image: {e}")
#         raise HTTPException(status_code=400, detail=f"Failed to read or open image: {str(e)}")

#     # --- 4. Call Gemini API ---
#     print("Sending request to Gemini API...")
#     try:
#         response = gemini_model.generate_content(contents=contents)
#         # response object structure can vary depending on the client lib version.
#         # For safety, try to return useful fields.
#         resp_text = None
#         try:
#             resp_text = response.text
#         except Exception:
#             # try nested candidates structure
#             resp_text = (
#                 response.candidates[0].content.parts[0].text
#                 if getattr(response, "candidates", None)
#                 else None
#             )

#         print("Received successful response from Gemini.")
#         return JSONResponse(content={"analysis": resp_text, "raw": response})

#     except Exception as e:
#         print(f"Error during Gemini API call: {e}")
#         raise HTTPException(status_code=500, detail=f"An error occurred with the Gemini API: {str(e)}")

@app.post("/analyze-image/")
async def analyze_image_endpoint(
    file: UploadFile = File(..., description="The image file to analyze."),
    prompt: str = Form(
        "Analyze this pediatric chest X-ray. Describe the key findings in the lung fields, heart, and mediastinum.",
        description="The specific prompt for the AI.",
    ),
):
    """
    Analyzes an uploaded image file and requests a very long, exhaustive clinical report.
    The system instruction is inserted into the contents list so it works with your client library.
    Adjust `max_output_tokens` as needed (higher = longer response; watch token limits and billing).
    """
    # Validate file
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read or open image: {str(e)}")

    # Strong system instruction demanding a very long, comprehensive report
    system_instruction = {
        "parts": [
            {
                "text": (
                    "You are a senior pediatric radiologist producing a formal diagnostic report. "
                    "Generate a comprehensive, highly detailed clinical radiology report based ONLY on the provided image and the prompt. "
                    "Structure the report with clear sections and headings including: Clinical question/hypothesis, Technique, Image Quality, "
                    "Detailed Findings (organized by anatomic region and side), Differential Diagnosis (ranked by likelihood with reasoning), "
                    "Impression (numbered prioritized findings), Recommendations (further imaging, clinical correlation, follow-up interval), "
                    "Teaching points, and Reference-level explanations for each major imaging appearance. "
                    "Include measured descriptions where appropriate (e.g., cardiothoracic ratio), describe distribution, symmetry, degree/severity, "
                    "and emphasize subtle findings. Provide possible pitfalls and explain how features support or refute key differentials. "
                    "Use clinical radiology language appropriate for a specialist audience. Be thorough — produce a long, exhaustive report (several paragraphs per section if needed). "
                    "Do NOT be concise; expand on reasoning and explanations. Do NOT hallucinate clinical history — if none is provided, state that history is not available. "
                    "Cite imaging signs (e.g., air bronchograms, reticular opacities) and explain their significance."
                )
            }
        ]
    }

    # Compose a thorough user prompt that includes the original prompt and indicates the image is attached.
    # You can add more clinical context to `prompt` if you capture it from user input.
    user_prompt_text = (
        f"{prompt}\n\n"
        "Attached: one chest radiograph (pediatric). Provide a thorough and exhaustive report as per the system instruction."
    )
    user_content = {"parts": [{"text": user_prompt_text}]}

    # Build contents list: system instruction first (client library compatible), then the textual prompt and the image object.
    contents = [system_instruction, user_content, img]

    # generation_config: increase max_output_tokens for longer output. Adjust as necessary.
    # NOTE: token usage and model limits vary; set conservatively or test iteratively.
    generation_config = {
        "temperature": 0.2,          # low temperature for more deterministic medical text; raise if you want more variability
        "max_output_tokens": 2048,   # increase for longer responses; change to 4096+ if your account & model support it
        # optionally: "top_p": 0.95, "candidate_count": 1
    }

    kwargs = {"contents": contents, "generation_config": generation_config}

    try:
        # Call model
        response = gemini_model.generate_content(**kwargs)

        # Convert response to serializable dict safely:
        resp_serializable = None
        if hasattr(response, "to_dict"):
            try:
                resp_serializable = response.to_dict()
            except Exception:
                resp_serializable = None

        if resp_serializable is None and hasattr(response, "_pb"):
            try:
                resp_serializable = MessageToDict(response._pb, preserving_proto_field_name=True)
            except Exception:
                resp_serializable = None

        # Best-effort extraction of text content
        text_answer = None
        if resp_serializable:
            # Try top-level text
            text_answer = resp_serializable.get("text")
            if not text_answer:
                # Try candidates -> content -> parts -> text
                try:
                    candidates = resp_serializable.get("candidates", [])
                    if candidates and isinstance(candidates, list):
                        first = candidates[0]
                        content = first.get("content", {})
                        parts = content.get("parts", [])
                        if parts and isinstance(parts, list):
                            text_answer = parts[0].get("text")
                except Exception:
                    text_answer = None
        else:
            # Fallback attribute access
            text_answer = getattr(response, "text", None)
            if not text_answer:
                try:
                    text_answer = response.candidates[0].content.parts[0].text
                except Exception:
                    text_answer = None

        # Build final JSON result. Include full serializable raw_response for debugging/inspection.
        result = {"analysis": text_answer}
        if resp_serializable:
            result["raw_response"] = resp_serializable
        else:
            # If no full serializable representation, still try to include limited candidates safely
            try:
                candidates = getattr(response, "candidates", None)
                result["candidates"] = candidates
            except Exception:
                result["candidates"] = None

        return JSONResponse(content=result)

    except Exception as e:
        # Helpful error for debugging
        print(f"Error during Gemini API call in analyze-image: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred with the Gemini API: {str(e)}")


@app.get("/", summary="Root Endpoint")
def read_root():
    """
    A simple root endpoint to confirm the server is running.
    """
    return {"message": "Welcome to the Gemini API Backend. Go to /docs to test the API."}


class QuestionRequest(BaseModel):
    question: str = ""
    report: str = ""


@app.post("/ask/")
async def ask_question(request: Request):
    """
    Accepts payload from frontend and forwards to gemini_model.generate_content.
    This version ensures the Gemini response is converted to plain JSON-serializable
    structures (handles protobuf RepeatedComposite / message objects).
    """
    try:
        data = await request.json()
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {str(e)}"})

    contents = data.get("contents")
    system_instruction = data.get("systemInstruction") or data.get("system_instruction")
    generation_config = data.get("generationConfig") or data.get("generation_config") or None

    # Backwards compatibility: accept single prompt string
    if contents is None:
        prompt = data.get("prompt") or data.get("question") or ""
        if not prompt:
            return JSONResponse(status_code=400, content={"error": "No prompt/contents provided."})
        contents = [prompt]

    # Merge systemInstruction into contents if provided
    if system_instruction:
        try:
            exists = any(isinstance(c, dict) and c == system_instruction for c in contents)
        except Exception:
            exists = False
        if not exists:
            contents = [system_instruction] + list(contents)

    kwargs = {"contents": contents}
    if generation_config:
        kwargs["generation_config"] = generation_config

    try:
        # Call the pre-initialized model
        response = gemini_model.generate_content(**kwargs)

        # Approach:
        # 1) Try response.to_dict() if available (client-provided helper)
        # 2) Fallback to protobuf MessageToDict on underlying ._pb if present
        # 3) If still not available, try to extract the common fields manually

        resp_serializable = None
        if hasattr(response, "to_dict"):
            try:
                resp_serializable = response.to_dict()
            except Exception:
                resp_serializable = None

        if resp_serializable is None and hasattr(response, "_pb"):
            try:
                # Convert protobuf message to a plain dict (serializable)
                resp_serializable = MessageToDict(response._pb, preserving_proto_field_name=True)
            except Exception:
                resp_serializable = None

        # Manual extraction as last resort
        text = None
        candidates = None
        if resp_serializable:
            # Best effort: find text and candidates in the dict
            candidates = resp_serializable.get("candidates") or resp_serializable.get("candidate") or None
            # try common top-level 'text'
            text = resp_serializable.get("text") or None
            # If no top-level text, try nested candidates -> content -> parts -> text
            if not text and candidates:
                try:
                    first = candidates[0]
                    content = first.get("content", {})
                    parts = content.get("parts", [])
                    if parts and isinstance(parts, list):
                        text = parts[0].get("text")
                except Exception:
                    text = None
        else:
            # Try to extract from response object attributes without serializing whole object
            try:
                # Some client versions expose .text
                text = getattr(response, "text", None)
            except Exception:
                text = None

            try:
                candidates = getattr(response, "candidates", None)
                # If candidates is a protobuf RepeatedComposite, convert it using MessageToDict
                if candidates is not None and not isinstance(candidates, (list, tuple)):
                    if hasattr(candidates, "_pb"):  # unlikely, but guarded
                        candidates = MessageToDict(candidates._pb)
            except Exception:
                candidates = None

            # If text still None, try nested attribute access (may raise)
            if not text:
                try:
                    text = response.candidates[0].content.parts[0].text
                except Exception:
                    text = None

        # Build final safe payload
        result = {"text": text, "candidates": candidates}

        # If we were able to get the full serializable dict, include it for debugging
        if resp_serializable:
            result["raw_response"] = resp_serializable

        return JSONResponse(content=result)

    except Exception as e:
        # Log server-side detail (do not attempt to JSON-serialize non-serializable objects here)
        print(f"Error in /ask/: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    
from base64 import b64decode

@app.post("/analyze-image-json")
async def analyze_image_json(request: Request):
    try:
        body = await request.json()
        payload = body.get("payload")

        if not payload:
            raise HTTPException(status_code=400, detail="Missing payload")

        # ================================
        # 1. Extract base64 image safely
        # ================================
        parts = payload["contents"][0]["parts"]

        text_prompt = None
        image_base64 = None
        mime_type = None

        for part in parts:
            if "text" in part:
                text_prompt = part["text"]
            if "inlineData" in part:
                image_base64 = part["inlineData"]["data"]
                mime_type = part["inlineData"]["mimeType"]

        if not image_base64:
            raise HTTPException(status_code=400, detail="Image data missing")

        # ================================
        # 2. Convert base64 → PIL Image
        # ================================
        image_bytes = b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # ================================
        # 3. Build Python-native contents
        # ================================
        contents = [text_prompt, image]

        # ================================
        # 4. Generation config (snake_case)
        # ================================
        generation_config = {
            "temperature": 0.2,
            "max_output_tokens": 2048,
        }

        # ================================
        # 5. Call Gemini CORRECTLY
        # ================================
        response = gemini_model.generate_content(
            contents=contents,
            generation_config=generation_config,
        )
        # extract text
        text = response.text
        # ============================
        # SAVE HISTORY TO MONGODB
        # ============================

        history_doc = {
        "image_base64": image_base64,   # base64 image
        "report": text,                 # AI-generated report
        "qa": [],                       # empty Q&A list
        "created_at": datetime.utcnow()
        }
        insert_result = history_collection.insert_one(history_doc)
        history_id = str(insert_result.inserted_id)

        # ================================
        # 6. Extract text safely
        # ================================
        try:
            text = response.text
        except Exception:
            try:
                text = response.candidates[0].content.parts[0].text
            except Exception:
                text = None

        #return {"text": text,"history_id": history_id}
        return {"text": text,"history_id": str(insert_result.inserted_id)}

    except Exception as e:
        print("🔥 Gemini backend error:", e)
        raise HTTPException(status_code=500, detail=str(e))
    
"""@app.post("/ask-followup")
async def ask_followup(request: Request):
    data = await request.json()
    history_id = data.get("history_id")
    question = data.get("question")

    response = gemini_model.generate_content(question)
    answer = response.text

    history_collection.update_one(
        {"_id": ObjectId(history_id)},
        {
            "$push": {
                "qa": {
                    "question": question,
                    "answer": answer,
                    "time": datetime.utcnow()
                }
            }
        }
    )

    return {"answer": answer}"""

@app.post("/ask-followup")
async def ask_followup(request: Request):
    print("🔥 /ask-followup API HIT")

    data = await request.json()
    print("📥 Payload received:", data)

    history_id = data.get("history_id")
    question = data.get("question")

    if not history_id or not question:
        raise HTTPException(status_code=400, detail="Missing history_id or question")

    # Call Gemini
    response = gemini_model.generate_content(question)
    answer = response.text

    # 🔥 IMPORTANT FIX: convert string → ObjectId
    result = history_collection.update_one(
        {"_id": ObjectId(history_id)},
        {
            "$push": {
                "qa": {
                    "question": question,
                    "answer": answer,
                    "time": datetime.utcnow()
                }
            }
        }
    )

    print("Matched documents:", result.matched_count)
    print("Modified documents:", result.modified_count)

    return {"answer": answer}


@app.get("/history")
def get_history():
    records = list(history_collection.find({}, {"image_base64": 0}))
    for r in records:
        r["_id"] = str(r["_id"])
    return records

