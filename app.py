from flask import Flask, render_template, request, jsonify
import os
import ollama
from PyPDF2 import PdfReader

import fitz  # PyMuPDF for PDF processing
from sentence_transformers import SentenceTransformer
import faiss  # FAISS for vector storage
import numpy as np
import subprocess
import edge_tts
import asyncio
import assemblyai as aai
import requests  # for sending files to Colab
import torch
from werkzeug.utils import secure_filename
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import LLMChain
import fitz  # PyMuPDF
from PIL import Image
import json
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel
# Set your AssemblyAI key
aai.settings.api_key = "d45eb71162fe42f38d8b629925e6ae00"

app = Flask(__name__)

###############################################################################
# (A) OCR Functions
###############################################################################

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/course")
def c   ():
    return render_template("course.html")

@app.route("/ar")
def ar():
    return render_template("ar.html")

@app.route("/quiz")
def quiz():
    return render_template("quiz.html")


@app.route("/pricing")
def pricing():
    return render_template("pricing.html")

@app.route("/3d")
def render3d():
    return render_template("3d.html")


###############################################################################
# (C) RAG / PDF + Embeddings
###############################################################################
# Constants
MODEL = "minicpm-v"
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Global variables for dynamic PDFs
pdf_index = None
pdf_chunks = []

# Global variables for static PDF contexts
static_pdf_paths = ["arduino-course.pdf", "Formation-Pro_-Tour-CNC.pdf"]
static_pdf_indexes = []
static_pdf_chunks = []

def extract_pdf_text(pdf_path: str) -> str:
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(document.page_count):
        page = document.load_page(page_num)
        text += page.get_text("text")
    return text

def chunk_text(text: str, chunk_size: int = 500) -> list:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def build_embeddings(chunks: list):
    return embedder.encode(chunks, convert_to_tensor=True)

def store_embeddings_in_faiss(embeddings):
    embeddings_cpu = embeddings.cpu().detach().numpy()
    dimension = embeddings_cpu.shape[1]
    index = faiss.IndexFlatL2(dimension)
    embeddings_np = np.ascontiguousarray(embeddings_cpu, dtype="float32")
    index.add(embeddings_np)
    return index

def generate_query_embedding(query: str):
    return embedder.encode(query, convert_to_tensor=True)

def search_relevant_context(query: str, index: faiss.IndexFlatL2, chunks: list) -> str:
    query_embedding = generate_query_embedding(query)
    query_embedding_cpu = query_embedding.cpu().detach().numpy().reshape(1, -1)
    D, I = index.search(query_embedding_cpu, k=1)
    return chunks[I[0][0]]

def build_and_store_index(pdf_path: str):
    pdf_text = extract_pdf_text(pdf_path)
    c = chunk_text(pdf_text)
    e = build_embeddings(c)
    idx = store_embeddings_in_faiss(e)
    return idx, c

def load_static_pdfs():
    global static_pdf_indexes, static_pdf_chunks
    for pdf_path in static_pdf_paths:
        idx, chunks = build_and_store_index(pdf_path)
        static_pdf_indexes.append(idx)
        static_pdf_chunks.append(chunks)

def build_prompt_with_context(
    query: str,
    model: str = MODEL,
    index: faiss.IndexFlatL2 = None,
    chunks: list = None,
    image_path: str = None
) -> dict:
    # Adding internal context retrieval for static PDFs
    static_context_messages = []
    for i, static_index in enumerate(static_pdf_indexes):
        static_context = search_relevant_context(query, static_index, static_pdf_chunks[i])
        static_context_messages.append(f"Relevant context from static PDF {i+1}: {static_context}")

    # Handle dynamic PDF context
    if index and chunks:
        relevant_context = search_relevant_context(query, index, chunks)
    else:
        relevant_context = "No PDF context provided."

    messages = [
        {
            "role": "user",
            "content": (
                "You are an expert AI assistant that will answer the following query strictly based on the context provided. "
                "Do not use any external knowledge or make assumptions beyond the provided content. "
                "You will receive a query along with optional context extracted from a PDF, and optionally, an image. "
                "If I give you an image please extract relevant information from it to use it."
                "Your answer should strictly reference and use the provided context (if any). "
                "The structure of your response should be:\n\n"
                "1. Provide a summary of the relevant context from the PDF (if applicable).\n"
                "2. Answer the user's query directly, based solely on the context.\n"
                "3. If an image is provided, integrate its content only if necessary.\n\n"
                "Keep the response concise and focused on the key points from the context."
            )
        },
        {
            "role": "system",
            "content": f"Relevant context from the uploaded PDF: {relevant_context}"
        }
    ]

    for static_message in static_context_messages:
        messages.append({
            "role": "system",
            "content": static_message
        })

    if image_path:
        messages.append({
            "role": "system",
            "images": [image_path]
        })

    messages.append({
        "role": "user",
        "content": query
    })

    return {
        "model": model,
        "messages": messages
    }

@app.route("/rag-chat", methods=["POST"])
def rag_chat():
    global pdf_index, pdf_chunks

    user_query = request.form.get("query", "").strip()

    # Process uploaded PDF
    uploaded_pdf = request.files.get("pdf")
    pdf_path = None
    if uploaded_pdf and uploaded_pdf.filename:
        pdf_path = os.path.join("static", "uploads", uploaded_pdf.filename)
        uploaded_pdf.save(pdf_path)
        pdf_index, pdf_chunks = build_and_store_index(pdf_path)

    # Process uploaded images
    image_paths = []
    for key in request.files:
        if key.startswith("image_"):
            uploaded_image = request.files[key]
            if uploaded_image and uploaded_image.filename:
                image_path = os.path.join("static", "uploads", uploaded_image.filename)
                uploaded_image.save(image_path)
                image_paths.append(image_path)

    # Build the prompt with context
    prompt = build_prompt_with_context(
        query=user_query,
        index=pdf_index,
        chunks=pdf_chunks,
        image_path=image_paths[0] if image_paths else None
    )

    # Handle the response
    try:
        response = ollama.chat(**prompt)
        answer = response.get("message", {}).get("content", "No response received.")
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": f"Error communicating with Ollama: {e}"}), 500


###############################################################################
# (E) TTS + Subtitles + Single Route to Colab
###############################################################################
def list_french_and_english_voices():
    result = subprocess.run(["edge-tts", "--list-voices"], capture_output=True, text=True)
    return result.stdout

voices = {
    "en": {
        "male": {
            "humorous, creative": ["en-US-GuyNeural"],
            "humorous": ["en-US-ChristopherNeural"],
            "creative": ["en-US-GuyNeural", "en-US-EricNeural"],
            "serious": ["en-US-ChristopherNeural", "en-US-MichelleNeural", "en-US-JennyNeural"],
        },
        "female": {
            "humorous, creative": ["en-US-AvaNeural"],
            "humorous": ["en-US-AriaNeural"],
            "creative": ["en-US-AvaNeural"],
            "serious": ["en-US-MichelleNeural", "en-US-JennyNeural"],
        },
    },
    "fr": {
        "male": {
            "humorous, creative": ["fr-FR-HenriNeural"],
            "humorous": ["fr-FR-HenriNeural"],
            "creative": ["fr-FR-HenriNeural"],
            "serious": ["fr-FR-HenriNeural", "fr-FR-DenisNeural"],
        },
        "female": {
            "humorous, creative": ["fr-FR-DeniseNeural"],
            "humorous": ["fr-FR-DeniseNeural"],
            "creative": ["fr-FR-DeniseNeural"],
            "serious": ["fr-FR-VivienneNeural", "fr-FR-DeniseNeural"],
        },
    },
    "ar": {
        "male": {
            "humorous, creative": ["ar-EG-ShakirNeural"],
            "humorous": ["ar-EG-ShakirNeural"],
            "creative": ["ar-EG-ShakirNeural"],
            "serious": ["ar-EG-ShakirNeural", "ar-SA-HamedNeural"],
        },
        "female": {
            "humorous, creative": ["ar-EG-SalmaNeural"],
            "humorous": ["ar-EG-SalmaNeural"],
            "creative": ["ar-EG-SalmaNeural"],
            "serious": ["ar-SA-ZariyahNeural", "ar-EG-SalmaNeural"],
        },
    }
}

def select_voice(gender, language, humor_level, creativity_level):
    if humor_level > 5 and creativity_level > 5:
        desired_attribute = "humorous, creative"
    elif humor_level > 5:
        desired_attribute = "humorous"
    elif creativity_level > 5:
        desired_attribute = "creative"
    else:
        desired_attribute = "serious"
    return voices[language][gender][desired_attribute][0]

async def text_to_speech(text, language, gender, humor_level, creativity_level):
    voice = select_voice(gender, language, humor_level, creativity_level)
    print(f"DEBUG: TTS selected voice is: {voice}")
    tts = edge_tts.Communicate(text=text, voice=voice)
    output_file = "output.mp3"
    await tts.save(output_file)
    print(f"DEBUG: TTS completed, saved to {output_file}")

@app.route("/list-voices", methods=["GET"])
def get_voices():
    voice_list = list_french_and_english_voices()
    return jsonify({"voices": voice_list})

@app.route("/generate-speech", methods=["POST"])

def generate_speech():
    """
    1) Extract PDF/Image content if provided; create speech via Ollama => speech.txt
    2) Convert to audio (output.mp3)
    3) Create subtitles (transcript.srt) with AssemblyAI
    4) Upload to Colab for lip-sync, get final video => final_colab_video.mp4
    5) Return success JSON
    """
    print("DEBUG: Entered /generate-speech route")
    try:
        # Step 1: Possibly read PDF or image
        print("DEBUG: Checking for uploaded PDF...")
        uploaded_pdf = request.files.get("pdf")
        pdf_text = ""
        if uploaded_pdf and uploaded_pdf.filename:
            pdf_path = os.path.join("static", "uploads", uploaded_pdf.filename)
            uploaded_pdf.save(pdf_path)
            pdf_text = extract_pdf_text(pdf_path)
            print("DEBUG: PDF extracted text length:", len(pdf_text))

        print("DEBUG: Checking for uploaded Image...")
        uploaded_image = request.files.get("image")
        image_text = ""
        if uploaded_image and uploaded_image.filename:
            image_path = os.path.join("static", "uploads", uploaded_image.filename)
            uploaded_image.save(image_path)
            image_text = image_path
            print("DEBUG: Image extracted text length:", len(image_text))

        # Step 2: Build Ollama prompt => speech.txt
        print("DEBUG: Building user instructions for Ollama.")
        user_instructions = {
            "role": "user",
            "content": (
                "Hello, Knowledgeable Guide! Your task is to help students understand the essentials "
                "of a course from the material provided in PDF/image. Summarize it in a clear, 60-second speech. "
                "Focus on main concepts, teacher-to-student style."
            )
        }
        context_message = {
            "role": "system",
            "content": (
                f"PDF Content:\n{pdf_text}\n\n"
                f"Image Content:\n{image_text}\n\n"
                "Use only this content to create your speech. Don't invent details beyond what is here."
            )
        }
        prompt_template = {
            "model": "minicpm-v",
            "messages": [user_instructions, context_message]
        }

        print("DEBUG: Sending request to Ollama for speech generation.")
        response = ollama.chat(**prompt_template)
        if (
            isinstance(response, dict)
            and "message" in response
            and "content" in response["message"]
        ):
            speech_text = response["message"]["content"]
        else:
            speech_text = str(response["message"]["content"])

        print("DEBUG: Ollama response length:", len(speech_text))
        with open("speech.txt", "w", encoding="utf-8") as f:
            f.write(speech_text.strip())

        # Step 3: TTS => output.mp3
        print("DEBUG: Starting TTS with edge-tts.")
        language = request.form.get("language", "en").strip().lower()
        gender = request.form.get("gender", "female").strip().lower()
        humor_level = int(request.form.get("humor_level", "7"))
        creativity_level = int(request.form.get("creativity_level", "7"))

        asyncio.run(text_to_speech(speech_text, language, gender, humor_level, creativity_level))
        print("DEBUG: TTS completed successfully.")

        # Step 4: Subtitles => transcript.srt
        print("DEBUG: Starting AssemblyAI transcription of output.mp3.")
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe("output.mp3")
        print("DEBUG: AssemblyAI transcript text length:", len(transcript.text))

        subtitles_srt = transcript.export_subtitles_srt()
        os.makedirs("subtitles", exist_ok=True)
        srt_path = os.path.join("subtitles", "transcript.srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(subtitles_srt)

        # Step 5: Upload output.mp3, transcript.srt, static/video.mp4 to Colab
        colab_url = "https://62a6-34-16-206-224.ngrok-free.app/generate-lipsync"  #  Colab endpoint
        static_video_path = "static/video.mp4"
        if not os.path.exists(static_video_path):
            print(f"ERROR: static video not found at {static_video_path}")
            return jsonify({"error": f"No static video found at {static_video_path}"}), 400

        print("DEBUG: Preparing files to POST to Colab:", colab_url)
        files_to_send = {
            "audio": open("output.mp3", "rb"),
            "video": open(static_video_path, "rb"),
            "subtitles": open(srt_path, "rb"),
        }

        print("DEBUG: Sending request to Colab for final lip-sync.")
        try:
            r = requests.post(colab_url, files=files_to_send, verify=False)  # 2.5 hours
        finally:
            for f in files_to_send.values():
                f.close()

        print("DEBUG: Colab response status code:", r.status_code)
        if r.status_code != 200:
            print("DEBUG: Colab returned an error:", r.text)
            return jsonify({"error": f"Colab returned status {r.status_code}: {r.text}"}), 500

        # Step 6: Save final video
        final_path = "final_colab_video.mp4"
        with open(final_path, "wb") as f:
            f.write(r.content)

        print("DEBUG: final_colab_video.mp4 saved.")
        return jsonify({
            "status": "success",
            "message": (
                f"Speech + audio + subtitles => final video saved as {final_path}. "
                f"language={language}, gender={gender}, humor={humor_level}, creativity={creativity_level}"
            )
        })

    except Exception as e:
        print("ERROR in /generate-speech route:", e)
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
vector_db_path = "vector_db_dir"
try:
    vectordb = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
    print("FAISS vector store loaded successfully.")
except Exception as e:
    vectordb = None
    print(f"Failed to load FAISS vector store: {e}")

# ---------- GOOGLE GENERATIVE AI MODEL ----------
GOOGLE_API_KEY = 'AIzaSyB2rdT1ZfKXqwVlePKeXlcUxltduC9psDU'

# ---------- RAG PIPELINE SETUP FOR PDF ----------
llm_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
)

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
text_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

dimension = 512  # Embedding size for CLIP
index = faiss.IndexFlatL2(dimension)
data_store = []  # Store metadata: (description, image_path)

def extract_images_and_text(pdf_path, output_folder):
    """Extract text + images from each page of a PDF."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    pdf_document = fitz.open(pdf_path)
    extracted_data = []
    for page_number in range(len(pdf_document)):
        page = pdf_document[page_number]
        text = page.get_text().strip()
        images = page.get_images(full=True)
        for i, img in enumerate(images):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_filename = os.path.join(output_folder, f"page_{page_number+1}_image_{i+1}.png")
            with open(image_filename, "wb") as image_file:
                image_file.write(image_bytes)
            extracted_data.append({"image_path": image_filename, "description": text})
    pdf_document.close()
    return extracted_data

def vectorize_image(image_path):
    image = Image.open(image_path)
    inputs = clip_processor(images=image, return_tensors="pt")
    embedding = clip_model.get_image_features(**inputs)
    return embedding.detach().numpy().flatten()

def vectorize_text(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    embeddings = text_model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.detach().numpy()
def retrieve_top_k(input_image_path, query, k=1):
    """
    Retrieve top-k relevant images and text from FAISS.
    Handles cases where the image path is None (text-only query) or the query is missing.
    """
    try:
        results = []

        # If an image is provided, vectorize and search by image embedding
        if input_image_path:
            input_image_embedding = vectorize_image(input_image_path)
            distances, indices = index.search(input_image_embedding.reshape(1, -1), k)
            results = [data_store[i] for i in indices[0]]
        else:
            # If no image is provided, initialize an empty results list
            results = [{"description": ""}]  # Add placeholder to handle text queries

        # If a text query is provided, reorder results by text similarity
        if query:
            if results and results[0]["description"]:  # Only if results contain descriptions
                text_embeddings = vectorize_text([r["description"] for r in results])
                query_embedding = vectorize_text([query])
                text_similarities = torch.cosine_similarity(
                    torch.tensor(text_embeddings),
                    torch.tensor(query_embedding),
                    dim=1
                )
                sorted_indices = torch.argsort(text_similarities, descending=True)
                results = [results[i] for i in sorted_indices]

        return results[:k]  # Return top-k results after sorting

    except Exception as e:
        print(f"Error in retrieve_top_k: {e}")
        return []

def generate_answerr(input_image_path, input_question):
    """
    Full RAG pipeline: retrieve + LLM answer.
    Handles optional image and processes based on available inputs.
    """
    # Validate question input
    if not input_question:
        return "No question provided. Please provide a question to proceed."

    try:
        # Initialize context variable
        context = ""

        # If image is provided, retrieve context using the image and question
        if input_image_path:
            top_matches = retrieve_top_k(input_image_path, input_question, k=1)
            if top_matches and top_matches[0].get("description"):
                context = top_matches[0]["description"]
            else:
                return "No relevant information found for the provided image and question."
        else:
                # Define the prompt template
            prompt_template = """
            You are an expert in explaining things. you need to be 100percent correct  you are a chatbot helping adults in a non formel way so you nee to know how to explain
            
            Answer the following question:
            Question: {question}
            
            Please respond concisely.
            """

            # Create a LangChain prompt template
            prompt_template_langchain = PromptTemplate.from_template(prompt_template)

            # Create a chain with the LLM model
            qa_chain = LLMChain(llm=llm_model, prompt=prompt_template_langchain)

            # Run the chain with context and question
            response = qa_chain.run({"context": context, "question": input_question})

            return response.strip()

        # Define the prompt template
        prompt_template = """
                    You are an expert in explaining things. you need to be 100percent correct  you are a chatbot helping adults in a non formel way so you nee to know how to explain
. Based on the provided context:
        {context}
        
        Answer the following question:
        Question: {question}
        
        Please respond concisely.
        """

        # Create a LangChain prompt template
        prompt_template_langchain = PromptTemplate.from_template(prompt_template)

        # Create a chain with the LLM model
        qa_chain = LLMChain(llm=llm_model, prompt=prompt_template_langchain)

        # Run the chain with context and question
        response = qa_chain.run({"context": context, "question": input_question})

        return response.strip()

    except Exception as e:
        # Log the error for debugging and return a user-friendly error message
        print(f"Error in generate_answerr: {e}")
        return "An error occurred while processing your request. Please try again later."

def initialize_pipeline(pdf_path, output_folder):
    """Extract data from PDF and load into FAISS index once."""
    print("Extraction des données...")
    extracted_data = extract_images_and_text(pdf_path, output_folder)
    print("Vectorisation des images...")
    for entry in extracted_data:
        image_embedding = vectorize_image(entry["image_path"])
        index.add(image_embedding.reshape(1, -1))
        data_store.append(entry)

import os

# Define upload folder
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')  # Create an 'uploads' directory in the current working directory
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)  # Create the folder if it doesn't exist

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/pdf-question', methods=['POST'])
def pdf_question():
    """
    Receives form data:
      - 'text': user question (string)
      - 'image': user uploaded image (optional)
    Returns: JSON { "response": "Answer from LLM" }
    """
    try:
        # Get the question from the form data
        question = request.form.get("text", "").strip()
        image_file = request.files.get("image")

        # Ensure a question is provided
        if not question:
            return jsonify({"error": "No question provided."}), 400

        # If an image is provided, save and process it
        if image_file:
            filename = secure_filename(image_file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(upload_path)
            # Call your function to handle the image and question
            answer = generate_answerr(upload_path, question)
        else:
            # Handle the case where only the question is provided
            answer = generate_answerr(None, question)

        # Return the response
        return jsonify({"response": answer}), 200

    except Exception as e:
        app.logger.error(f"Error during PDF question route: {e}")
        return jsonify({
            "error": "An error occurred during PDF question processing.",
            "details": str(e)
        }), 500

PDF_PATH = "Arduino .pdf"
MAX_QUIZZES = 5
QUESTIONS_PER_QUIZ = 3

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text content from a PDF file.
    """
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

import random
import json

def generate_quiz(course_material: str, num_questions: int):
    """
    Generates quiz questions using Google GenAI, returning them as JSON.
    If it fails or the model response is invalid, returns a random subset of the static fallback quiz.
    """

    # 10 fallback questions
    fallback_quiz_list = [
        {
            "question": "What is Arduino?",
            "answers": ["A microcontroller board", "A programming language", "A video game", "A web browser"],
            "correct_answer": "A microcontroller board"
        },
        {
            "question": "Which language is primarily used to program Arduino?",
            "answers": ["Python", "Java", "C/C++", "Ruby"],
            "correct_answer": "C/C++"
        },
        {
            "question": "Which company originally developed Arduino?",
            "answers": ["Arduino LLC", "Microsoft", "Intel", "IBM"],
            "correct_answer": "Arduino LLC"
        },
        {
            "question": "Which of the following is NOT an Arduino board?",
            "answers": ["Arduino Uno", "Arduino Mega", "Raspberry Pi", "Arduino Nano"],
            "correct_answer": "Raspberry Pi"
        },
        {
            "question": "Arduino's default programming language is based on?",
            "answers": ["Visual Basic", "Swift", "C/C++", "Assembly"],
            "correct_answer": "C/C++"
        },
        {
            "question": "What is the typical operating voltage of an Arduino Uno?",
            "answers": ["3.3V", "5V", "12V", "9V"],
            "correct_answer": "5V"
        },
        {
            "question": "In Arduino IDE, the function that runs once at startup is?",
            "answers": ["setup()", "loop()", "main()", "init()"],
            "correct_answer": "setup()"
        },
        {
            "question": "Which method is repeatedly executed in an Arduino program?",
            "answers": ["setup()", "loop()", "run()", "start()"],
            "correct_answer": "loop()"
        },
        {
            "question": "How can you power an Arduino Uno board?",
            "answers": ["USB", "Barrel jack", "VIN pin", "All of the above"],
            "correct_answer": "All of the above"
        },
        {
            "question": "Which communication protocol does Arduino often use to upload code?",
            "answers": ["HTTP", "ISP (In-System Programming)", "SPI", "TCP/IP"],
            "correct_answer": "ISP (In-System Programming)"
        }
    ]

    try:
        prompt = f"""
        You are a quiz generator. Respond ONLY with valid JSON, no extra text.
        Generate {num_questions} multiple-choice questions from the material below.

        Each question must have:
        - "question": string
        - "answers": array of exactly 4 strings
        - "correct_answer": one string from the answers

        Course Material:
        {course_material}

        Your entire response must be valid JSON in the format:
        [
            {{
                "question": "Question text",
                "answers": ["A", "B", "C", "D"],
                "correct_answer": "A"
            }},
            ...
        ]
        """

        response = llm_model.invoke(prompt)
        response_content = response.content.strip()

        if not response_content:
            raise ValueError("No content received from the LLM.")

        # Attempt to parse LLM JSON
        quiz_data = json.loads(response_content)
        return quiz_data

    except Exception as e:
        print(f"Error generating quiz: {e}\nReturning a static fallback quiz.")
        # Pick 5 random questions from the fallback list
        return random.sample(fallback_quiz_list, 5)

@app.route('/generate-quiz', methods=['GET'])
def generate_quiz_route():
    """
    Generates multiple quizzes (each with a defined number of questions) from the embedded PDF.
    """
    try:
        # Extract text from PDF
        course_material = extract_text_from_pdf(PDF_PATH)
        if not course_material:
            return jsonify({"error": "Failed to extract course material from the PDF."}), 500

        # Create several quizzes
        quizzes = []
        for _ in range(MAX_QUIZZES):
            quiz = generate_quiz(course_material, QUESTIONS_PER_QUIZ)
            quizzes.append(quiz)

        return jsonify({"quizzes": quizzes}), 200
    except Exception as e:
        app.logger.error(f"Error during quiz generation: {e}")
        return jsonify({"error": "An error occurred during quiz generation.", "details": str(e)}), 500

@app.route('/submit-quiz', methods=['POST'])
def submit_quiz():
  
    try:
        data = request.get_json()
        quiz = data.get("quiz", [])
        user_answers = data.get("answers", [])

        if not quiz or not user_answers:
            return jsonify({"error": "Quiz or answers missing in the request."}), 400

        # Calculate the score
        score = 0
        for idx, question in enumerate(quiz):
            if idx < len(user_answers) and question["correct_answer"] == user_answers[idx]:
                score += 1

        return jsonify({
            "score": score,
            "total": len(quiz),
            "correct_answers": [q["correct_answer"] for q in quiz]
        }), 200
    except Exception as e:
        app.logger.error(f"Error during quiz submission: {e}")
        return jsonify({"error": "An error occurred while calculating the score.", "details": str(e)}), 500

if __name__ == "__main__":
    load_static_pdfs()
    initialize_pipeline("Arduino .pdf", "images_extraites")

    app.run(debug=True)