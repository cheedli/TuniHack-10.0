from flask import Flask, render_template, request, jsonify
import os
import ollama
import fitz  # PyMuPDF for PDF processing
from sentence_transformers import SentenceTransformer
import faiss  # FAISS for vector storage
import numpy as np

import subprocess
import edge_tts
import asyncio
import assemblyai as aai
import requests  # for sending files to Colab

# Set your AssemblyAI key
aai.settings.api_key = "d45eb71162fe42f38d8b629925e6ae00"

app = Flask(__name__)

###############################################################################
# (A) OCR Functions
###############################################################################
def extract_text_from_image(image_path):
    """
    Uses an OCR model (via Ollama) to extract text from an uploaded image.
    """
    prompt_template = {
        "model": "minicpm-v",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Please describe the math-related content in this image, ensuring that any LaTeX formulas "
                    "are correctly transcribed. Non-mathematical details do not need to be described. "
                    "Transcript all the mathematical expressions."
                ),
                "images": [image_path]
            }
        ]
    }
    response = ollama.chat(**prompt_template)
    return response["message"]["content"]

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
            image_text = extract_text_from_image(image_path)
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
import random
import re
def simplify_text(text):
    """ Simplifies the text to remove less essential parts for concise questions. """
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'[\r\n]+', ' ', text).strip()
    text = re.sub(r'^[^a-zA-Z0-9]*|[^a-zA-Z0-9]*$', '', text)
    return text

def generate_key_phrase(chunk):
    """ Extracts a concise key phrase from a chunk of text for question generation. """
    sentences = chunk.split('.')
    for sentence in sentences:
        if 10 < len(sentence.split()) < 30:  # Suitable sentence length
            return sentence.strip()
    return "specific content"


def generate_quiz_from_pdf(pdf_chunks: list) -> list:
    questions = []
    correct_answers = []
    total_questions = min(10, len(pdf_chunks))
    if total_questions == 0:
        return ["Not enough content in the PDF to generate questions."]

    chunk_sample = random.sample(pdf_chunks, total_questions)
    
    for i, chunk in enumerate(chunk_sample):
        simplified_chunk = simplify_text(chunk)
        key_phrase = generate_key_phrase(simplified_chunk)
        question = f"Q{i+1}: Describe the role or purpose of '{key_phrase}'?"
        correct_answer = f"The role or purpose is {key_phrase}."
        distractors = create_distractors(pdf_chunks, chunk)
        all_answers = [correct_answer] + distractors
        random.shuffle(all_answers)  # Shuffle to place the correct answer randomly among the options
        formatted_answers = "\n".join([f"({chr(65 + j)}) {answer}" for j, answer in enumerate(all_answers)])
        questions.append({"question": question, "answers": formatted_answers})
        correct_answers.append(f"Correct Answer for Q{i+1}: {correct_answer}")

    # Print all correct answers in the terminal
    for answer in correct_answers:
        print(answer)

    return questions

def create_distractors(pdf_chunks, correct_answer):
    """ Generates two incorrect answers from the list of chunks, ensuring they differ from the correct answer. """
    distractors = random.sample([chunk for chunk in pdf_chunks if chunk != correct_answer], 2)
    return [simplify_text(distractor.split('.')[0]) for distractor in distractors]

# Route to handle quiz generation requests
@app.route("/generate_quiz/<int:pdf_id>", methods=["GET"])
def handle_quiz_request(pdf_id):
    if pdf_id < 1 or pdf_id > len(static_pdf_chunks):
        return jsonify({"error": "Invalid PDF ID"}), 404
    
    quiz_questions = generate_quiz_from_pdf(static_pdf_chunks[pdf_id-1])
    return jsonify({"quiz": quiz_questions})


if __name__ == "__main__":
    load_static_pdfs()
    app.run(debug=True)