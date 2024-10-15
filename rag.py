from flask import Flask, request, render_template, jsonify, send_from_directory
import pypdfium2 as pdfium
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import validators
import docx2txt
import os
import pickle
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from flask_cors import CORS
from better_profanity import profanity
from pydub import AudioSegment
import subprocess
from werkzeug.utils import secure_filename
from openai import OpenAI
import yt_dlp  # Ensure yt-dlp is installed

app = Flask(__name__)
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_AUDIO_EXTENSIONS'] = {'mp3', 'wav', 'm4a', 'flac', 'ogg'}
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Retrieve Groq API Key from environment variables for security
groq_api_key = 'YOUR API KEY'

class RAGSystem:
    def __init__(self, model_name='all-MiniLM-L6-v2', llm_model='llama-3.1-8b-instant'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.memory = ConversationBufferWindowMemory(k=5)  # Adjustable memory length
        self.groq_chat = ChatGroq(
            groq_api_key=groq_api_key, 
            model_name=llm_model
        )
        self.conversation = ConversationChain(
            llm=self.groq_chat,
            memory=self.memory
        )
        self.embeddings_file = 'embeddings.pkl'
        self.transcription_client = OpenAI(
            api_key='YOUR API KEY',
            base_url="https://api.groq.com/openai/v1"
        )

    def save_embeddings(self, embeddings, sentences):
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump((embeddings, sentences), f)

    def load_embeddings(self):
        if not os.path.exists(self.embeddings_file):
            raise FileNotFoundError("Embeddings file not found.")
        with open(self.embeddings_file, 'rb') as f:
            embeddings, sentences = pickle.load(f)
        self.index = create_faiss_index(embeddings)
        return sentences

    def extract_text_from_pdfs(self, pdf_files):
        texts = []
        try:
            for pdf_file in pdf_files:
                text = ""
                pdf = pdfium.PdfDocument(pdf_file)
                for page_num in range(len(pdf)):
                    page = pdf[page_num]
                    textpage = page.get_textpage()
                    text += textpage.get_text_range()
                texts.append(text)
        except Exception as e:
            return str(e), False
        return texts, True

    def extract_text_from_word(self, docx_files):
        texts = []
        try:
            for docx_file in docx_files:
                text = docx2txt.process(docx_file)
                texts.append(text)
        except Exception as e:
            return str(e), False
        return texts, True

    def extract_text_from_txt(self, txt_files):
        texts = []
        try:
            for txt_file in txt_files:
                text = txt_file.read().decode("utf-8")
                texts.append(text)
        except Exception as e:
            return str(e), False
        return texts, True

    def fetch_url_content(self, urls):
        contents = []
        for url in urls:
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                text = ' '.join([p.get_text() for p in soup.find_all('p')])
                if len(text.split()) < 50:
                    text = ' '.join([div.get_text() for div in soup.find_all('div') if len(div.get_text().split()) > 50])
                contents.append(text)
            except requests.exceptions.RequestException as e:
                return str(e), False
        return contents, True

    def extract_text_from_audio(self, audio_files):
        """
        Transcribe audio files using Groq's Whisper API.
        """
        texts = []
        for audio_file in audio_files:
            try:
                # Secure the filename
                filename = secure_filename(audio_file.filename)
                input_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                # Save the uploaded file
                audio_file.save(input_file_path)

                # Re-encode the uploaded audio to OGG (Opus) format to reduce file size
                reencoded_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"encoded_{filename}.ogg")
                self.reencode_audio_to_ogg(input_file_path, reencoded_file_path)

                # Transcribe the re-encoded audio using Groq's Whisper model
                with open(reencoded_file_path, "rb") as audio:
                    transcription = self.transcription_client.audio.transcriptions.create(
                        model="whisper-large-v3",
                        file=audio,
                        response_format="text"
                    )

                texts.append(transcription)

                # Clean up temporary files
                os.remove(input_file_path)
                os.remove(reencoded_file_path)

            except Exception as e:
                return str(e), False
        return texts, True

    def extract_text_from_youtube(self, youtube_urls):
        """
        Extract and transcribe text from YouTube videos using yt-dlp and Groq's Whisper API.
        """
        texts = []
        for url in youtube_urls:
            try:
                # Define paths for downloaded audio
                video_id = self.extract_video_id(url)
                if not video_id:
                    return f"Invalid YouTube URL: {url}", False
                filename = f"{video_id}.%(ext)s"
                input_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

                # Download audio using yt-dlp
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'outtmpl': input_file_path,
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'wav',
                        'preferredquality': '192',
                    }],
                    'quiet': True,
                    'no_warnings': True,
                }

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])

                # The downloaded file will have .wav extension
                downloaded_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{video_id}.wav")

                if not os.path.exists(downloaded_audio_path):
                    return f"Failed to download audio from YouTube URL: {url}", False

                # Re-encode the downloaded audio to OGG (Opus) format
                reencoded_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"encoded_{video_id}.ogg")
                self.reencode_audio_to_ogg(downloaded_audio_path, reencoded_file_path)

                # Transcribe the re-encoded audio using Groq's Whisper model
                with open(reencoded_file_path, "rb") as audio:
                    transcription = self.transcription_client.audio.transcriptions.create(
                        model="whisper-large-v3",
                        file=audio,
                        response_format="text"
                    )

                texts.append(transcription)

                # Clean up temporary files
                os.remove(downloaded_audio_path)
                os.remove(reencoded_file_path)

            except Exception as e:
                return str(e), False
        return texts, True

    def extract_video_id(self, url):
        """
        Extract the video ID from a YouTube URL.
        """
        try:
            from urllib.parse import urlparse, parse_qs
            parsed_url = urlparse(url)
            if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
                query = parse_qs(parsed_url.query)
                return query.get('v', [None])[0]
            elif parsed_url.hostname in ['youtu.be']:
                return parsed_url.path.lstrip('/')
            else:
                return None
        except:
            return None

    def reencode_audio_to_ogg(self, input_file, output_file="encoded_audio.ogg"):
        """
        Re-encode audio file to OGG Opus format using ffmpeg.
        """
        command = [
            "ffmpeg", "-y",  # Overwrite output files without asking
            "-i", input_file,
            "-vn",  # No video
            "-map_metadata", "-1",  # Remove metadata
            "-ac", "1",  # Mono audio
            "-c:a", "libopus",
            "-b:a", "12k",
            "-application", "voip",
            output_file
        ]
        subprocess.run(command, check=True)

    def vectorize_content(self, texts):
        sentences = []
        for text in texts:
            sentences.extend(text.split('. '))
        
        embeddings = self.model.encode(sentences)
        self.index = create_faiss_index(embeddings)
        
        # Save embeddings and sentences to a pickle file
        self.save_embeddings(embeddings, sentences)

    def retrieve_relevant_content(self, query, k=10):
        if self.index is None:
            raise ValueError("Embeddings have not been loaded yet.")
        
        query_embedding = self.model.encode([query])
        D, I = self.index.search(query_embedding, k=k)
        sentences = self.load_embeddings()
        relevant_sentences = [sentences[i] for i in I[0]]
        return relevant_sentences

    def generate_answer(self, context, query):
        user_message = f'{context}\n\n{query}'
        response = self.conversation(user_message)
        return response['response']

    def chat_with_llm(self, user_message):
        response = self.conversation(user_message)
        return response['response']

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def allowed_file(filename, allowed_extensions):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_documents', methods=['POST'])
def process_documents():
    rag_system = RAGSystem()
    input_type = request.form.get('input_type')

    if input_type == "PDFs":
        pdf_files = request.files.getlist("files")
        texts, success = rag_system.extract_text_from_pdfs(pdf_files)
        if not success:
            return jsonify({'error': texts}), 400

    elif input_type == "Word Files":
        docx_files = request.files.getlist("files")
        texts, success = rag_system.extract_text_from_word(docx_files)
        if not success:
            return jsonify({'error': texts}), 400

    elif input_type == "TXT Files":
        txt_files = request.files.getlist("files")
        texts, success = rag_system.extract_text_from_txt(txt_files)
        if not success:
            return jsonify({'error': texts}), 400

    elif input_type == "URLs":
        urls = request.form.get("urls").splitlines()
        url_list = [url.strip() for url in urls if url.strip()]
        if not all(validators.url(url) for url in url_list):
            return jsonify({'error': "Please enter valid URLs."}), 400

        texts, success = rag_system.fetch_url_content(url_list)
        if not success:
            return jsonify({'error': texts}), 400

    elif input_type == "Audio Files":
        audio_files = request.files.getlist("files")
        if not audio_files:
            return jsonify({'error': "No audio files uploaded."}), 400
        # Validate audio file extensions
        for file in audio_files:
            if not allowed_file(file.filename, app.config['ALLOWED_AUDIO_EXTENSIONS']):
                return jsonify({'error': f"Invalid file type: {file.filename}"}), 400
        texts, success = rag_system.extract_text_from_audio(audio_files)
        if not success:
            return jsonify({'error': texts}), 400

    elif input_type == "YouTube Links":
        youtube_urls = request.form.get("youtube_urls").splitlines()
        youtube_urls = [url.strip() for url in youtube_urls if url.strip()]
        if not all(validators.url(url) and ('youtube.com' in url or 'youtu.be' in url) for url in youtube_urls):
            return jsonify({'error': "Please enter valid YouTube URLs."}), 400

        texts, success = rag_system.extract_text_from_youtube(youtube_urls)
        if not success:
            return jsonify({'error': texts}), 400

    else:
        return jsonify({'error': "Invalid input type."}), 400

    rag_system.vectorize_content(texts)
    return jsonify({'message': "Documents processed and embeddings generated successfully."})

@app.route('/answer_question', methods=['POST'])
def answer_question():
    query = request.form.get('query')

    # Profanity check before processing the query
    if profanity.contains_profanity(query):
        return jsonify({'error': "Please use appropriate language to ask your question."}), 400
    if not query:
        return jsonify({'error': "Please provide a query."}), 400

    rag_system = RAGSystem()
    try:
        rag_system.load_embeddings()  # Load pre-computed embeddings
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 400
    
    try:
        relevant_chunks = rag_system.retrieve_relevant_content(query, k=10)
        combined_context = ' '.join(relevant_chunks)
        answer = rag_system.generate_answer(combined_context, query)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({
        'relevant_chunks': relevant_chunks,
        'answer': answer
    })

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form.get('message')

    # Profanity check before processing the message
    if profanity.contains_profanity(user_message):
        return jsonify({'response': "Please use appropriate language to chat."}), 400
    
    if not user_message:
        return jsonify({'error': "Please enter a message to chat with the LLM."}), 400

    rag_system = RAGSystem()
    try:
        response = rag_system.chat_with_llm(user_message)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    return jsonify({'response': response})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=8001)
