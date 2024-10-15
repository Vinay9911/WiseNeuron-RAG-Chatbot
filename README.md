
# 🚀 WiseNeuron-RAG-based Chatbot

Welcome to the **RAG-based Chatbot** project! This repository contains the code for a Flask-based chatbot that leverages a Retrieval-Augmented Generation (RAG) model to handle queries related to documents, audio, URLs, and YouTube videos. It combines the power of Sentence Transformers, FAISS, and a large language model to provide an efficient and interactive chat experience. 

## 📋 Features

- 🗂 **Document Processing**: Handles PDF, Word, and TXT files for text extraction.
- 🌐 **Web Content Extraction**: Retrieves content from URLs for processing.
- 🔊 **Audio Transcription**: Converts audio files into text using Groq's Whisper API.
- 🎥 **YouTube Video Transcription**: Transcribes the audio of YouTube videos.
- 💬 **Conversation Memory**: Keeps track of conversation context for a seamless chat experience.
- 🚫 **Profanity Filter**: Ensures user inputs are free from inappropriate language.

## 🛠️ Setup and Installation

Follow these steps to set up and run the project locally:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Vinay9911/WiseNeuron-RAG-Chatbot.git
   ```

2. **Create a Virtual Environment** (Optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**
   - Create a `.env` file and add the required API keys or directly edit rag.py to replace "YOUR API KEY" with your API key from GROQ 
     

5. **Run the Application**
   ```bash
   python rag.py
   ```

6. **Access the App**
   - Open your browser and go to: `http://localhost:8001`

## 🚀 Usage

1. **Document Processing**
2. **Ask Questions**
3. **Interactive Chat**

## 📦 Dependencies

Some of the key dependencies used in this project include:
- **Flask**: For building the web server.
- **Sentence Transformers**: For text embeddings.
- **FAISS**: For efficient vector search.
- **pypdfium2**, **docx2txt**, **BeautifulSoup**: For text extraction from files and URLs.
- **yt-dlp**, **pydub**, **ffmpeg**: For handling and processing audio and video files.

## 🤝 Contributing

We welcome contributions to this project! If you'd like to add new features, fix bugs, or improve documentation, feel free to open a pull request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## 🌟 Acknowledgments

A big thank you to the open-source community and libraries that made this project possible!

---
