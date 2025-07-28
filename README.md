# A Retrieval-Augmented Generation (RAG) chatbot system that can answer questions based on your PDF documents using AI.

## Features

- 🤖 AI-powered question answering using Google's Gemini model
- 📚 PDF document processing and indexing
- 🔍 Semantic search using FAISS and sentence transformers
- 🌐 Web interface with Gradio
- 🔌 REST API with FastAPI
- 📖 Automatic document indexing and caching

## Prerequisites

- Python 3.8 or higher
- Google Gemini API key (already configured in the code)

## Quick Start

### Method 1: Direct Run (Recommended)

1. Open PowerShell or Command Prompt
2. Navigate to your project directory:
   ```powershell
   cd "C:\Users\rg817\OneDrive\Desktop\QA_System"
   ```

3. Run the application:
   ```powershell
   python temp.py
   ```

### Method 2: Manual Installation

1. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

2. Run the application:
   ```powershell
   python temp.py
   ```

## What Happens When You Run It

1. **First Run**: The system will:
   - Install all required packages
   - Process all PDF files in your directory
   - Create embeddings and store them in FAISS index
   - Save the index for future use (this may take 5-10 minutes)

2. **Subsequent Runs**: The system will:
   - Load the pre-built index (much faster)
   - Start both Gradio and FastAPI servers

## Accessing the Application

Once running, you'll have access to:

- **Gradio Web Interface**: A user-friendly chat interface
  - URL will be displayed in the terminal (usually `http://127.0.0.1:7860`)
  - You can also access it via the public URL provided by Gradio

- **FastAPI REST API**: For programmatic access
  - API: `http://localhost:8000`
  - Documentation: `http://localhost:8000/docs`
  - Interactive docs: `http://localhost:8000/redoc`

## API Usage

### Using the Web Interface
Simply type your questions in the Gradio interface and get AI-powered answers based on your PDF content.

### Using the REST API
```bash
curl -X POST "http://localhost:8000/ask/" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is machine learning?"}'
```

## Troubleshooting

### Common Issues

1. **Port Already in Use**: If you get a port error, the application might already be running. Check for existing processes.

2. **Missing PDFs**: Make sure all PDF files listed in the code are present in your directory.

3. **Memory Issues**: The first run requires significant memory for processing PDFs. Close other applications if needed.

4. **API Key Issues**: The Gemini API key is already configured, but if you encounter issues, you may need to get your own key from Google AI Studio.

### Getting Help

- Check the terminal output for error messages
- Ensure all PDF files are in the correct directory
- Verify your internet connection (needed for AI model downloads)

## Files Generated

The system will create these files:
- `faiss_index.bin`: FAISS index for fast similarity search
- `texts.pkl`: Pickled text data for retrieval

## Stopping the Application

Press `Ctrl+C` in the terminal to stop both servers.

---

**Note**: The first run will take longer as it processes all your PDF documents. Subsequent runs will be much faster!
