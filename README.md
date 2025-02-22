# FinGraph

FinGraph is a **RAG (Retrieval-Augmented Generation) model pipeline** built with **Flask** (backend) and **React** (frontend). It enables efficient financial data retrieval and analysis using AI-powered embeddings and similarity matching.

## Features
- **Flask Backend:** Handles API requests, processes embeddings, and manages retrieval logic.
- **React Frontend:** Provides an interactive UI for querying and visualizing financial data.
- **Sentence Embeddings:** Utilizes `sentence_transformers` for financial text representation.
- **Similarity Search:** Implements `sklearn.metrics.pairwise` cosine similarity.
- **OpenAI API Integration:** Enhances text generation and query refinement.
- **Environment Variables Management:** Uses `python-dotenv` for secure API key handling.

## Tech Stack
### Backend
- Flask
- Flask-CORS
- Sentence Transformers
- Scikit-learn
- OpenAI API
- Python-dotenv

### Frontend
- React

## Installation

### Prerequisites
- Python 3.x
- Node.js & npm
- Virtual environment (recommended)

### Backend Setup
```bash
# Clone the repository
git clone https://github.com/avishi-sreenidhi/fingraph.git
cd fingraph/backend

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run Flask server
flask run
```

### Frontend Setup
```bash
cd ../frontend

# Install dependencies
npm install

# Start React development server
npm start
```

## Usage
1. Start the **Flask backend** (`flask run`).
2. Start the **React frontend** (`npm start`).
3. Open `http://localhost:3000` in your browser.
4. Query financial data and receive AI-generated responses.

## Environment Variables
Create a `.env` file in `backend/` with:
```env
OPENAI_API_KEY=your_api_key_here
```

