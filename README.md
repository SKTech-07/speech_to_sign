# Speech → Indian Sign Language (ISL) Gloss Pipeline

A modular FastAPI backend prototype for converting **Speech → English Text → NLP Processing → ISL Gloss**.

This system is designed specifically so that an **ISL Gloss → HamNoSys / SiGML dictionary** can be plugged in later without modifying the speech, NLP, or gloss generation modules.

---

## 🏗️ Architecture

```text
User Speech
    ↓
Speech-to-Text (Whisper)
    ↓
English Text
    ↓
Text Preprocessing (Negation-preserving)
    ↓
NLP Analysis (spaCy)
    ↓
English → ISL Gloss (T5 Model / Rule-based Fallback)
    ↓
Gloss Normalization
    ↓
Gloss Output
    ↓
[FUTURE: ISL Dictionary (HamNoSys / SiGML)]
    ↓
[FUTURE: Avatar Animation]
```

The current version produces structured JSON containing English text and normalized ISL Gloss array:

```json
{
  "text": "I want to drink water",
  "gloss": ["I", "WATER", "DRINK", "WANT"]
}
```

---

## 🛠️ Technology Stack

* **Python 3.10+**
* **FastAPI** & **Uvicorn** for HTTP server & REST APIs
* **OpenAI Whisper** for Speech-to-Text
* **spaCy** (`en_core_web_sm`) for NLP tokenization, lemmatization, POS tagging, and dependency parsing
* **Transformers / T5 Interface** with **Linguistic Rule-Based Fallback**
* **Pydantic v2** for request/response schemas

---

## 📁 Project Structure

```text
isl_translation/
│
├── app/
│   ├── main.py                   # FastAPI server entrypoint
│   │
│   ├── api/
│   │   └── routes.py             # API route handlers
│   │
│   ├── stt/
│   │   └── whisper_service.py    # OpenAI Whisper STT service
│   │
│   ├── nlp/
│   │   ├── preprocess.py         # Text cleaning & contraction expansion (preserves negation)
│   │   └── analyzer.py           # spaCy linguistic analyzer
│   │
│   ├── gloss/
│   │   ├── gloss_generator.py    # T5 / Rule-based gloss generator & normalizer
│   │   └── rules.py              # ISL grammar rules (SOV, questions, negation)
│   │
│   ├── sign_dictionary/
│   │   └── dictionary_service.py # SignDictionary interface for HamNoSys lookup
│   │
│   └── models/
│       └── schemas.py            # Pydantic data models
│
├── data/
│   └── sign_dictionary.json      # Placeholder ISL sign dictionary
│
├── tests/
│   └── test_pipeline.py          # Pytest integration suite
│
├── requirements.txt
├── README.md
└── .env
```

---

## 🚀 Setup & Installation

### 1. Clone & Navigate to Project

```bash
cd d:/projects/speech_to_sign/isl_translation
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Linux / macOS:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download spaCy English Language Model

```bash
python -m spacy download en_core_web_sm
```

---

## 🏃 Running the Server

Start the API server using Uvicorn:

```bash
uvicorn app.main:app --reload --port 8000
```

The interactive OpenAPI documentation will be available at `http://127.0.0.1:8000/docs`.

---

## 📡 API Endpoints

### 1. `GET /health`
Returns backend service health status.

**Example Request:**
```bash
curl -X GET "http://127.0.0.1:8000/health"
```

**Response:**
```json
{
  "status": "ok"
}
```

---

### 2. `POST /translate`
Translates English text to ISL Gloss.

**Example Request:**
```bash
curl -X POST "http://127.0.0.1:8000/translate" \
     -H "Content-Type: application/json" \
     -d '{"text": "I want to drink water"}'
```

**Response:**
```json
{
  "text": "I want to drink water",
  "gloss": [
    "I",
    "WATER",
    "DRINK",
    "WANT"
  ]
}
```

---

### 3. `POST /transcribe`
Transcribes audio file to English text using Whisper.

**Example Request:**
```bash
curl -X POST "http://127.0.0.1:8000/transcribe" \
     -F "file=@sample_speech.wav"
```

**Response:**
```json
{
  "text": "I want to drink water"
}
```

---

### 4. `POST /speech-to-gloss`
Runs the complete Speech → Text → NLP → ISL Gloss pipeline.

**Example Request:**
```bash
curl -X POST "http://127.0.0.1:8000/speech-to-gloss" \
     -F "file=@sample_speech.wav"
```

**Response:**
```json
{
  "text": "I want to drink water",
  "gloss": [
    "I",
    "WATER",
    "DRINK",
    "WANT"
  ]
}
```

---

### 5. `POST /gloss-to-sign`
Placeholder endpoint for the future avatar pipeline stage.

**Example Request:**
```bash
curl -X POST "http://127.0.0.1:8000/gloss-to-sign" \
     -H "Content-Type: application/json" \
     -d '{"gloss": ["I", "WATER", "DRINK", "WANT"]}'
```

**Response:**
```json
{
  "status": "dictionary_not_connected",
  "gloss": [
    "I",
    "WATER",
    "DRINK",
    "WANT"
  ],
  "signs": [
    {"status": "not_implemented", "gloss": "I"},
    {"status": "not_implemented", "gloss": "WATER"},
    {"status": "not_implemented", "gloss": "DRINK"},
    {"status": "not_implemented", "gloss": "WANT"}
  ]
}
```

---

## 🧪 Running Tests

Execute the automated test suite using `pytest`:

```bash
pytest tests/test_pipeline.py -v
```

All 7 required sentence test cases, negation preservation checks, unknown word handling, and endpoint integrations are validated.

---

## 🔌 Connecting HamNoSys Dictionary Later

To populate the sign dictionary in the future:
1. Update `data/sign_dictionary.json` with HamNoSys / SiGML payload objects replacing `null` values:
```json
{
  "WATER": {
    "hamnosys": "𓂋𓏤...",
    "sigml": "<sigml>...</sigml>"
  }
}
```
2. The `SignDictionary` service in `app/sign_dictionary/dictionary_service.py` will automatically return the new sign representations without modifying any existing pipeline code.
