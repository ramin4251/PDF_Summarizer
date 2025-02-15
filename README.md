# PDF Summarizer with Groq AI

A powerful PDF analysis tool that extracts key knowledge points and generates professional summaries using Groq's AI models.

## Features
- PDF content extraction and analysis
- Knowledge point identification
- AI-powered summarization
- Professional PDF report generation
- Multi-model support (Gemma, LLaMA, DeepSeek, Qwen)

## Prerequisites
- Python 3.9+
- Groq API key (free tier available)
- PDF file to analyze

## Installation
```bash
git clone https://github.com/ramin4251/pdf-summarizer.git
cd pdf-summarizer
python -m venv venv
source venv/bin/activate  # Linux/MacOS
# venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Configuration
- Get your Groq API key from Groq Cloud (https://console.groq.com/playground)
- Create `.env` file in root directory:
```
GROQ_API_KEY=your_api_key_here
```
- Modify `Path_To_Source_PDF` in `PDF_Summerizer.py` to point to your PDF

## Usage
```
python PDF_Summerizer.py
```

Generated files will be created in:
- `analysis_results/knowledge_bases/`: JSON knowledge base
- `analysis_results/summaries/`: Final PDF report

## Output Structure
```bash
analysis_results/
├── knowledge_bases/
│   └── [PDF-name]_knowledge.json
├── pdfs/
│   └── input.pdf
└── summaries/
    └── Summary_[PDF-name].pdf
```

## Troubleshooting
- API Errors: Verify `.env` file configuration
- Missing Dependencies: Reinstall requirements.txt
- PDF Errors: Ensure valid PDF file path

## Contributing
Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request
