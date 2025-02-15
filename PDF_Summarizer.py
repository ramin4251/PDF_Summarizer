from typing import Dict, Any
from pydantic import BaseModel
from openai import OpenAI
import pymupdf
import shutil
import os
from fpdf import FPDF
from pathlib import Path
from termcolor import colored
import json
from dotenv import load_dotenv

load_dotenv(dotenv_path=r'.env')
groq_api_key = os.getenv('GROQ_API_KEY')

Path_To_Source_PDF = r'ENTER FULL PATH FOR YOUR PDF FILE'

SOURCE_PDF = Path(Path_To_Source_PDF)
PDF_NAME = SOURCE_PDF.name

BASE_DIR = Path("analysis_results")
PDF_DIR = BASE_DIR / "pdfs"
KNOWLEDGE_DIR = BASE_DIR / "knowledge_bases"
SUMMARIES_DIR = BASE_DIR / "summaries"
PDF_PATH = PDF_DIR / PDF_NAME
OUTPUT_PATH = KNOWLEDGE_DIR / f"{PDF_NAME.replace('.pdf', '_knowledge.json')}"

MODEL_LIST = [
    "gemma2-9b-it",
    "llama-3.3-70b-specdec",
    "deepseek-r1-distill-llama-70b",
    "qwen-2.5-32b",
    "deepseek-r1-distill-qwen-32b",
]
MODEL = MODEL_LIST[0]


client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=groq_api_key,
)


class PageContent(BaseModel):
    has_content: bool
    knowledge: list[str]


def load_or_create_knowledge_base() -> Dict[str, Any]:
    print(colored("🔍 Checking if knowledge base exists...", "cyan"))  # Debug message
    if Path(OUTPUT_PATH).exists():
        print(colored("📚 Knowledge base found. Loading...", "green"))  # Debug message
        with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    print(colored("🆕 No existing knowledge base. Starting fresh...", "yellow"))  # Debug message
    return {}


def save_knowledge_base(knowledge_base: list[str]):
    print(colored("💾 Saving knowledge base...", "blue"))  # Debug message
    output_path = KNOWLEDGE_DIR / f"{PDF_NAME.replace('.pdf', '')}_knowledge.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({"knowledge": knowledge_base}, f, indent=2)
    print(colored(f"✅ Knowledge base saved with {len(knowledge_base)} items.", "green"))  # Debug message


def process_page(client: OpenAI, page_text: str, current_knowledge: list[str], page_num: int) -> list[str]:
    print(colored(f"\n📖 Processing page {page_num + 1}...", "yellow"))
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": f"""Analyze this page as if you're studying from a book.
                Strictly return valid JSON with this structure:
                {{
                    "has_content": boolean,
                    "knowledge": list[str]
                }}
                Rules:
                1. Ensure JSON syntax is perfect (commas, brackets)
                2. Escape double quotes in strings
                3. No markdown formatting
                4. Limit to 10 knowledge points
                5. Skip pages with tables of contents/indexes
                """},
                {"role": "user", "content": f"Page text: {page_text}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )

        # Validate and parse response
        response_text = completion.choices[0].message.content
        try:
            validated = json.loads(response_text)
            result = PageContent(**validated)
        except json.JSONDecodeError:
            print(colored("⚠️ Invalid JSON response from API", "yellow"))
            return current_knowledge

        if result.has_content:
            print(colored(f"✅ Found {len(result.knowledge)} new knowledge points", "green"))
            return current_knowledge + result.knowledge

        return current_knowledge

    except Exception as e:
        print(colored(f"❌ Error processing page {page_num + 1}: {str(e)}", "red"))
        return current_knowledge


def load_existing_knowledge() -> list[str]:
    knowledge_file = KNOWLEDGE_DIR / f"{PDF_NAME.replace('.pdf', '')}_knowledge.json"
    if knowledge_file.exists():
        print(colored("📚 Loading existing knowledge base...", "cyan"))
        with open(knowledge_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(colored(f"✅ Loaded {len(data['knowledge'])} existing knowledge points", "green"))
            return data['knowledge']
    print(colored("🆕 Starting with fresh knowledge base", "cyan"))
    return []


def analyze_knowledge_base(client: OpenAI, knowledge_base: list[str]) -> str:
    if not knowledge_base:
        print(colored("\n⚠️  Skipping analysis: No knowledge points collected", "yellow"))
        return ""
    print(colored("\n🤔 Generating final book analysis...", "cyan"))
    try:
        print(colored("📞 Sending request to Groq API for analysis...", "cyan"))
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system",
                 "content": """You are an expert in summarizing technical content. 
                 Your task is to create a concise yet detailed summary of the provided content using **strict markdown formatting**. 
                 Follow these guidelines carefully:

                 ### Formatting Rules:
                 1. Use `##` for main sections.
                 2. Use `###` for subsections.
                 3. Use bullet points (`*`) for lists.
                 4. Use code blocks (```) for any code snippets, formulas, or technical terms.
                 5. Use **bold** for emphasis and *italics* for terminology.

                 ### Output Requirements:
                 - The summary should be **comprehensive but concise**, capturing key insights, concepts, and important details.
                 - Do not include introductory phrases like "Here is the summary" or "In conclusion."
                 - Return **only the markdown-formatted summary** without any additional text before or after.
                 - Ensure the summary is well-organized and easy to read, with clear section headers and logical flow.

                 ### Task:
                 Analyze the provided content and generate a summary that adheres to the above rules. 
                 Focus on extracting the most important information while maintaining clarity and brevity."""},
                {"role": "user",
                 "content": f"Content to summarize:\n" + "\n".join(knowledge_base)}
            ]
        )
        print(colored("✅ Analysis generated successfully!", "green"))
        return completion.choices[0].message.content
    except Exception as e:
        print(colored(f"❌ Error generating analysis: {str(e)}", "red"))  # Debug message
        return ""


def setup_directories():
    # Get the stem (filename without extension) of the current PDF
    pdf_stem = Path(PDF_NAME).stem

    # Clear only files related to the current PDF from KNOWLEDGE_DIR and SUMMARIES_DIR
    for directory in [KNOWLEDGE_DIR, SUMMARIES_DIR]:
        if directory.exists():
            print(colored(f"🧹 Cleaning directory: {directory}", "yellow"))  # Debug message
            for file in directory.glob("*"):
                # Check if the file name contains the current PDF's stem
                if file.stem.startswith(pdf_stem):
                    print(colored(f"🗑️ Deleting file: {file}", "yellow"))
                    file.unlink()  # Delete only relevant files

    # Create all necessary directories
    for directory in [PDF_DIR, KNOWLEDGE_DIR, SUMMARIES_DIR]:
        print(colored(f"📁 Creating directory: {directory}", "green"))  # Debug message
        directory.mkdir(parents=True, exist_ok=True)

    # Ensure PDF exists in correct location
    if not PDF_PATH.exists():
        source_pdf = Path(PDF_NAME)
        if source_pdf.exists():
            # Copy the PDF instead of moving it
            print(colored(f"📄 Copying PDF to analysis directory: {PDF_PATH}", "green"))  # Debug message
            shutil.copy2(source_pdf, PDF_PATH)
        else:
            raise FileNotFoundError(f"PDF file {PDF_NAME} not found")


def save_summary(summary: str, is_final: bool = False):
    """Save analysis summary to a perfectly formatted PDF"""
    if not summary:
        print(colored("⏭️ Skipping summary save: No content to save", "yellow"))
        return
    pdf_stem = Path(PDF_NAME).stem
    # Generate title from filename stem (replace underscores/hyphens with spaces and title case)
    processed_title = "Summary of " + pdf_stem.replace("_", " ").replace("-", " ").title()
    output_path = SUMMARIES_DIR / f"Summary of {pdf_stem}.pdf"

    class ProfessionalPDF(FPDF):
        def __init__(self):
            super().__init__()
            self.page_break_margin = 280
            self.line_height = 5  # Reduced initial line height
            self.set_auto_page_break(auto=True, margin=25)
            self._setup_fonts()

        def _setup_fonts(self):
            """Font configuration with 'fonts' directory"""
            try:
                # Add fonts from 'fonts' subdirectory
                self.add_font("DejaVu", "", "fonts/DejaVuSans.ttf")
                self.add_font("DejaVu", "B", "fonts/DejaVuSans-Bold.ttf")
                self.set_font("DejaVu", size=10)
                self.line_height = self.font_size * 1.2  # Reduced multiplier
                self.unicode_enabled = True
            except Exception as e:
                print(colored(f"⚠️ Font error: {str(e)}", "yellow"))
                self.set_font("Helvetica", size=10)
                self.line_height = self.font_size * 1.2  # Consistent spacing
                self.unicode_enabled = False

        def _add_line(self, text, style="", indent=0):
            """Line adder with tighter spacing"""
            if self.get_y() + self.line_height > self.page_break_margin:
                self.add_page()
            self.set_font(style=style)
            self.x = 10 + indent
            self.multi_cell(0, self.line_height, text)
            self.ln(self.line_height * 0.6)  # Adjusted vertical spacing

        def header(self):
            """Dynamic header using processed filename"""
            if self.page_no() == 1:
                self.set_font("DejaVu", "B", 16) if self.unicode_enabled else self.set_font("Helvetica", "B", 16)
                self._add_line(processed_title, "B")  # Dynamic title injected here
                self.line(10, 30, self.w - 10, 30)
                self.set_y(35)

        def footer(self):
            """Minimal footer styling"""
            self.set_y(-15)
            self.set_font(size=8)
            self.cell(0, 10, f"Page {self.page_no()}", align="C")

        def _process_content_line(self, line):
            """Advanced markdown parser"""
            line = line.strip()
            if not line:
                return

            # Handle headers
            if line.startswith("#### "):
                self.set_font(size=10, style="B")
                self._add_line(line[5:].strip(), "B", indent=15)
                self.set_font(size=10)
            elif line.startswith("### "):
                self.set_font(size=12, style="B")
                self._add_line(line[4:].strip(), "B", indent=10)
                self.set_font(size=10)
            elif line.startswith("## "):
                self.set_font(size=14, style="B")
                self._add_line(line[3:].strip(), "B")
                self.set_font(size=10)

            # Handle bold text
            elif "**" in line:
                parts = line.split("**")
                for i, part in enumerate(parts):
                    if i % 2 == 1:  # Odd parts are bold
                        self._add_line(part, "B", indent=10)
                    else:
                        self._add_line(part, indent=10)

            # Handle italic text
            elif line.startswith("*") and line.endswith("*"):
                italic_text = line.strip("*")
                self._add_line(italic_text, style="I", indent=10)

            # Handle code blocks
            elif line.startswith("```") and line.endswith("```"):
                code_content = line.strip("`")
                self.set_font("Courier", size=10)  # Use monospaced font for code
                self._add_line(code_content, indent=10)
                self.set_font("DejaVu", size=10)  # Reset font after code block

            # Handle bullet points
            elif line.startswith("* "):
                bullet = "• " if self.unicode_enabled else "- "
                self._add_line(f"{bullet}{line[2:].strip()}", indent=15)

            else:
                self._add_line(line, indent=10)

        def add_section(self, content):
            """Structural content organizer"""
            for line in content.split('\n'):
                self._process_content_line(line)

    try:
        # Clean content and remove markdown artifacts
        cleaned_content = []
        in_code_block = False
        for line in summary.split('\n'):
            line = line.strip()
            # Handle code blocks
            if line.startswith("```"):
                in_code_block = not in_code_block  # Toggle code block state
                continue
            if in_code_block:
                cleaned_content.append(line)  # Keep code block content as-is
                continue
            # Remove excessive stars and empty lines
            if line == "*" or line in ["***", "**", "*****"]:
                continue
            if line:
                cleaned_content.append(line)

        # Create PDF
        pdf = ProfessionalPDF()
        pdf.add_page()
        pdf.add_section('\n'.join(cleaned_content))
        pdf.output(output_path)
        print(colored(f"✅ Perfect PDF saved: {output_path}", "green"))
    except Exception as e:
        print(colored(f"❌ PDF generation failed: {str(e)}", "red"))
        print(colored(f"Problematic content: {cleaned_content}", "yellow"))  # Log problematic content
        raise


def main():
    print(colored("🔧 Setting up directories...", "cyan"))  # Debug message
    setup_directories()
    # Load or initialize knowledge base
    print(colored("📚 Loading or initializing knowledge base...", "cyan"))  # Debug message
    knowledge_base = load_existing_knowledge()
    initial_knowledge_count = len(knowledge_base)
    print(colored("📖 Opening PDF document...", "cyan"))  # Debug message
    pdf_document = pymupdf.open(PDF_PATH)  # Replaced fitz with pymupdf
    pages_to_process = pdf_document.page_count
    print(colored(f"📚 Processing {pages_to_process} pages...", "cyan"))
    for page_num in range(min(pages_to_process, pdf_document.page_count)):
        print(colored(f"🔄 Processing page {page_num + 1} of {pages_to_process}...", "yellow"))  # Debug message
        previous_count = len(knowledge_base)
        page = pdf_document[page_num]
        page_text = page.get_text()
        knowledge_base = process_page(client, page_text, knowledge_base, page_num)
        # Save only if new knowledge was added
        if len(knowledge_base) > previous_count:
            save_knowledge_base(knowledge_base)

        # Always generate final analysis on last page
        if page_num + 1 == pages_to_process:
            print(colored(f"\n📊 Generating final analysis after page {page_num + 1}...", "cyan"))  # Debug message
            final_summary = analyze_knowledge_base(client, knowledge_base)
            save_summary(final_summary, is_final=True)
    # Final save
    if len(knowledge_base) > initial_knowledge_count:
        save_knowledge_base(knowledge_base)
    print(colored("\n✨ Processing complete! ✨", "green", attrs=['bold']))


if __name__ == "__main__":
    main()
