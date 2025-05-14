import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from service.orchestrator import answer_user_question
from service.service import DocumentIndexer
from service.config import CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL

from docx import Document
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

app = Flask(__name__)
CORS(app, origins=["http://localhost:4200"])

UPLOAD_FOLDER = 'documents'
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_txt_to_pdf(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as txt_file:
        text = txt_file.read()

    pdf = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    y = height - 40
    for line in text.splitlines():
        pdf.drawString(40, y, line)
        y -= 15
        if y < 40:
            pdf.showPage()
            y = height - 40
    pdf.save()

def convert_docx_to_pdf(input_path, output_path):
    doc = Document(input_path)
    pdf = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    y = height - 40
    for para in doc.paragraphs:
        for line in para.text.splitlines():
            pdf.drawString(40, y, line)
            y -= 15
            if y < 40:
                pdf.showPage()
                y = height - 40
    pdf.save()

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'Brak pliku w żądaniu'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nie podano nazwy pliku'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Niedozwolony format pliku'}), 400

    filename = secure_filename(file.filename)
    ext = filename.rsplit('.', 1)[1].lower()
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    final_pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_document.pdf')
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(temp_path)

    try:
        if ext == 'pdf':
            os.rename(temp_path, final_pdf_path)
        elif ext == 'txt':
            convert_txt_to_pdf(temp_path, final_pdf_path)
            os.remove(temp_path)
        elif ext == 'docx':
            convert_docx_to_pdf(temp_path, final_pdf_path)
            os.remove(temp_path)
        else:
            return jsonify({'error': 'Nieobsługiwany format'}), 400
    except Exception as e:
        return jsonify({'error': f'Błąd podczas konwersji: {str(e)}'}), 500

    indexer = DocumentIndexer(
        pdf_path=final_pdf_path,
        chroma_db_path=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME,
        embedding_model=EMBEDDING_MODEL
    )

    return jsonify({'message': 'Plik zapisany, skonwertowany i zaindeksowany pomyślnie.'}), 200

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    user_prompt = data.get('prompt', '')
    if not user_prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    answer = answer_user_question(user_prompt)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)
