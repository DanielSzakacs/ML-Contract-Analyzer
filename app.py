import gradio as gr
from PIL import Image
import fitz  # PyMuPDF a PDF képpé alakításához
from transformers import pipeline


doc_qa = pipeline("document-question-answering", model="naver-clova-ix/donut-base-finetuned-docvqa")

def pdf_to_image(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap()
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return image

def answer_question(pdf_file, question):
    image = pdf_to_image(pdf_file)
    output = doc_qa(image=image, question=question)
    return output[0]['answer']  # legvalószínűbb válasz

with gr.Blocks() as demo:
    gr.Markdown("# 📄 ProcureSense - Dokumentum Kérdés-Válasz Demo")
    pdf_input = gr.File(label="📎 PDF feltöltése")
    question_input = gr.Textbox(label="❓ Kérdés")
    answer_output = gr.Textbox(label="💬 Válasz")
    run_button = gr.Button("🔍 Kérdezz")
    
    run_button.click(fn=answer_question, inputs=[pdf_input, question_input], outputs=answer_output)

if __name__ == "__main__":
    demo.launch()
