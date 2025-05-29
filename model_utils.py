from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image
import torch
from pdf2image import convert_from_path

class ContractModel: 
    def __init__(self, model_name="microsoft/layoutlmv3-base"):
        self.processor = LayoutLMv3Processor.from_pretrained(model_name, apply_ocr=True)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_name)

    def pdf_to_image(self, pdf_path, page_number=0):
        """
            Convert PDF to image
        """
        images = convert_from_path(pdf_path)
        image = images[page_number]
        image_path = "temp_page.jpg"
        image.save(image_path, "JPEG")
        return image_path
    
    def extract_tokens(self, image_path):
        """
            Read PDF and take the text out as token
        """
        image = Image.open(image_path).conver("RGB")
        inputs = self.processor(image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)

        predicted_ids = torch.argmax(outputs.logits, dim=2)
        tokens = self.processor.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        labels = predicted_ids[0].tolist()

        extracted = []
        for token, label in zip(tokens, labels):
            if label != 0:
                extracted.append((token, label))
        return extracted