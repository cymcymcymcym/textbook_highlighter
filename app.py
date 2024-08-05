import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import fitz
import pikepdf
import pdfplumber
from PIL import Image,ImageDraw
import io
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow import keras
import numpy as np
import gradio as gr
import os
import shutil
import uuid
import datetime

euro_new_path="euro_new.pdf"
euro_old_path="euro_old.pdf"
western_civ_path="western_civ.pdf"
world_path="world.pdf"
us_path = "us.pdf"


def write_pickle(obj_in,out_path_in,name_in):
    pickle.dump(obj_in,open(out_path_in+name_in+".pk","wb"))
    
def read_pickle(path):
    tmp=pickle.load(open(path, "rb"))
    return tmp
def load_model(tok_path,model_path):
    # Load the tokenizer from the file
    loaded_tokenizer = read_pickle(tok_path)

# Load the model from the file
    loaded_model = tf.keras.models.load_model(model_path)
    return loaded_tokenizer,loaded_model
loaded_tokenizer,loaded_model=load_model("tokenizer_h5.pk",'my_model.h5')

def split_into_phrases(sentence):
    return re.split(r'[,.;:!?]+', sentence)

def highlight_phrases(paragraph, binary_array, min_highlight_ratio=0.25):
    words = paragraph.split(" ")

    if len(words) != len(binary_array):
        binary_array=binary_array[0:len(words)]

    sentences = re.split(r'(?<=[.!?]) +', paragraph)

    updated_binary_array = []
    index = 0
    for sentence in sentences:
        for phrase in split_into_phrases(sentence):
            phrase_words = phrase.split()
            phrase_binary_array = binary_array[index:index + len(phrase_words)]
            index += len(phrase_words)

            total_words = len(phrase_words)
            highlighted_words = sum(phrase_binary_array)
            
            if total_words!=0 and highlighted_words / total_words >= min_highlight_ratio:
                updated_binary_array.extend([1] * total_words)
            else:
                updated_binary_array.extend(phrase_binary_array)

    return updated_binary_array

def get_binary_array(paragraph,loaded_tokenizer,loaded_model,max_len=492):
    new_sentence = paragraph
    new_sequence = loaded_tokenizer.texts_to_sequences([new_sentence])
    padded_new_sequence = pad_sequences(new_sequence, maxlen=max_len, padding='post')
    predicted_binary_sequence = loaded_model.predict(padded_new_sequence).flatten()
    #print(predicted_binary_sequence)
    processed=[1 if predicted_binary_sequence[i]>=0.5 else 0 for i in range(np.size(predicted_binary_sequence))]
    return processed

def extract_words_info(pdf_path):
    doc = fitz.open(pdf_path)
    words_info = []

    for page_number, page in enumerate(doc):
        words = page.get_text("words")
        for word in words:
            x0, y0, x1, y1, text, block, line, word_number = word
            words_info.append({
                "text": text,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "size": y1 - y0,
                "block": block,
                "line": line,
                "page_number": page_number  # Add the page number to the word information
            })

    return words_info


def words_info_to_paragraphs(words_info):
    paragraphs = []
    if(not(words_info)):
        return paragraphs
    current_block = words_info[0]["block"]
    paragraph_words_info = []

    for word_info in words_info:
        if word_info["block"] == current_block:
            paragraph_words_info.append(word_info)
        else:
            paragraphs.append(paragraph_words_info)
            paragraph_words_info = [word_info]
            current_block = word_info["block"]

    paragraphs.append(paragraph_words_info)

    return paragraphs



def create_highlighted_pdf(input_pdf_path, output_pdf_path, words_info, binary_arrays, dpi=288):
    doc = fitz.open(input_pdf_path)
    if not words_info:
        doc.save(output_pdf_path)
        return
    new_doc = fitz.Document()
    if(len(doc)>7):
        dpi=144
    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)

    for page_number in range(len(doc)):
        page = doc[page_number]
        new_page = new_doc.new_page(width=page.rect.width * scale, height=page.rect.height * scale)

        # Render the original PDF page as an image
        pix = page.get_pixmap(matrix=mat)
        img = Image.open(io.BytesIO(pix.tobytes("png")))

        # Draw highlights on the image
        draw = ImageDraw.Draw(img, "RGBA")
        for word_info, binary_value in zip(words_info, binary_arrays):
            if word_info["page_number"] == page_number:  # Only process words belonging to the current page
                x0, y0, x1, y1 = word_info["x0"] * scale, word_info["y0"] * scale, word_info["x1"] * scale, word_info["y1"] * scale

                if binary_value == 1:
                    color = (255, 255, 0, 128)  # Yellow color with alpha value
                    draw.rectangle([x0, y0, x1, y1], fill=color)

        # Insert the image of the original page with highlights into the new page
        image_data = io.BytesIO()
        img.save(image_data, "png")
        image_data.seek(0)
        new_page.insert_image(new_page.rect, stream=image_data)

    new_doc.save(output_pdf_path)


def extract_required_pages(input_pdf_path, page_range, temp_pdf_name):
    temp_pdf_path = temp_pdf_name + "_" + os.path.basename(input_pdf_path)
    input_doc = fitz.open(input_pdf_path)
    output_doc = fitz.open()

    for pg in page_range:
        if pg <= input_doc.page_count and pg > 0:
            output_doc.insert_pdf(input_doc, from_page=pg-1, to_page=pg-1)

    output_doc.save(temp_pdf_path)
    output_doc.close()
    input_doc.close()

    return temp_pdf_path


# Modify the create_pdf_modified function to accept an additional parameter, page_range
def create_pdf_modified(in_path, page_range, temp_pdf_name):
    temp_pdf_path = extract_required_pages(in_path, page_range,temp_pdf_name)
    input_pdf_path = temp_pdf_path
    output_pdf_path = "highlighted_" + os.path.basename(input_pdf_path)
    words_info = extract_words_info(input_pdf_path)
    paragraphs_words_info = words_info_to_paragraphs(words_info)
    now = datetime.datetime.now()
    print(now)
    print("extracted all paragraphs")
    #print(paragraphs_words_info)
    binary_arrays = []
    for paragraph_words_info in tqdm(paragraphs_words_info, desc="Processing paragraphs"):
        paragraph = " ".join([word_info["text"] for word_info in paragraph_words_info])
        binary_array = get_binary_array(paragraph, loaded_tokenizer, loaded_model)
        binary_array = highlight_phrases(paragraph, binary_array)
        binary_arrays.extend(binary_array)
    print("start_creating_pdf")
    create_highlighted_pdf(input_pdf_path, output_pdf_path, words_info, binary_arrays)
    print("success")
    os.remove(temp_pdf_path)  # Remove the temporary PDF
    return output_pdf_path

def pdf_highlighter(input_pdf_source, input_page_range, input_pdf):
    pdf_source_map = {
        "euro_new": euro_new_path,
        "euro_old": euro_old_path,
        "western_civ": western_civ_path,
        "world": world_path,
        "us_history": us_path,
    }


    if input_pdf:
        pdf_path = input_pdf.name
        #print(pdf_path)
    else:
        pdf_path = pdf_source_map[input_pdf_source]

    page_numbers = []
    for part in input_page_range.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            page_numbers.extend(range(start, end + 1))
        else:
            page_numbers.append(int(part))

    if len(page_numbers) > 10:
        page_numbers = page_numbers[:10]

    temp_pdf_name = str(uuid.uuid4())  # Generate a unique UUID for the temporary PDF name
    output_pdf_path = create_pdf_modified(pdf_path, page_numbers, temp_pdf_name)
    
    return output_pdf_path
input_pdf_source = gr.Radio(choices=["euro_new", "euro_old", "western_civ", "world", "us_history"], label="Select PDF Source or upload your own")
input_page_range = gr.Textbox(value="1-5", label="Enter page range (e.g. 1-5,8,11), maximum page number is 10")
input_pdf = gr.File(file_count="single", type="filepath", label="Upload PDF")
output_pdf = gr.File(label="Download Highlighted PDF")

iface = gr.Interface(fn=pdf_highlighter, inputs=[input_pdf_source, input_page_range, input_pdf], outputs=output_pdf, title="AI PDF Highlighter, specialized at history docs")
iface.launch()