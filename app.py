from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = Flask(__name__)

# Load model and tokenizer
checkpoint_path = "./results/checkpoint-4375"
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)

def generate_sql(question):
    input_text = "translate English to SQL: " + question
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model.generate(**inputs, max_length=128)
    return tokenizer.decode(output[0], skip_special_tokens=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("message", "")
    sql = generate_sql(question)
    return jsonify({"response": sql})

if __name__ == "__main__":
    app.run(debug=True)
