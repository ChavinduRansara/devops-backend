from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)
CORS(app)

# Load model and tokenizer
model_name = os.getenv("MODEL_NAME")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set the pad_token_id to eos_token_id if not already set
model.config.pad_token_id = model.config.eos_token_id

@app.route('/generate', methods=['POST'])
def generate_text():
    # Extract the prompt from the request JSON body
    data = request.get_json()
    prompt = data.get('query', '')

    # Check if prompt exists
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    # Tokenize the prompt and generate response
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(
        inputs["input_ids"],
        max_new_tokens=300,         # Increase for longer responses
        temperature=0.6,            # Lower temperature for predictable responses
        top_p=0.9,
        do_sample=True,
        no_repeat_ngram_size=2,     # Avoid repetition
        eos_token_id=tokenizer.eos_token_id  # Stop at end-of-sequence
    )

    # Decode and prepare the output
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_text = response.replace(prompt, "").strip()  # Remove the prompt from the response

    # Return the response as JSON
    return jsonify({"input_prompt": prompt, "generated_summary": generated_text})

if __name__ == "__main__":
    # Use the PORT environment variable provided by Render, default to 5000
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
