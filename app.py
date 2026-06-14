
from dotenv import load_dotenv
load_dotenv()
import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from groq import Groq

app = Flask(__name__)
CORS(app)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

system_prompts = {
    "1": "You are a tutor. Explain the given concept clearly with a simple real-world example.",
    "2": "You are a summarizer. Summarize the given text into 3-5 concise bullet points.",
    "3": "You are a quiz maker. Generate 3 multiple choice questions based on the given topic.",
    "4": "You are a professional with a lot of experiece. Rewrite the given text in a way a proffessional with a lot of experience would write it"
}

@app.route("/")
def home():
    return send_from_directory(".", "index.html")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    choice = data.get("choice")
    user_input = data.get("user_input", "").strip()
    history = data.get("history", [])

    if not user_input:
        return jsonify({"error": "Input cannot be empty."}), 400

    if choice not in system_prompts:
        return jsonify({"error": "Invalid choice."}), 400
    
    messages=[
            {"role": "system", "content": system_prompts[choice]}]
    messages.extend(history)
    messages.append({"role": "user",   "content": user_input})
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )

    return jsonify({"response": response.choices[0].message.content})

if __name__ == "__main__":
    app.run(debug=True)