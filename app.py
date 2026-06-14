
from dotenv import load_dotenv
load_dotenv()
import os
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from groq import Groq
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

client = Groq(api_key=os.environ.get("groq_api_key"))
genai.configure(api_key=os.environ.get("gemini_api_key"))

MODELS = {
    "groq": {
        "llama-3.3-70b": "llama-3.3-70b-versatile",
        "llama-3.1-8b":  "llama-3.1-8b-instant",
        "mixtral-8x7b":  "mixtral-8x7b-32768",
    },
    "gemini": {
        "gemini-2.0-flash": "gemini-2.0-flash",
        "gemini-1.5-flash": "gemini-1.5-flash",
        "gemini-1.5-pro":   "gemini-1.5-pro",
    }
}

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
    provider = data.get("provider", "groq")
    model_key = data.get("model", "llama-3.3-70b")

    if not user_input:
        return jsonify({"error": "Input cannot be empty."}), 400

    if choice not in system_prompts:
        return jsonify({"error": "Invalid choice."}), 400
    
    system = system_prompts[choice]

    try:
        if provider == "groq":
            model_id = MODELS["groq"].get(model_key, "llama-3.3-70b-versatile")
            messages = [{"role": "system", "content": system}]
            messages.extend(history)
            messages.append({"role": "user", "content": user_input})
            response = client.chat.completions.create(model=model_id, messages=messages)
            reply = response.choices[0].message.content

        elif provider == "gemini":
            model_id = MODELS["gemini"].get(model_key, "gemini-2.0-flash")
            model = genai.GenerativeModel(model_name=model_id, system_instruction=system)

            # convert history to gemini format
            gemini_history = []
            for msg in history:
                role = "user" if msg["role"] == "user" else "model"
                gemini_history.append({"role": role, "parts": [msg["content"]]})

            chat = model.start_chat(history=gemini_history)
            response = chat.send_message(user_input)
            reply = response.text

        else:
            return jsonify({"error": "Unknown provider."}), 400

        return jsonify({"response": reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/search", methods=["POST"])
def search():
    data = request.json
    query = data.get("query", "").strip()
    history = data.get("history", [])

    if not query or not history:
        return jsonify({"results": []})

    try:
        all_texts = [query] + [pair["you"] for pair in history]

        vecs = []
        for text in all_texts:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_query"
            )
            vecs.append(np.array(result["embedding"]))

        query_vec = vecs[0]
        scores = []
        for i, pair in enumerate(history):
            vec = vecs[i + 1]
            similarity = float(np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec)))
            scores.append({"pair": pair, "score": similarity})

        scores.sort(key=lambda x: x["score"], reverse=True)
        return jsonify({"results": [s["pair"] for s in scores[:3]]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)