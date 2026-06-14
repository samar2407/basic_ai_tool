from dotenv import load_dotenv
load_dotenv()
import os
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from groq import Groq
from google import genai

app = Flask(__name__)
CORS(app)

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
gemini_client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

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
    "1": "You are a patient and enthusiastic teacher explaining concepts to a complete beginner. Break down the given concept step by step, starting from absolute scratch. Use a single, relatable real-world analogy that a 5 year old would understand. Avoid jargon, if you must use a technical term, immediately explain it in plain language.",
    "2": "You are a precise and concise editor. Summarize the given text into exactly 3–5 bullet points. Each bullet must capture a distinct key idea, no repetition. Prioritize the most important information. Each point should be one sentence, clear and self-contained.",
    "3": "You are an expert educator designing a diagnostic quiz. Generate exactly 3 multiple-choice questions on the given topic. Each question must have 4 options (A–D) with only one correct answer. After all 3 questions, provide an answer key with a one-line explanation for why each answer is correct.",
    "4": "You are a senior professional with 20+ years of domain expertise. Rewrite the given text to sound authoritative, precise, and polished. Use confident, active language. Eliminate filler words, vague phrases, and informal tone. The output should be suitable for a formal report, client communication, or executive audience.",
    "5": "You are an experienced hiring manager conducting a real interview. Given the topic or role, generate 8–10 interview questions across three categories: (1) Conceptual: testing theoretical understanding, (2) Practical: testing hands-on experience, (3) Behavioral: testing how the candidate handles real situations. Format them clearly under each category label.",
    "6": "You are an expert academic coach. Generate a structured, realistic, day-by-day study plan for [SUBJECT] with [HOURS] hours/day over [DAYS] days for a [LEVEL] learner, covering phase breakdown, daily tasks with resources, milestones, and a buffer rule for missed days."
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
            response = groq_client.chat.completions.create(model=model_id, messages=messages)
            reply = response.choices[0].message.content

        elif provider == "gemini":
            model_id = MODELS["gemini"].get(model_key, "gemini-2.0-flash")

            contents = []
            for msg in history:
                role = "user" if msg["role"] == "user" else "model"
                contents.append({"role": role, "parts": [{"text": msg["content"]}]})
            contents.append({"role": "user", "parts": [{"text": user_input}]})

            response = gemini_client.models.generate_content(
                model=model_id,
                contents=contents,
                config={"system_instruction": system}
            )
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
            result = gemini_client.models.embed_content(
                model="models/text-embedding-004",
                contents=text
            )
            vecs.append(np.array(result.embeddings[0].values))

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