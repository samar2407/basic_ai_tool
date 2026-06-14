from dotenv import load_dotenv
load_dotenv()
import os
import httpx
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from groq import Groq

app = Flask(__name__)
CORS(app)

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

CF_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
CF_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN")
CF_BASE_URL = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run"

MODELS = {
    "groq": {
        "llama-3.3-70b": "llama-3.3-70b-versatile",
        "llama-3.1-8b":  "llama-3.1-8b-instant",
        "mixtral-8x7b":  "mixtral-8x7b-32768",
    },
    "cloudflare": {
        "llama-3.3-70b": "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        "llama-3.1-8b":  "@cf/meta/llama-3.1-8b-instruct",
        "mistral-7b":    "@cf/mistral/mistral-7b-instruct-v0.1",
    }
}

system_prompts = {
    "1": "You are a patient and enthusiastic teacher explaining concepts to a complete beginner. Break down the given concept step by step, starting from absolute scratch. Use a single, relatable real-world analogy that a 5 year old would understand. Avoid jargon, if you must use a technical term, immediately explain it in plain language.",
    "2": "You are a precise and concise editor. Summarize the given text into exactly 3–5 bullet points. Each bullet must capture a distinct key idea, no repetition. Prioritize the most important information. Each point should be one sentence, clear and self-contained.",
    "3": "You are an expert educator designing a diagnostic quiz. Generate exactly 3 multiple-choice questions on the given topic. Each question must have 4 options (A–D) with only one correct answer. After all 3 questions, provide an answer key with a one-line explanation for why each answer is correct.",
    "4": "You are a senior professional with 20+ years of domain expertise. Rewrite the given text to sound authoritative, precise, and polished. Use confident, active language. Eliminate filler words, vague phrases, and informal tone. The output should be suitable for a formal report, client communication, or executive audience.",
    "5": "You are an experienced hiring manager conducting a real interview. Given the topic or role, generate 8–10 interview questions across three categories: (1) Conceptual: testing theoretical understanding, (2) Practical: testing hands-on experience, (3) Behavioral: testing how the candidate handles real situations. Format them clearly under each category label.",
    "6": "You are an expert academic coach. Generate a structured, realistic, day-by-day study plan for the given subject, covering phase breakdown, daily tasks with resources, milestones, and a buffer rule for missed days."
}

def cloudflare_chat(model_id, messages):
    headers = {
        "Authorization": f"Bearer {CF_API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {"messages": messages}
    response = httpx.post(
        f"{CF_BASE_URL}/{model_id}",
        headers=headers,
        json=payload,
        timeout=60
    )
    result = response.json()
    print("CLOUDFLARE RESPONSE:", result)
    return result["result"]["response"]

def cloudflare_embed(text):
    headers = {
        "Authorization": f"Bearer {CF_API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {"text": [text]}
    response = httpx.post(
        f"{CF_BASE_URL}/@cf/baai/bge-base-en-v1.5",
        headers=headers,
        json=payload,
        timeout=60
    )
    result = response.json()
    return result["result"]["data"][0]

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
        messages = [{"role": "system", "content": system}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_input})

        if provider == "groq":
            model_id = MODELS["groq"].get(model_key, "llama-3.3-70b-versatile")
            response = groq_client.chat.completions.create(model=model_id, messages=messages)
            reply = response.choices[0].message.content

        elif provider == "cloudflare":
            model_id = MODELS["cloudflare"].get(model_key, "@cf/meta/llama-3.3-70b-instruct-fp8-fast")
            reply = cloudflare_chat(model_id, messages)

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
        query_vec = np.array(cloudflare_embed(query))
        scores = []
        for pair in history:
            vec = np.array(cloudflare_embed(pair["you"]))
            similarity = float(np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec)))
            scores.append({"pair": pair, "score": similarity})

        scores.sort(key=lambda x: x["score"], reverse=True)
        return jsonify({"results": [s["pair"] for s in scores[:3]]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)