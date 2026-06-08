from dotenv import load_dotenv
load_dotenv()
import os
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

system_prompts = {
    "1": "You are a tutor. Explain the given concept clearly with a simple real-world example.",
    "2": "You are a summarizer. Summarize the given text into 3-5 concise bullet points.",
    "3": "You are a quiz maker. Generate 3 multiple choice questions based on the given topic."
}

feature_names = {
    "1": "Explain a Concept",
    "2": "Summarize Text",
    "3": "Generate Quiz Questions"
}

print("Choose a feature:")
for key, name in feature_names.items():
    print(f"  {key}. {name}")

choice = input("\nEnter 1, 2, or 3: ").strip()

if choice not in system_prompts:
    print("Invalid choice.")
else:
    user_input = input(f"\n[{feature_names[choice]}] Enter your text: ")

    if not user_input.strip():
        print("Input cannot be empty.")
    else:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
        {"role": "system", "content": system_prompts[choice]},
        {"role": "user",   "content": user_input}
    ]
        )
        print(f"\nResponse:\n{response.choices[0].message.content}")