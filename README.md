# Basic AI Tool

A simple app that takes user input, sends it to an LLM (Llama 3.3 70B via Groq API), and displays the response.

## Deployed link to run it on the browser of any machine

https://basic-ai-tool.onrender.com

## Features

- **Explain a Concept**:Enter any topic and get a clear explanation with a real-world example
- **Summarize Text**:Paste any text and get a concise 3-5 bullet point summary
- **Generate Quiz Questions**:Enter a topic and get 3 multiple choice questions

## How It Works

1. User selects a feature from the dropdown
2. User enters text input
3. Frontend sends the input to the Flask backend via a POST request
4. Backend attaches the appropriate system prompt and calls the Groq API
5. LLM response is returned and displayed on the page

## Tech Stack

- **Frontend:** HTML, CSS, JavaScript
- **Backend:** Python, Flask
- **LLM:** Llama 3.3 70B (via Groq API)
- **Deployment:** Render

