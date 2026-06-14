# Basic AI Tool

A simple app that takes user input, sends it to an LLM, and displays the response.
It has two AI providers, Grok and Cloudflare AI, each having 3 different LLMs from which the user can choose.

## Deployed link to run it on the browser of any machine

https://basic-ai-tool.onrender.com (deployed using render.com)

## Features

- **Explain a Concept**:Enter any topic and get a clear explanation with a real-world example
- **Summarize Text**:Paste any text and get a concise 3-5 bullet point summary
- **Generate Quiz Questions**:Enter a topic and get 3 multiple choice questions
- **Professional Rewrite**:Enter informal or unprofessional text and get a more formal and professional version of it
- **Generate Interview Questions**:Enter a topic and get theoretical and practical questions which are asked in interviews
- **Generate a Study Plan**:Enter all the specific requirements and get a well structured study plan which you can follow

## Additional Features

- **Chat History**: You can see the chat history on the side panel. You can also access previous chats by searching relevant  terms and also clear chat history.
- **Mode**: Change from Light mode to Dark mode or vice versa any time you like

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

