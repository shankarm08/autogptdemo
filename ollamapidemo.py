import requests
import ollama

# ---------------- FETCH API DATA ---------------- #

url = "https://jsonplaceholder.typicode.com/posts/1"

response = requests.get(url)

data = response.json()

# Extract values
title = data["title"]
body = data["body"]

# ---------------- CREATE PROMPT ---------------- #

prompt = f"""
You are an AI assistant.

Explain this blog post in simple words.

Title:
{title}

Body:
{body}
"""

# ---------------- OLLAMA RESPONSE ---------------- #

result = ollama.chat(
    model="llama3.2:1b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

# ---------------- PRINT OUTPUT ---------------- #

print("\nAI Response:\n")

print(result["message"]["content"])