import ollama
import requests
import math

# -----------------------------
# TOOL 1 - Calculator
# -----------------------------
def calculator(expression):
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculator error: {e}"


# -----------------------------
# TOOL 2 - Weather API
# -----------------------------
def get_weather(city):
    try:
        url = f"https://wttr.in/{city}?format=3"
        response = requests.get(url)

        return response.text
    except Exception as e:
        return f"Weather API error: {e}"


# -----------------------------
# TOOL 3 - Simple Search Tool
# -----------------------------
def search_tool(query):
    fake_database = {
        "python": "Python is a programming language.",
        "ollama": "Ollama runs LLMs locally.",
        "agent": "AI agents can use tools and memory."
    }

    return fake_database.get(query.lower(), "No results found.")


# -----------------------------
# MAIN AGENT
# -----------------------------
def agent(user_input):

    prompt = f"""
    You are an AI agent.

    Available tools:
    1. calculator(expression)
    2. weather(city)
    3. search(query)

    Decide which tool to use.

    User input:
    {user_input}

    Respond ONLY in this format:
    TOOL: tool_name:parameter

    Example:
    TOOL: calculator:5*8
    """

    response = ollama.chat(
        model="llama3.2:1b",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    reply = response['message']['content']
    print("\nLLM Decision:", reply)

    # -----------------------------
    # TOOL EXECUTION
    # -----------------------------
    if "calculator:" in reply:
        expression = reply.split("calculator:")[1].strip()
        result = calculator(expression)

    elif "weather:" in reply:
        city = reply.split("weather:")[1].strip()
        result = get_weather(city)

    elif "search:" in reply:
        query = reply.split("search:")[1].strip()
        result = search_tool(query)

    else:
        result = "No valid tool selected."

    return result


# -----------------------------
# RUN AGENT
# -----------------------------
while True:
    user = input("\nYou: ")

    if user.lower() == "exit":
        break

    output = agent(user)

    print("Agent:", output)