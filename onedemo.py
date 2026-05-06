import requests

goal = "Give me steps to build a chatbot"

memory = []

for step in range(3):

    prompt = f"""
    Goal: {goal}

    Previous memory:
    {memory}

    What should be the next step?
    """

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2:1b",
            "prompt": prompt,
            "stream": False
        }
    )

    answer = response.json()["response"]

    print(f"\nStep {step+1}:")
    print(answer)

    memory.append(answer)