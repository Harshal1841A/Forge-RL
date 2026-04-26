import os
import requests
from dotenv import load_dotenv

load_dotenv()

def check_openai_compat(name, api_key, base_url, model):
    print(f"Checking {name} ({model})...")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 5
    }
    try:
        response = requests.post(f"{base_url.rstrip('/')}/chat/completions", headers=headers, json=data, timeout=10)
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content'].strip()
            print(f"  [OK] {name} is working. Response: {content.encode('ascii', 'ignore').decode()}")
            return True
        else:
            print(f"  [FAIL] {name} failed. Status: {response.status_code}, Body: {response.text}")
            return False
    except Exception as e:
        print(f"  [FAIL] {name} error: {e}")
        return False

def main():
    print("--- API Connectivity Check ---")
    
    # 1. Groq (Auditor)
    check_openai_compat(
        "Groq (Auditor)",
        os.getenv("OPENAI_API_KEY"),
        os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1"),
        os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
    )

    # 2. Cerebras (Historian)
    check_openai_compat(
        "Cerebras (Historian)",
        os.getenv("CEREBRAS_API_KEY"),
        os.getenv("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1"),
        os.getenv("CEREBRAS_MODEL", "llama3.1-70b")
    )

    # 3. Mistral (Critic)
    # Mistral uses a slightly different URL usually, but chat/completions is standard
    check_openai_compat(
        "Mistral (Critic)",
        os.getenv("MISTRAL_API_KEY"),
        os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1"),
        os.getenv("MISTRAL_MODEL", "mistral-small-latest")
    )

    # 4. OpenRouter (Negotiated Search)
    check_openai_compat(
        "OpenRouter",
        os.getenv("OPENROUTER_API_KEY"),
        os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3-8b-instruct:free")
    )

if __name__ == "__main__":
    main()
