import os, time
from dotenv import load_dotenv

load_dotenv()

MODEL        = "gemini-2.0-flash-lite"
MAX_RETRIES  = 3
MAX_HIST     = 2
USE_OLLAMA   = False
USE_GROQ     = True               # fast free alternative to Gemini
OLLAMA_MODEL = "llama3.2"
GROQ_MODEL = "llama-3.3-70b-versatile"  # fast + free on Groq

SYSTEM_PROMPT = """You are a precise document assistant.
Answer ONLY from the provided context chunks.
If the answer is not in the context, say "I couldn't find this in the uploaded documents."
Never hallucinate or use outside knowledge.

For every factual claim, append a citation in this format:
[Source: <doc_name>, p.<page>]

Be concise and structured."""


def format_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(
            f"[Chunk {i} | {c['doc_name']} | Page {c['page']}]\n{c['text']}"
        )
    return "\n\n---\n\n".join(parts)


def build_prompt(query: str,
                 chunks: list[dict],
                 history: list[dict]) -> str:
    context  = format_context(chunks)
    hist_txt = ""
    for turn in history[-MAX_HIST:]:
        hist_txt += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n\n"

    return f"""--- CONTEXT ---
{context}

--- CONVERSATION HISTORY ---
{hist_txt}--- CURRENT QUESTION ---
{query}

Answer:"""


def generate_answer_groq(prompt: str) -> str:
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    resp   = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.1,
        max_tokens=1024,
    )
    return resp.choices[0].message.content


def generate_answer_ollama(prompt: str) -> str:
    import requests
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model":  OLLAMA_MODEL,
            "prompt": SYSTEM_PROMPT + "\n\n" + prompt,
            "stream": False,
        },
        timeout=120
    )
    return resp.json().get("response", "No response from Ollama.")


def generate_answer_gemini(prompt: str) -> str:
    import google.generativeai as genai
    from google.api_core.exceptions import ResourceExhausted
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel(MODEL)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = model.generate_content(prompt)
            return resp.text
        except ResourceExhausted as e:
            wait = 60
            try:
                wait = int(str(e).split("seconds:")[1].split("}")[0].strip()) + 5
            except Exception:
                pass
            if attempt < MAX_RETRIES:
                time.sleep(wait)
            else:
                return (
                    f"⚠️ Rate limit reached. Wait {wait}s or "
                    f"create a new key at aistudio.google.com/apikey."
                )


def generate_answer(query: str,
                    chunks: list[dict],
                    history: list[dict]) -> tuple[str, list[dict]]:
    prompt = build_prompt(query, chunks, history)

    if USE_OLLAMA:
        answer = generate_answer_ollama(prompt)
    elif USE_GROQ:
        answer = generate_answer_groq(prompt)
    else:
        answer = generate_answer_gemini(prompt)

    return answer, chunks