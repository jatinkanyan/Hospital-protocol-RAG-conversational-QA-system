import os
import pathlib

# Prefer an HF cache directory on D: to avoid running out of space on C:
# Create D:\hf_cache and point HF cache env vars there if the path exists or can be created.
try:
    hf_cache_dir = pathlib.Path("D:/hf_cache")
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_cache_dir))
except Exception:
    # If creating on D: fails, continue with defaults (we'll handle errors later)
    pass
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import requests
from urllib.parse import quote_plus
from google.api_core import exceptions as google_exceptions
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

class DummyLLM:
    """Very small local LLM-like fallback for testing.

    Provides minimal `__call__` and `generate` compatible behavior expected by
    RetrievalQA so the chain can run without external API calls.
    """
    def __init__(self, response="This is a dummy response for testing."):
        self._response = response

    def __call__(self, prompt: dict, *args, **kwargs):
        # LangChain may call the LLM with a mapping; return mapping with a result key
        return {"result": self._response}

    def generate(self, prompts, **kwargs):
        class _R:
            def __init__(self, text):
                class G:
                    def __init__(self, t):
                        self.text = t

                self.generations = [[G(text)]]

        return _R(self._response)


class LocalHFLLM:
    """Local HuggingFace text2text generation wrapper.

    Uses a small instruction-following model (default: google/flan-t5-small).
    Falls back gracefully if transformers or torch are not installed.
    """
    def __init__(self, model_name: str = None):
        import os
        model_name = model_name or os.getenv("LOCAL_MODEL", "google/flan-t5-small")
        try:
            from transformers import pipeline
        except Exception as e:
            raise RuntimeError("transformers not installed or import failed: " + str(e))

        # Use text2text-generation to support instruction-style models
        try:
            self.pipe = pipeline("text2text-generation", model=model_name, truncation=True)
        except Exception as e:
            # Re-raise with context for the caller to fallback
            raise RuntimeError(f"Failed to load local model {model_name}: {e}")

    def __call__(self, prompt: dict, *args, **kwargs):
        # Accept mappings with 'query' or 'input'
        text = None
        if isinstance(prompt, dict):
            text = prompt.get("query") or prompt.get("input") or str(prompt)
        else:
            text = str(prompt)

        out = self.pipe(text, max_new_tokens=256)
        # pipeline returns a list of outputs
        gen = out[0].get("generated_text") or out[0].get("text") or str(out[0])
        return {"result": gen}

    def generate(self, prompts, **kwargs):
        # Keep compatibility with the simple .generate() wrapper used elsewhere
        text = prompts[0] if isinstance(prompts, (list, tuple)) and prompts else str(prompts)
        out = self.pipe(text, max_new_tokens=256)

        class _R:
            def __init__(self, text):
                class G:
                    def __init__(self, t):
                        self.text = t

                self.generations = [[G(text)]]

        gen = out[0].get("generated_text") or out[0].get("text") or str(out[0])
        return _R(gen)

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
use_google = os.getenv("USE_GOOGLE", "0") == "1"
google_model = os.getenv("GOOGLE_MODEL", "models/gemini-1.5-pro")
# Optional env override to force immediate fast-fail of remote calls (skip remote LLM)
# Set GOOGLE_FAST_FAIL=1 to force local fallback even if USE_GOOGLE=1
google_fast_fail = os.getenv("GOOGLE_FAST_FAIL", "0") == "1"

# Load FAISS index with local embeddings to avoid external quota during retrieval
local_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(
    "faiss_index",
    local_embeddings,
    allow_dangerous_deserialization=True,
)

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

def _quick_google_quota_check(model: str, key: str) -> bool:
    """Make a tiny direct REST call to the Generative API to detect an immediate
    quota 429 response. Returns True if generation appears allowed, False if
    we detect quota=0 (429). This is intentionally simple and uses the API
    key if available so we avoid langchain's retry wrapper.
    """
    try:
        # Construct a minimal payload.
        url_model = quote_plus(model)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generate"
        params = {}
        if key:
            params["key"] = key

        payload = {
            "prompt": {"text": "hi"},
            "maxOutputTokens": 1,
        }

        # Short timeout to fail fast; do not follow redirects.
        resp = requests.post(url, params=params, json=payload, timeout=6)
        if resp.status_code == 429:
            return False
        # Any 2xx we consider allowed; otherwise treat as allowed (we'll let the
        # langchain client surface other errors when attempting generation).
        return resp.status_code < 400
    except requests.RequestException:
        # Network issues or auth; allow proceeding so the existing client can
        # surface more specific errors (we don't want false negatives).
        return True


# LLM selection
if api_key and use_google and not google_fast_fail:
    print(f"Using Google model: {google_model}")
    # Run a quick quota check to avoid long retries when quota is 0.
    allowed = _quick_google_quota_check(google_model, api_key)
    if not allowed:
        print("Detected immediate 429 (quota=0) from Google. Skipping remote LLM and falling back to local retriever.")
        use_google = False
        # Try to use a local HF LLM if available
        try:
            llm = LocalHFLLM()
            print("Using local HuggingFace model as fallback.")
        except Exception as e:
            print("Local transformers model unavailable, falling back to DummyLLM:", e)
            print("To use local generation install: pip install transformers torch sentence-transformers")
            llm = DummyLLM()
    else:
        llm = ChatGoogleGenerativeAI(model=google_model, temperature=0)
else:
    # Fallback for local testing without calling external APIs: prefer local HF model
    try:
        llm = LocalHFLLM()
        print("Using local HuggingFace model for generation.")
    except Exception as e:
        print("Local transformers model unavailable, falling back to DummyLLM:", e)
        print("To use local generation install: pip install transformers torch sentence-transformers")
        llm = DummyLLM()

query = "What is the standard treatment protocol for Type 2 Diabetes?"

# If we're using the DummyLLM fallback, avoid constructing LangChain's
# RetrievalQA (which expects a Runnable LLM implementation). Instead run a
# direct similarity search and return the dummy response with source metadata.
def _print_local_fallback(query_text: str):
    """Run a local similarity search and print a stable dummy answer + sources.

    This keeps the UX friendly when remote generation fails (quota, network,
    auth). The dummy answer is deterministic to avoid unexpected billing.
    """
    docs = db.similarity_search(query_text, k=3)
    answer_map = DummyLLM()("query") if False else DummyLLM()({"query": query_text})
    answer_text = (
        answer_map.get("result")
        or answer_map.get("text")
        or str(answer_map)
    )
    print("Answer:", answer_text)
    print("\nSources:", [doc.metadata for doc in docs])


if llm.__class__.__name__ in ("DummyLLM", "LocalHFLLM"):
    # Local-only mode (no external LangChain Runnable LLM used).
    docs = db.similarity_search(query, k=3)
    if llm.__class__.__name__ == "LocalHFLLM":
        # Build a concise prompt from the top-k retrieved chunks
        context = "\n\n".join(d.page_content for d in docs[:3])
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer concisely:"
        try:
            answer_map = llm({"query": prompt})
            answer_text = answer_map.get("result") or answer_map.get("text") or str(answer_map)
        except Exception as e:
            print("Local model generation failed, falling back to DummyLLM:", e)
            answer_text = DummyLLM()({"query": query}).get("result")
    else:
        answer_map = DummyLLM()({"query": query})
        answer_text = answer_map.get("result") or answer_map.get("text") or str(answer_map)

    print("Answer:", answer_text)
    print("\nSources:", [doc.metadata for doc in docs])
else:
    # Remote LLM path: wrap call in try/except to catch quota/billing errors
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )

    try:
        result = qa_chain({"query": query})
        # LangChain's new API may return 'result' or 'answer' depending on LLM
        answer = result.get("result") or result.get("answer") or str(result)
        print("Answer:", answer)
        print("\nSources:", [doc.metadata for doc in result.get("source_documents", [])])
    except Exception as e:
        # Prefer catching Google's ResourceExhausted explicitly when available
        is_quota = False
        if isinstance(e, google_exceptions.ResourceExhausted):
            is_quota = True
        else:
            # langchain_google_genai may wrap/raise a generic Exception with HTTP 429
            msg = str(e).lower()
            if "quota" in msg or "429" in msg or "resourceexhausted" in msg:
                is_quota = True

        print("\nRemote LLM call failed:", e)
        if is_quota:
            print("\nDetected Google Generative API quota/billing error (HTTP 429).")
            print("Possible fixes:")
            print(" - Enable billing for your Google Cloud project and request generation quota.")
            print(" - Use a different project with generation quota, or set USE_GOOGLE=0 to use local fallback.")
        else:
            print("An unexpected error occurred while calling the remote LLM. Falling back to local retriever.")

        # Fallback to local similarity search + deterministic dummy answer
        _print_local_fallback(query)
