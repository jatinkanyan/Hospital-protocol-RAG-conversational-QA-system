import os
import pathlib
import argparse
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.retrievers import MultiQueryRetriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Optional
import requests
from urllib.parse import quote_plus


class DummyLLM:
    def __init__(self, response="This is a dummy response for testing."):
        self._response = response

    def __call__(self, prompt: dict, *args, **kwargs):
        return {"result": self._response}


def get_llm(api_key: Optional[str], use_google: bool, google_model: str):
    if api_key and use_google:
        return ChatGoogleGenerativeAI(model=google_model, temperature=0)
    # prefer a local HF model when Google is disabled
    try:
        return LocalHFLLM()
    except Exception:
        return DummyLLM()


def _quick_google_quota_check(model: str, key: str) -> bool:
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generate"
        params = {}
        if key:
            params["key"] = key
        payload = {"prompt": {"text": "hi"}, "maxOutputTokens": 1}
        resp = requests.post(url, params=params, json=payload, timeout=6)
        if resp.status_code == 429:
            return False
        return resp.status_code < 400
    except requests.RequestException:
        return True


class LocalHFLLM:
    """Local HuggingFace text2text generation wrapper (small and simple).

    Copied behaviour from qa_system.py to provide a local fallback without
    requiring a LangChain Runnable implementation.
    """
    def __init__(self, model_name: str = None):
        import os
        model_name = model_name or os.getenv("LOCAL_MODEL", "google/flan-t5-small")
        try:
            from transformers import pipeline
        except Exception as e:
            raise RuntimeError("transformers not installed or import failed: " + str(e))

        try:
            self.pipe = pipeline("text2text-generation", model=model_name, truncation=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load local model {model_name}: {e}")

    def __call__(self, prompt: dict, *args, **kwargs):
        text = None
        if isinstance(prompt, dict):
            text = prompt.get("query") or prompt.get("input") or str(prompt)
        else:
            text = str(prompt)

        out = self.pipe(text, max_new_tokens=256)
        gen = out[0].get("generated_text") or out[0].get("text") or str(out[0])
        return {"result": gen}


def main():
    # Ensure HF cache directory on D: if available (helps avoid C: space issues)
    try:
        hf_cache_dir = pathlib.Path("D:/hf_cache")
        hf_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(hf_cache_dir))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_cache_dir))
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Conversational QA (2-turn demo)")
    parser.add_argument("--q1", help="First question (non-interactive)")
    parser.add_argument("--q2", help="Second (follow-up) question (non-interactive)")
    parser.add_argument("--use_parent", action="store_true", help="Use ParentDocumentRetriever when available for hierarchical documents")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    use_google = os.getenv("USE_GOOGLE", "0") == "1"
    google_model = os.getenv("GOOGLE_MODEL", "models/gemini-1.5-pro")

    # load FAISS index + local embeddings for retrieval
    local_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("faiss_index", local_embeddings, allow_dangerous_deserialization=True)

    # MultiQueryRetriever wraps a retriever to support query reformulation and multi-turn
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    # Optionally try to use ParentDocumentRetriever for hierarchical document retrieval
    if args.use_parent:
        try:
            from langchain.retrievers import ParentDocumentRetriever
            parent = ParentDocumentRetriever(parent_db=db)
            retriever = parent
            print("Using ParentDocumentRetriever for retrieval (hierarchical).")
        except Exception:
            print("ParentDocumentRetriever unavailable; continuing with standard retriever.")
    try:
        multi = MultiQueryRetriever.from_retriever(retriever)
    except Exception:
        try:
            multi = MultiQueryRetriever(retriever=retriever)
        except Exception:
            multi = retriever

    # LLM selection with quick quota check to avoid long remote retries
    llm = None
    if api_key and use_google:
        allowed = _quick_google_quota_check(google_model, api_key)
        if allowed:
            print(f"Using Google model: {google_model}")
            llm = ChatGoogleGenerativeAI(model=google_model, temperature=0)
        else:
            print("Detected immediate 429 from Google. Falling back to local model.")

    if llm is None:
        try:
            llm = LocalHFLLM()
            print("Using local HuggingFace model for generation.")
        except Exception as e:
            print("Local HF model unavailable, using DummyLLM:", e)
            llm = DummyLLM()

    # Conversation memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Try to build a LangChain ConversationalRetrievalChain when possible.
    ConversationalRetrievalChain = None
    try:
        # Different LangChain versions expose this differently
        from langchain.chains import ConversationalRetrievalChain as _CRC
        ConversationalRetrievalChain = _CRC
    except Exception:
        try:
            from langchain.chains.conversational_retrieval import ConversationalRetrievalChain as _CRC
            ConversationalRetrievalChain = _CRC
        except Exception:
            ConversationalRetrievalChain = None

    conv_chain = None
    # Only attempt to create the conversational chain when we have a remote
    # Runnable LLM (e.g., ChatGoogleGenerativeAI). Our LocalHFLLM is not a
    # LangChain Runnable so we continue using the local fallback path.
    if ConversationalRetrievalChain is not None and llm.__class__.__name__ not in ("LocalHFLLM", "DummyLLM"):
        try:
            conv_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=multi, memory=memory, return_source_documents=True)
            print("Built ConversationalRetrievalChain using remote LLM and MultiQueryRetriever.")
        except Exception as e:
            # If building the chain fails, fall back to the local prompt flow
            print("Failed to build ConversationalRetrievalChain; falling back to local flow:", e)
            conv_chain = None

    # Helper that runs retrieval + generation with optional memory
    def answer_with_local_fallback(question: str, previous_ai: str | None = None, compare: bool = False):
        # first get relevant docs
        docs = db.similarity_search(question, k=3)
        context = "\n\n".join(d.page_content for d in docs[:3])

        # Build a more explicit prompt. If this is a compare-followup, include
        # the previous assistant reply and instruct the model to compare.
        if compare and previous_ai:
            prompt = (
                f"Previous assistant reply:\n{previous_ai}\n\n"
                f"Context:\n{context}\n\n"
                f"Now compare the previous reply with the answer to the new question:\n{question}\n"
                f"Give a concise bullet list with 1) key similarities, 2) key differences, and 3) clear recommendation if one applies. End with explicit source citations in the form [title - page]."
            )
        else:
            prompt = (
                f"Context:\n{context}\n\nQuestion: {question}\n"
                f"Answer concisely in 3-6 bullets. For each bullet, include an explicit source citation in the form [title - page]."
            )

        if llm.__class__.__name__ == "LocalHFLLM":
            out = llm({"query": prompt})
            text = out.get("result")
        else:
            # Remote runnable LLMs may accept a simpler query mapping
            out = llm({"query": question})
            text = out.get("result")

        return text, docs

    # interactive or non-interactive two-turn demo
    if args.q1 and args.q2:
        user1 = args.q1
        user2 = args.q2
    else:
        print("Conversational QA demo (2 turns).\nDoctor: Ask your first question.")
        user1 = input("Doctor: ")

    # First turn
    res_text1, docs1 = answer_with_local_fallback(user1)
    print("Bot:", res_text1)
    print("Sources:", [d.metadata for d in docs1])
    try:
        memory.chat_memory.add_user_message(user1)
        memory.chat_memory.add_ai_message(res_text1)
    except Exception:
        pass

    # Second turn
    if not (args.q1 and args.q2):
        print("\nDoctor: Ask a follow-up that refers to the previous answer.")
        user2 = input("Doctor: ")

    # For the follow-up, instruct the local model to compare with the previous reply
    res_text2, docs2 = answer_with_local_fallback(user2, previous_ai=res_text1, compare=True)
    print("Bot:", res_text2)
    print("Sources:", [d.metadata for d in docs2])


if __name__ == "__main__":
    main()
