import os
import time
import requests
import json
from dotenv import load_dotenv
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from loader import load_example_pdf


class GoogleRESTEmbeddings:
	"""Minimal Embeddings wrapper that calls Google Generative API embedding endpoint.

	This implements the small subset of the LangChain Embeddings API we need:
	- embed_documents(list[str]) -> list[list[float]]
	- embed_query(str) -> list[float]

	The implementation is intentionally tolerant: on any error it raises an
	exception so callers can fall back to local embeddings.
	"""
	def __init__(self, api_key: str, model: str = "models/embedding-001", batch_size: int = 16):
		self.api_key = api_key
		self.model = model
		self.batch_size = batch_size
		self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:embed"

	def _call_embed(self, inputs: List[str]) -> List[List[float]]:
		params = {"key": self.api_key} if self.api_key else {}
		payload = {"input": inputs}
		resp = requests.post(self.base_url, params=params, json=payload, timeout=30)
		if resp.status_code != 200:
			raise RuntimeError(f"Embedding request failed: {resp.status_code} {resp.text}")
		data = resp.json()
		# The API returns a list of embeddings under 'embeddings' or 'data' depending on version
		if "embeddings" in data:
			items = data["embeddings"]
			return [item["embedding"] if isinstance(item, dict) and "embedding" in item else item for item in items]
		if "data" in data:
			# data -> [{embedding: [...]}, ...]
			return [d.get("embedding") or d.get("vector") for d in data["data"]]
		# fallback: try 'responses'
		if "responses" in data:
			return [r.get("embedding") for r in data.get("responses", [])]
		raise RuntimeError("Unknown embedding response format: " + json.dumps(data)[:200])

	def embed_documents(self, texts: List[str]) -> List[List[float]]:
		res = []
		for i in range(0, len(texts), self.batch_size):
			batch = texts[i : i + self.batch_size]
			res.extend(self._call_embed(batch))
			time.sleep(0.1)
		return res

	def embed_query(self, text: str) -> List[float]:
		return self.embed_documents([text])[0]

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
use_google = os.getenv("USE_GOOGLE", "0") == "1"

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = load_example_pdf()
chunks = splitter.split_documents(docs)

# Embedding selection
embeddings = None
if use_google and api_key:
	# Try a couple of known embedding models; if they fail, we'll fallback.
	tried = []
	for candidate in ("models/embedding-gecko-001", "models/embedding-001"):
		try:
			print(f"Attempting Google REST embeddings (model: {candidate})")
			embeddings = GoogleRESTEmbeddings(api_key, model=candidate)
			# Try a tiny call to verify the model works (single query)
			embeddings.embed_query("test")
			print(f"Google embeddings ready using {candidate}")
			break
		except Exception as e:
			print(f"Google embedding model {candidate} failed: {e}")
			tried.append((candidate, str(e)))

if embeddings is None:
	print("Using local HuggingFace embeddings")
	embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create FAISS index (FAISS.from_documents supports either an embeddings object or a callable)
try:
	db = FAISS.from_documents(chunks, embeddings)
except Exception as e:
	print("Failed to build FAISS with chosen embeddings, falling back to local HuggingFace. Error:", e)
	embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
	db = FAISS.from_documents(chunks, embeddings)

db.save_local("faiss_index")
