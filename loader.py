import logging
import warnings
import contextlib
import io
import os

from langchain_community.document_loaders import PyPDFLoader


# Reduce verbose PDF parser messages that are informational only (logging + warnings)
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("pypdf").setLevel(logging.WARNING)
logging.getLogger("langchain_community").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")


def load_example_pdf(path=r"C:\Users\jatin\Desktop\standard-treatment-guidelines.pdf"):
	"""Load the example PDF and return documents.

	This function temporarily redirects stdout and stderr and suppresses warnings
	while the PDF parser runs, which hides noisy messages like
	"Advanced encoding /SymbolSetEncoding not implemented yet" that some
	parsers print directly to stderr.
	"""
	loader = PyPDFLoader(path)

	# suppress prints to stdout/stderr from underlying libraries during parse
	# (they sometimes use print() or write to stderr rather than proper logging)
	devnull = open(os.devnull, "w")
	try:
		with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				docs = loader.load()
	finally:
		devnull.close()

	return docs


if __name__ == "__main__":
	docs = load_example_pdf()
	if docs:
		print(docs[0].page_content[:500])
	else:
		print("No documents returned from PDF loader.")
