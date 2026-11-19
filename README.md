# Stylist RAG (prototype)

Multimodal retrieval prototype for an AI clothing stylist. A text and/or photo query is embedded with CLIP and matched against a knowledge base of fabric/style notes and clothing images.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Put a few clothing photos in `images/`. Edit `kb/fabric_notes.txt` (one fact per line; the starter notes are placeholders, verify before relying on them).

## Run
```bash
python rag.py --text "grey suit for a wedding"
python rag.py --image my_photo.jpg --k 3
python rag.py --text "lighter fabric" --image jacket.jpg
```

## Web interface
```bash
python app.py
```
Then open http://127.0.0.1:7860 to search by text, photo, or both.

## Next steps
- Pass retrieved results to an LLM to generate the final recommendation (the "G" in RAG)
- Replace the numpy search with FAISS when the KB grows