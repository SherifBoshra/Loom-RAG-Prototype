"""Gradio web interface for the Stylist RAG prototype.

Run: python app.py  (then open http://127.0.0.1:7860)
"""

import gradio as gr

from rag import build_index, embed_query, load_kb, load_model, search

# Load the model and embed the knowledge base once, at startup.
model, processor = load_model()
notes, images = load_kb()
items, index = build_index(model, processor, notes, images)


def query(text, image_path, k):
    text = (text or "").strip()
    if not text and not image_path:
        raise gr.Error("Enter a text query, upload a photo, or both.")
    query_vec = embed_query(model, processor, text or None, image_path)
    results = search(query_vec, index, int(k))

    rows, gallery = [], []
    for rank, (i, score) in enumerate(results, 1):
        kind, content = items[i]
        rows.append([rank, kind, round(score, 3), content])
        if kind == "image":
            gallery.append((content, f"#{rank}  score {score:.3f}"))
    return rows, gallery


with gr.Blocks(title="Stylist RAG") as demo:
    gr.Markdown(
        "# Stylist RAG (prototype)\n"
        f"Search {len(notes)} fabric/style notes and {len(images)} clothing photos "
        "by text, photo, or both."
    )
    with gr.Row():
        with gr.Column():
            text_in = gr.Textbox(label="Text query", placeholder="grey suit for a wedding")
            image_in = gr.Image(label="Photo query (optional)", type="filepath")
            k_in = gr.Slider(1, max(1, min(20, len(items))), value=min(5, len(items)), step=1, label="Results (k)")
            btn = gr.Button("Search", variant="primary")
        with gr.Column():
            table_out = gr.Dataframe(headers=["rank", "type", "score", "match"], label="Results", wrap=True)
            gallery_out = gr.Gallery(label="Matching photos", columns=3, height="auto")

    btn.click(query, [text_in, image_in, k_in], [table_out, gallery_out])
    text_in.submit(query, [text_in, image_in, k_in], [table_out, gallery_out])

if __name__ == "__main__":
    demo.launch()
