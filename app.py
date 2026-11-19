"""Gradio web interface: describe an outfit (and optionally add a reference
photo) and get back the closest photos from the images/ database.

Run: python app.py  (then open http://127.0.0.1:7860)
"""

import gradio as gr

from rag import build_index, embed_query, load_kb, load_model, search

model, processor = load_model()
state = {"items": [], "index": None}


def scan_images():
    """Embed every photo in images/ (the photo database)."""
    _, images = load_kb()
    if images:
        state["items"], state["index"] = build_index(model, processor, [], images)
    else:
        state["items"], state["index"] = [], None
    return f"Photo database: {len(images)} photo(s) in the images/ folder."


def find(text, ref_image, image_weight, k):
    text = (text or "").strip()
    if state["index"] is None:
        raise gr.Error("No photos found. Add clothing photos to the images/ folder, then click 'Rescan photos'.")
    if not text and not ref_image:
        raise gr.Error("Describe what you want, add a reference photo, or both.")

    # CLIP matches captions best, so phrase the text like a photo caption.
    query_text = f"a photo of {text}" if text else None
    query_vec = embed_query(model, processor, query_text, ref_image, image_weight)
    results = search(query_vec, state["index"], int(k))

    best_i, best_score = results[0]
    best_path = state["items"][best_i][1]
    others = [(state["items"][i][1], f"#{r}  score {s:.3f}") for r, (i, s) in enumerate(results[1:], 2)]
    return best_path, f"Best match: {best_path} (score {best_score:.3f})", others


with gr.Blocks(title="Stylist RAG") as demo:
    gr.Markdown("# Stylist RAG\nDescribe what you're looking for. A reference photo is optional.")
    with gr.Row():
        with gr.Column():
            text_in = gr.Textbox(label="What are you looking for?", placeholder="a low cut red evening dress")
            ref_in = gr.Image(label="Reference / inspiration photo (optional)", type="filepath")
            weight_in = gr.Slider(0, 1, value=0.5, step=0.1,
                                  label="Photo influence (used only when you give both text and a photo)")
            k_in = gr.Slider(1, 12, value=4, step=1, label="How many photos to return")
            btn = gr.Button("Find", variant="primary")
            with gr.Row():
                db_status = gr.Markdown()
                rescan = gr.Button("Rescan photos", size="sm")
        with gr.Column():
            best_out = gr.Image(label="Closest match", interactive=False)
            best_label = gr.Markdown()
            others_out = gr.Gallery(label="Other close matches", columns=3, height="auto")

    inputs = [text_in, ref_in, weight_in, k_in]
    outputs = [best_out, best_label, others_out]
    btn.click(find, inputs, outputs)
    text_in.submit(find, inputs, outputs)
    rescan.click(scan_images, None, db_status)
    demo.load(lambda: scan_status, None, db_status)

scan_status = scan_images()

if __name__ == "__main__":
    demo.launch()
