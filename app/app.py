import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="MNIST CNN Demo", page_icon="🧠")

# -------------------------------
# Load trained model
# -------------------------------
@st.cache_resource
def load_m():
    return load_model("artifacts/mnist_cnn.keras")

model = load_m()

# -------------------------------
# Preprocessing helpers
# -------------------------------
def rgba_to_white_bg(img_rgba: Image.Image) -> Image.Image:
    """Composite RGBA onto white background to remove transparency artifacts."""
    if img_rgba.mode != "RGBA":
        img_rgba = img_rgba.convert("RGBA")
    white_bg = Image.new("RGBA", img_rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(white_bg, img_rgba).convert("RGB")

def center_and_resize_28x28(gray_0_255: np.ndarray) -> np.ndarray:
    """
    Input: grayscale (0..255), foreground bright, background dark.
    Output: 28x28 float32 [0,1], MNIST-style centered digit (~20px).
    """
    # Threshold to find digit
    mask = gray_0_255 > 30
    if not mask.any():
        img = Image.fromarray(gray_0_255).resize((28, 28), Image.LANCZOS)
        return np.array(img).astype("float32") / 255.0

    # Crop to bounding box
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    cropped = gray_0_255[y0:y1, x0:x1]

    # Pad to square
    h, w = cropped.shape
    side = max(h, w)
    square = np.zeros((side, side), dtype=np.uint8)  # black bg
    y_off = (side - h) // 2
    x_off = (side - w) // 2
    square[y_off:y_off + h, x_off:x_off + w] = cropped

    # Resize to 20x20
    img20 = Image.fromarray(square).resize((20, 20), Image.LANCZOS)

    # Paste centered on 28x28 canvas
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(img20, (4, 4))

    arr = np.array(canvas).astype("float32") / 255.0
    return arr

def preprocess(img_in: Image.Image) -> np.ndarray:
    """
    1) Remove transparency
    2) Grayscale
    3) Invert so strokes = white, bg = black (MNIST style)
    4) Center & resize to 28x28
    """
    img = rgba_to_white_bg(img_in)
    img = img.convert("L")
    img = ImageOps.invert(img)
    arr = np.array(img)
    arr = center_and_resize_28x28(arr)
    return arr.reshape(1, 28, 28, 1).astype("float32")

def predict(img: Image.Image):
    x = preprocess(img)
    probs = model.predict(x, verbose=0)[0]
    top3 = probs.argsort()[-3:][::-1]
    return top3, probs, x

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("MNIST Digit Classifier 🧠✍️")
tab1, tab2 = st.tabs(["Draw a digit", "Upload an image"])

# ---- Draw tab ----
with tab1:
    st.write("Draw a **single digit (0–9)** below.")
    canvas = st_canvas(
        fill_color="rgba(255,255,255,1)",
        stroke_width=18,
        stroke_color="#000000",
        background_color="#FFFFFF",
        width=256, height=256,
        drawing_mode="freedraw", key="canvas",
    )

    colA, colB = st.columns(2)
    if colA.button("Predict from Canvas") and canvas.image_data is not None:
        img_rgba = Image.fromarray(canvas.image_data.astype("uint8"))
        top3, probs, x = predict(img_rgba)

        st.subheader(f"Prediction: **{top3[0]}**")
        for i in top3:
            st.write(f"{i}: {probs[i]:.4f}")

        st.caption("Model input (28×28 grayscale):")
        st.image((x[0, :, :, 0] * 255).astype("uint8"), width=140, clamp=True)

    if colB.button("Clear Canvas"):
        st.experimental_rerun()

# ---- Upload tab ----
with tab2:
    up = st.file_uploader("Upload a PNG/JPG of a digit", type=["png","jpg","jpeg"])
    if up is not None:
        img = Image.open(up)
        st.image(img, caption="Uploaded", width=224)
        if st.button("Predict from Upload"):
            top3, probs, x = predict(img)
            st.subheader(f"Prediction: **{top3[0]}**")
            for i in top3:
                st.write(f"{i}: {probs[i]:.4f}")
            st.caption("Model input (28×28 grayscale):")
            st.image((x[0, :, :, 0] * 255).astype("uint8"), width=140, clamp=True)