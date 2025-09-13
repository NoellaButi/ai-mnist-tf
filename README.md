# MNIST Digit Classifier 🧠✍️  
Classify hand-drawn digits (0–9) using a Convolutional Neural Network trained on MNIST  

![Language](https://img.shields.io/badge/language-Python-blue.svg)  
![Notebook](https://img.shields.io/badge/tool-Jupyter-orange.svg)  
![App](https://img.shields.io/badge/app-Streamlit-red.svg)  
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)  

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-mnist-tf-noellabuti.streamlit.app)  

---

✨ **Overview**  
This project trains a CNN on the MNIST dataset (handwritten digits) and achieves ≈99% test accuracy.  
It includes model training, saved artifacts, evaluation reports, and an interactive **Streamlit app** where you can **draw or upload a digit** to classify it.  

🛠️ **Workflow**  
- Load MNIST dataset from TensorFlow  
- Preprocess images (normalize, center, resize, invert)  
- Train a baseline CNN to >98.5% accuracy  
- Save trained model & metrics as artifacts  
- Deploy an interactive Streamlit app  

📁 **Repository Layout**  
```bash
app/            # Streamlit app
artifacts/      # saved model
reports/        # metrics, plots, confusion matrix
notebooks/      # training notebook(s)
src/            # optional training scripts
requirements.txt
README.md
LICENSE
```

🚦 **Demo**

Train model:

```bash
python notebooks/train_mnist.py
```

Run Streamlit app:
```bash
streamlit run app/app.py
```

🔍 **Features**

- Convolutional Neural Network (CNN) built with TensorFlow/Keras
- Preprocessing pipeline: invert → crop → center → resize (MNIST-style)
- ≈99% test accuracy on MNIST
- Interactive Streamlit app (draw or upload digits)
- Metrics & plots saved as artifacts

🚦 **Results (Held-Out Test Set)**
```bash
Metric          Value
-----------------------
Test Accuracy   99.11%
Test Loss       0.0266
```

🚀 **Deployment**

This app is deployed on Streamlit Cloud:
👉 [Try it here](https://ai-mnist-tf-noellabuti.streamlit.app)  

📜 **License**

MIT (see [LICENSE](LICENSE))

---
