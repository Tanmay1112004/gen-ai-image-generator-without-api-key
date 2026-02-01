# 🎨 CLIP + VQGAN: Latent Space Text-to-Image Synthesis

An end-to-end generative AI pipeline that synthesizes high-fidelity images from natural language descriptions. By leveraging **VQGAN** (Vector Quantized Generative Adversarial Network) for image representation and **CLIP** (Contrastive Language-Image Pre-training) for semantic alignment, this project explores the intersection of computer vision and linguistics.

**[Run on Colab](https://www.google.com/search?q=https://colab.research.google.com/)** | **[View Architecture](https://www.google.com/search?q=%23-how-it-works)** | **[Get Started](https://www.google.com/search?q=%23-quick-start)**

---

## 🚀 Key Features

* **Zero-Shot Generation:** Create unique visuals without task-specific fine-tuning.
* **Intuitive UI:** A fully integrated **Streamlit** dashboard for real-time prompt engineering.
* **Optimized for Colab:** Seamless integration with cloud GPU environments.
* **Local Deployment:** Clean, modular code structure ready for local Python environments.

---

## 🧠 How It Works

This project implements a feedback loop where:

1. **VQGAN** generates a candidate image from the latent space.
2. **CLIP** evaluates how well the image matches the provided text prompt.
3. **Backpropagation** updates the latent vector to minimize the "distance" between the image and the text.

---

## 🛠️ Installation & Setup

### 1. Environment Configuration

Ensure you have a CUDA-enabled GPU for efficient generation.

```bash
git clone https://github.com/your-username/clip-vqgan-text2image.git
cd clip-vqgan-text2image
pip install -r requirements.txt

```

### 2. Model Weights

The system requires pre-trained weights to function. Download and place them as follows:

* **[Download VQGAN Checkpoint](https://heibox.uni-heidelberg.de/f/867b05fc8c4841768640/?dl=1)** → `./models/vqgan_imagenet_f16_16384/checkpoints/last.ckpt`
* **[Download VQGAN Config](https://heibox.uni-heidelberg.de/f/274fb24ed38341bfa753/?dl=1)** → `./models/vqgan_imagenet_f16_16384/configs/model.yaml`

---

## 💻 Usage

### Interactive Web App

Perfect for experimenting with prompts and hyperparameters (learning rate, iterations).

```bash
streamlit run app.py

```

### Google Colab (Headless)

For those without local GPUs, open `GenerativeAI_Colab.ipynb` and follow the sequential cells to generate images directly in the cloud.

---

## 🖼️ Gallery & Examples

| "A cyberpunk city at sunset" | "An oil painting of a cosmic cat" | "Minimalist mountain landscape" |
| --- | --- | --- |
| *(Insert Image)* | *(Insert Image)* | *(Insert Image)* |

---

## 📊 Technical Requirements

* **Hardware:** NVIDIA GPU (Minimum 8GB VRAM recommended).
* **Software:** Python 3.8+, PyTorch 1.10+, Torchvision.
* **Key Libraries:** `ftfy`, `regex`, `tqdm`, `omegaconf`, `pytorch-lightning`.

---

## 🤝 Contributing

Contributions are welcome! If you have ideas for improving the optimization loop or adding new model support (like Stable Diffusion), feel free to fork the repo and submit a PR.

---

## 📜 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---
