# 🎨 CLIP + VQGAN

## Latent Space Text-to-Image Synthesis Engine

> A research-inspired generative AI pipeline that synthesizes high-fidelity images directly from natural language prompts using CLIP-guided latent optimization.

---

## 📌 Project Overview

This project implements a **closed-loop generative optimization system** combining:

* 🧠 **CLIP** — Semantic alignment between text and image
* 🎨 **VQGAN** — High-quality latent space image synthesis
* ⚡ PyTorch-based optimization loop
* 🌐 Streamlit-based real-time UI

Unlike diffusion-based models, this architecture performs **direct latent vector optimization**, demonstrating a strong understanding of representation learning and multimodal alignment.

---

## 🧠 Core Architecture

![Image](https://ljvmiranda921.github.io/assets/png/vqgan/two_stage_v2.png)

![Image](https://storage.googleapis.com/gweb-research2023-media/original_images/414dc401fa628588cd915e3b4b06fcb8-image2.jpg)

![Image](https://miro.medium.com/1%2Ah5xJzfFAfjdysNvqQbB9nQ.png)

![Image](https://www.researchgate.net/publication/376989243/figure/fig3/AS%3A11431281250058236%401717748953759/top-We-show-standard-CLIP-usage-where-an-image-is-embedded-into-a-multi-modal-space.png)

### 🔁 Optimization Workflow

```
Text Prompt
     ↓
CLIP Text Encoder → Text Embedding
     ↓
Initialize Random Latent Vector
     ↓
VQGAN Decoder → Candidate Image
     ↓
CLIP Image Encoder → Image Embedding
     ↓
Compute Similarity Loss
     ↓
Backpropagation on Latent Vector
     ↓
Refined Image Output
```

This iterative loop minimizes the embedding distance between text and generated image.

---

## 🚀 Key Capabilities

### 🧬 Zero-Shot Image Generation

No task-specific fine-tuning required.

### 🎛 Latent Space Control

Adjust:

* Learning rate
* Number of iterations
* Cutout strategy
* Prompt weighting

### 🌐 Dual Deployment Modes

* **Streamlit Web App** (local GPU)
* **Google Colab Notebook** (cloud GPU)

### 🧱 Modular Design

* Separate model loader
* Clean optimization loop
* UI abstraction layer

---

## 🛠 Tech Stack

| Layer                 | Technology |
| --------------------- | ---------- |
| Vision-Language Model | CLIP       |
| Generative Model      | VQGAN      |
| Framework             | PyTorch    |
| UI                    | Streamlit  |
| Config Handling       | OmegaConf  |
| Acceleration          | CUDA       |

---

## 🖼️ Sample Generations

![Image](https://i.etsystatic.com/50530210/r/il/c7ad66/6687247499/il_570xN.6687247499_5mk0.jpg)

![Image](https://media.posterlounge.com/img/products/760000/758668/758668_poster.jpg)

![Image](https://i.etsystatic.com/36147858/r/il/13dd52/4248749593/il_570xN.4248749593_jn16.jpg)

![Image](https://png.pngtree.com/png-clipart/20240416/original/pngtree-vector-landscapes-in-a-minimalist-style-landscape-illustration-flat-design-vector-png-image_14869534.png)

**Prompt Examples:**

* `"A cyberpunk city at sunset"`
* `"An oil painting of a cosmic cat"`
* `"Minimalist mountain landscape"`
* `"Surreal neon futuristic skyline"`

---

## 🛠 Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com//clip-vqgan-text2image.git
cd clip-vqgan-text2image
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Model Weights Setup

Download pretrained VQGAN weights:

* VQGAN Checkpoint →
  `./models/vqgan_imagenet_f16_16384/checkpoints/last.ckpt`

* VQGAN Config →
  `./models/vqgan_imagenet_f16_16384/configs/model.yaml`

(Links provided in repository)

---

## 💻 Usage

### 🌐 Streamlit App (Recommended)

```bash
streamlit run app.py
```

Features:

* Real-time prompt editing
* Adjustable hyperparameters
* Iteration preview updates

---

### ☁️ Google Colab (Headless Mode)

Open:

```
GenerativeAI_Colab.ipynb
```

Run sequential cells to:

* Load models
* Optimize latent vector
* Generate image
* Download output

---

## ⚙️ Technical Requirements

### Hardware

* NVIDIA GPU (8GB+ VRAM recommended)
* CUDA-enabled environment

### Software

* Python 3.8+
* PyTorch 1.10+
* Torchvision

### Key Libraries

```
ftfy
regex
tqdm
omegaconf
pytorch-lightning
streamlit
```

---

## 🧪 Why This Project Stands Out

This project demonstrates:

✔ Deep understanding of multimodal embeddings
✔ Latent space optimization
✔ GAN architecture knowledge
✔ Gradient-based image synthesis
✔ End-to-end AI system deployment

It shows you understand **how generative models work internally**, not just how to call Stable Diffusion.

---

## 🔮 Future Enhancements

* Add Stable Diffusion backend
* Add negative prompt support
* Multi-prompt blending
* Prompt weighting sliders in UI
* Docker containerization
* FastAPI inference API
* Image history gallery

---

## 📂 Project Structure

```
clip-vqgan-text2image/
│
├── app.py
├── GenerativeAI_Colab.ipynb
├── models/
├── utils/
├── requirements.txt
└── README.md
```

---

## 🤝 Contributing

PRs welcome.

Ideas:

* Better cutout strategies
* Faster convergence methods
* Hybrid CLIP guidance approaches
* LoRA-style lightweight tuning

---

## 📜 License

MIT License — open for experimentation and extension.

---

