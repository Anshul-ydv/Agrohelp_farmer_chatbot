# AGROHELP – AI-Powered Multilingual Farming Assistant

AGROHELP is a modular, AI-powered, **multilingual digital assistant for farmers**, providing timely, contextual, and sustainable agricultural support. Built with a blend of **computer vision**, **natural language understanding**, and **neural machine translation**, it bridges the knowledge gap between cutting-edge agronomic science and smallholder farmers in linguistically diverse rural India.

> "Empowering Indian farmers with AI – one query, one diagnosis, one season at a time."

---

## Table of Contents

* [Table of Contents](#-table-of-contents)
* [Features](#-features)
* [Architecture](#-architecture)
* [Project Structure](#-project-structure)
* [Setup & Installation](#️-setup--installation)
* [How to Use](#-how-to-use)
* [Results](#-results)
* [Exploratory Data Analysis](#-exploratory-data-analysis)
* [Future Work](#-future-work)
* [References](#-references)
* [Team](#-team)

---

## Features

* **Crop Disease Detection** using CNN-based image analysis
* **Multilingual Query Understanding** via Transformer-based NLU
* **Bi-Directional Translation Engine** for English, Hindi, Marathi, Punjabi, and more
* **Season-Aware Advisory** based on Indian agro-climatic calendars
* **Sustainable Farming Knowledge Base** for eco-friendly and adaptive agriculture
* **Farmer-Centric Interface** for real-time, local-language interaction

---

## Architecture

AGROHELP follows a modular design, composed of:

1. **Vision Module (`vision_model.py`)**
   Detects crop diseases using convolutional neural networks trained on annotated leaf/stem images.

2. **Language Understanding (`language_model.py`)**
   Processes multilingual farmer queries using intent classification and entity recognition with fine-tuned transformer models.

3. **Translation Engine (`translation_model.py`)**
   Converts queries and responses between English and Indian vernaculars using seq2seq models with attention mechanisms.

4. **Knowledge Base**
   Seasonally-aligned agronomic insights on sustainable farming techniques.

5. **Main Orchestration (`__init__.py`)**
   Connects all components for seamless interaction and real-time response generation.

---

## Project Structure

```bash
AGROHELP/
├── data/                        # Datasets (images, farmer queries)
├── vision_model.py             # CNN-based plant disease detection
├── language_model.py           # Transformer-based NLU
├── translation_model.py        # Seq2Seq model with attention
├── utils/                      # Preprocessing, augmentation, helpers
├── __init__.py                 # Module orchestration
├── AGROHELP.ipynb              # Jupyter Notebook for demo/inference
└── README.md                   # You are here
```

---

## Setup & Installation

### Prerequisites

* Python 3.8+
* PyTorch or TensorFlow
* Transformers (`HuggingFace`)
* OpenCV
* Flask / Streamlit (for UI, optional)

### Create Environment

```bash
git clone https://github.com/yourusername/AGROHELP.git
cd AGROHELP
pip install -r requirements.txt
```

---

## How to Use

1. **Disease Detection from Image**

   ```python
   from vision_model import detect_disease
   detect_disease("sample_leaf.jpg")
   ```

2. **Query Understanding & Translation**

   ```python
   from language_model import parse_query
   from translation_model import translate_query

   translated = translate_query("मिर्ची के लिए कीट नियंत्रण कैसे करें?", target_lang='en')
   response = parse_query(translated)
   ```

3. **Get Advisory**

   * Input: Image or text query
   * Output: Language-specific diagnosis and recommendation

---

## Results

| Module                 | Accuracy/Score                            |
| ---------------------- | ----------------------------------------- |
| Disease Detection      | **87.2%**                                 |
| Intent Classification  | **91%**                                   |
| Entity Extraction      | **88%**                                   |
| Translation BLEU Score | **38–45**                                 |
| User Satisfaction      | **94%** reported clear & relevant replies |

---

## Exploratory Data Analysis

* 10,000+ Images of crop diseases
* Multilingual Queries (60% Hindi, 25% Marathi, rest regional)
* Applied data augmentation to balance classes
* Used NER to extract crop names, symptoms, and seasons

> Query Trends:
>
> * 32%: Disease diagnosis
> * 28%: Sowing schedules
> * 20%: Organic treatments
> * 15%: Soil/water advice

---

## Future Work

* Voice input support for low-literacy users
* Real-time weather-based advisory
* Yield prediction using temporal crop data
* Deployment on mobile and cloud platforms

---

## References

1. [Ferentinos (2018)](https://doi.org/10.1016/j.compag.2018.01.009) – Deep learning for plant disease detection
2. [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762) – Transformer architecture
3. [FAO e-Agriculture Guide (2016)](https://www.fao.org/3/i5564e/i5564e.pdf)

*Complete list in the report.*

---

## Team

* **Anshul Yadav** – Vision + NLP
* **Shruti Bajpayee** – EDA + Knowledge Base
* **Saransh Bhargava** – Translation + Integration

📧 [anshul.yadav.22cse@bmu.edu.in](mailto:anshul.yadav.22cse@bmu.edu.in)
📧 [shruti.bajpayee.22cse@bmu.edu.in](mailto:shruti.bajpayee.22cse@bmu.edu.in)
📧 [saransh.bhargava.22cse@bmu.edu.in](mailto:saransh.bhargava.22cse@bmu.edu.in)
