# MultiMWP: A Multilingual Dataset and Benchmark for Math Word Problem Generation

This repository accompanies the paper:

**“A Multilingual Dataset (MultiMWP) and Benchmark for Math Word Problem Generation”**
Published in *IEEE/ACM Transactions on Audio, Speech and Language Processing*, 2025
[[DOI: 10.1109/TASLPRO.2025.3552936](https://doi.org/10.1109/TASLPRO.2025.3552936)]

---

## 📘 Overview

MultiMWP is a multi-way parallel dataset comprising 7,470 Math Word Problems (MWPs) in 9 languages, including 6 low-resource languages. It also provides baseline and constraint-based methods for MWP generation using pre-trained multilingual language models such as mBART50, mT5, M2M-100, and IndicBART.

---


## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/OmegaGamage/multiMWP.git
   cd multiMWP
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🧪 Experiments (Branch-wise Setup)

Each experiment is separated into its own branch:
- `baseline`: Language-specific model using seed-based MWP generation.
- `baseline-multilingual`: Single multilingual model.
- `constraint-based`: Adds mathematical constraint loss (`mwp2eq`) to each language.
- `constraint-based-multi-lingual`: Multilingual + constraints (best-performing model in the paper).

---

## 🚀 Usage

### Step 1: Prepare Dataset

```bash
python prepare_dataset.py \
  --data_dir '/path/to/raw_dataset/' \
  --dataset_dir '/path/to/processed_data/' \
  --language 'en' \
  --mwp_type 'Simple' \
  --experiment 'A1' \
  --seed_len 0.25
```

### Step 2: Train Model

```bash
python train.py \
  --work_dir '/path/to/working_dir/' \
  --data_dir '/path/to/processed_data/' \
  --language 'en' \
  --mwp_type 'Simple' \
  --experiment 'A1' \
  --seed_len 0.25 \
  --model 'mbart-large-50' \
  --batch_size 1 \
  --num_workers 4 \
  --lr 0.0001 \
  --warmup_steps 1000 \
  --scheduler 'cosine' \
  --epoch 30
```

### Step 3: Evaluate

Use either:
- `eval_score.ipynb`: For notebook-based evaluation
- `cal_metrics.py`: For batch evaluation with CLI

---

## 📊 Dataset Details

- **Languages**: English, Sinhala, Tamil, Hindi, Urdu, Assamese, Oriya, Albanian, Chinese
- **Types**: Simple Arithmetic and Algebraic MWPs
- **Format**: Multi-way parallel, ~2 sentences per MWP, aligned across languages

---

## 📈 Metrics

- **Automatic**: BLEU-4, METEOR (via Hugging Face)
- **Constraint-Aware**: Equation consistency via `mwp2eq`
- **Human Evaluation**: Fluency, adequacy, mathematical validity

---

## 🧠 Models

- `mbart-large-50`
- `mt5-base`
- `m2m100_418M`
- `indic-bartSS`

Download via Hugging Face when running the training script.

---

## 📜 Citation

Please cite our work if you use the dataset or code in your research:

```bibtex
@ARTICLE{10933586,
  author={Gamage, Omega and Ranathunga, Surangika and Lee, Annie and Sun, Xiao and Singh, Aryaveer and Skenduli, Marjana Prifti and Alam, Mehreen and Nayak, Ajit Kumar and Gao, Haonan and Deori, Barga and Ji, Jingwen and Zhang, Qiyue and Zeng, Yuchen and Tian, Muxin and Mao, Yanke and Trico, Endi and Nako, Danja and Shqezi, Sonila and Hoxha, Sara and Imami, Dezi and Doksani, Dea and Pandey, Virat Kumar and Ananya, Ananya and Aggarwal, Nitisha and Hussain, Naiyarah and Dwivedi, Vandana and Sinha, Rajkumari Monimala and Kalita, Dhrubajyoti},
  journal={IEEE Transactions on Audio, Speech and Language Processing},
  title={A Multilingual Dataset (MultiMWP) and Benchmark for Math Word Problem Generation},
  year={2025},
  doi={10.1109/TASLPRO.2025.3552936}
}
```

---

## 🔐 License

This project is licensed under the [MIT License](./LICENSE).
