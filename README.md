# 📰 NewsDetector

A simple **Fake News Detection** web app built with **Python, Streamlit, and Machine Learning**.  
Paste any news headline or text, and the model predicts whether it is **Fake** or **Real**.  

---

## 🚀 Quick Start  

### 1. Clone this repo  
```bash
git clone https://github.com/meetbhayani/NewsDetector.git
cd NewsDetector
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Train the model (optional)
```bash
python FakeNewstrain-model.py
```

### 4. Run the web app
```bash
python -m streamlit run app.py
```

Now open 👉 http://127.0.0.1:5000/ in your browser.


## 📂 Project Files
  - app.py → Streamlit app (main entry point)
  - fakenews.py → Prediction logic
  - FakeNewstrain-model.py → Script to train model
  - real-news.csv → Dataset sample
  - requirements.txt → Dependencies
  - Procfile / runtime.txt → For deployment (Heroku-ready)


## ⚙️ Tech Stack  

- **Python 3.x** – Core programming language  
- **Pandas** – Data loading, preprocessing, and handling CSV datasets  
- **PyTorch** – Deep learning framework used for model training and evaluation  
- **Torch Dataset API** – Custom dataset class for managing input sequences and labels  
- **Transformers (Hugging Face)** –  
  - `BertTokenizer` → Converts raw text into tokens suitable for BERT  
  - `BertForSequenceClassification` → Pre-trained BERT model fine-tuned for fake news detection  
  - `Trainer` / `TrainingArguments` → Simplifies training and evaluation loops  
- **scikit-learn** – Evaluation metrics like `classification_report` and `confusion_matrix`  
- **NumPy** – Numerical computations and efficient array handling  

---

## 🧠 How It Works  

1. **Dataset Preparation**  
   - Load the dataset (CSV) using **Pandas**.  
   - Preprocess text and assign labels (e.g., *Fake* = 0, *Real* = 1).  

2. **Tokenization**  
   - Use **BERT Tokenizer** to convert raw text into token IDs.  
   - Pad or truncate sequences to a fixed length for consistency.  

3. **Custom Dataset**  
   - Build a **PyTorch Dataset** class that returns `(input_ids, attention_mask, labels)` for each sample.  

4. **Model**  
   - Load **BERT for Sequence Classification** (binary classification head).  
   - Fine-tune it on the fake news dataset.  

5. **Training**  
   - Configure **TrainingArguments** (batch size, learning rate, epochs).  
   - Use **Trainer API** to handle training loop and evaluation.  

6. **Evaluation**  
   - Generate predictions on validation/test sets.  
   - Evaluate using **classification report** (precision, recall, F1) and **confusion matrix** for detailed performance.  

7. **Output**  
   - Final model predicts whether unseen text is **“Fake News”** or **“Real News.”**  



## 🌍 Deployment
  - This project is ready for cloud deployment.
  - Has Procfile and runtime.txt → works on Heroku out of the box.


## 📜 License
  - Open-source — feel free to use and modify!
