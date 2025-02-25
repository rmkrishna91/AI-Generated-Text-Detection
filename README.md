### **LLM-Generated Text Detection**  

#### **Overview**  
This project aims to classify text as either **human-written (0)** or **AI-generated (1)** using a combination of **linguistic features** and **deep learning embeddings**. It utilizes **Sentence Transformers** for semantic text embeddings and **PyTorch-based neural networks** for classification.  

#### **Features Used for Detection**  
The model extracts the following linguistic and syntactic features from the input text:  

- **General Text Properties:**  
  - Average line length  
  - Vocabulary size  
  - Word density  
  - Word count  
  - Sentence count  

- **Linguistic & POS Tag Features:**  
  - Count of **nouns, verbs, adjectives, adverbs, pronouns, etc.**  
  - Number of **active vs. passive sentences**  
  - Count of **stopwords**  
  - Number of **punctuation marks**  
  - Linking words count (e.g., *and, the, to*)  

- **Deep Learning-Based Features:**  
  - Sentence embeddings using **all-mpnet-base-v2** model  
  - Feature standardization using **StandardScaler**  
  - Predictions from a **fully connected neural network**  

#### **Technologies Used**  
- **Python**  
- **PyTorch**  
- **NLTK** & **SpaCy** for NLP processing  
- **Sentence Transformers** for embedding generation  
- **Scikit-learn** for feature scaling & preprocessing  

#### **Model Architecture**  
The classification model consists of:  
- **Input layer** (795 features: 768 from embeddings + 27 numerical features)  
- **Fully connected layers** with batch normalization and dropout  
- **Final output layer** with sigmoid activation for binary classification  

#### **Installation & Setup**  
1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/ai_text_detection.git
   cd ai_text_detection
   ```  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  
3. Download necessary NLP models:  
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('averaged_perceptron_tagger')
   spacy.cli.download("en_core_web_sm")
   ```  
4. Ensure the trained model files (`classification_model1.pth`, `scaler_mean.npy`, `scaler_scale.npy`) are placed in the correct directory.  

#### **Usage**  
Run the inference script to classify text:  
```python
from your_script_name import predict

text = "She typed 'I miss you' but deleted it before hitting send."
predictions, probabilities = predict([text], final_df)

print(f"Prediction: {'AI-generated' if predictions[0] == 1 else 'Human-written'}")
print(f"Confidence Score: {probabilities[0]}")
```  

#### **Example Output**  
```
Prediction: Human-written
Confidence Score: 0.23
```  

#### **Future Improvements**  
- Expand dataset with more diverse AI-generated texts  
- Fine-tune deep learning models for better accuracy  
- Develop a web-based interface for real-time text analysis  

🚀 **Contributions & feedback are welcome!** 🎉

