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




