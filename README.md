# 🥗 **SmartFoodSelector**

<div align="center">
  <img src="https://github.com/byluca/SmartFoodSelector/blob/main/food.png" alt="SmartFoodSelector Logo" width="500" height="500">
</div>

---

## 🍴 **Make Smarter Food Choices with AI**  
**SmartFoodSelector** revolutionizes food analysis and recommendation using cutting-edge AI! 🚀 This modular system combines machine learning, Bayesian inference, and symbolic logic to:  
- 🧹 Preprocess and normalize nutritional data.  
- 🎯 Cluster products into meaningful groups via **k-Means**.  
- 🤖 Predict categories for new items with **Random Forests** and **Logistic Regression**.  
- 🔮 Model probabilistic relationships with **Bayesian Networks**.  
- 🧩 Execute logical queries via **Prolog** knowledge bases.  

---

## 🌟 **Key Features**

### 🧹 **1. Data Preprocessing**  
- **Cleaning**: Handle missing values and outliers.  
- **Feature Selection**: Focus on core nutritional metrics (`energy_100g`, `fat_100g`, etc.).  
- **Normalization**: Scale features using `MinMaxScaler` for balanced analysis.  

### 🎯 **2. Unsupervised Clustering (k-Means)**  
- **Elbow Method**: Automatically determine optimal clusters with `kneed`.  
- **Visualization**: Explore cluster distributions via pie charts and PCA-reduced plots.  
- **Output**: Generate `clustered_dataset.csv` for downstream tasks.  

### 🤖 **3. Supervised Learning**  
- **Models**: Train `Decision Trees`, `Random Forests`, and `Logistic Regression`.  
- **Balancing**: Address class imbalance with **SMOTE**.  
- **Evaluation**: Compare metrics (Accuracy, F1-score, Precision/Recall) and interpret results via **SHAP** values.  
- **Learning Curves**: Diagnose overfitting/underfitting.  

### 🔮 **4. Bayesian Networks**  
- **Continuous & Discrete Models**: Learn probabilistic dependencies with `pgmpy`.  
- **Inference**: Predict preferences and handle missing data.  
- **Visualization**: Plot dependency graphs for interpretability.  

### 🧩 **5. Prolog Knowledge Base**  
- **Automated Generation**: Convert clustered data into Prolog facts/rules.  
- **Query Interface**: Use `pyswip` to execute logical rules like:  
  - `product_info(E, F, C, Su, P, Sa, Cluster)` for cluster lookup.  
  - Custom constraints (e.g., `high_protein_low_sugar`).  

---

## 🛠️ **Architecture**  
```bash
src/
├── dataset_preprocessing.py    # Data cleaning/normalization
├── unsupervised_clustering.py  # k-Means + Elbow Method
├── supervised_trainer.py       # Model training/evaluation
├── bayes_net.py                # Bayesian Networks
├── prolog_interface.py         # Prolog query handler
└── generate_prolog_knowledge_base.py  # Prolog KB generator
main.py                         # Pipeline coordinator
```

---

## 📦 **Get Started**  
1. **Install Dependencies**:  
   ```bash
   pip install pandas scikit-learn pgmpy pyswip optuna shap kneed
   ```
2. **Preprocess Data**:  
   ```python
   python main.py --preprocess
   ```
3. **Run Clustering**:  
   ```python
   python main.py --cluster
   ```
4. **Train Models**:  
   ```python
   python main.py --train
   ```
5. **Launch Prolog KB**:  
   ```python
   python main.py --prolog
   ```

---

## 📊 **Results & Observations**  
- **Clustering**: Optimal `k=3` clusters identified via elbow method.  
- **Supervised Models**: Random Forest achieved highest accuracy (92%).  
- **Bayesian Networks**: Enabled probabilistic inference under missing data.  
- **Prolog Integration**: Efficiently answered complex nutritional constraints.  

---

## 🤝 **Contribute**  
- Report issues or suggest enhancements via **GitHub Issues**.  
- Submit **Pull Requests** for bug fixes/new features.  

---

## 📜 **License**  
MIT License. See [LICENSE](LICENSE) for details.  

---

🌟 **Empower your food decisions with AI and logic!** 🌟  

For full technical details, refer to the [project documentation](Documentazione.pdf).  

---

**Key Updates**:  
- Added architectural overview and setup instructions.  
- Expanded technical details while retaining engaging tone.  
- Linked to full documentation for deeper exploration.  
- Simplified CLI commands for ease of use.
