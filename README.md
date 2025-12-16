Explainable AI-Enabled Network Intrusion Detection System (NIDS)
Course: CIS3004-N Computing Project
Programme: BSc (Hons) Cybersecurity & Networks
Institution: Teesside University (Delivered at MDIS, Singapore)
Student: Shrikant More
________________________________________
1. Project Overview
This project investigates the design and implementation of an AI-enabled Network Intrusion Detection System (NIDS) that integrates Explainable Artificial Intelligence (XAI) techniques to improve transparency and analyst trust.
Traditional intrusion detection systems rely heavily on static rules and predefined attack signatures, which limits their ability to detect novel or evolving threats. While machine learning improves detection capability, most ML-based systems operate as black boxes and provide little insight into why traffic is flagged as malicious.
This project addresses both challenges by combining:
•	Machine learning–based intrusion detection
•	Explainable AI techniques (SHAP & LIME)
•	An interactive SOC-style dashboard
The system processes network traffic data, classifies traffic as benign or malicious, explains model decisions, and visualises results through a professional dashboard interface.
________________________________________
2. Key Objectives
The primary objectives of this project are:
•	To design a machine learning–based NIDS capable of detecting malicious network traffic.
•	To accurately classify traffic using supervised learning models.
•	To integrate Explainable AI methods to interpret model decisions.
•	To present results through an interactive SOC-style dashboard.
•	To ensure the solution is modular, reproducible, and demonstrable.
________________________________________
3. Dataset
•	Dataset Used: CIC-IDS 2017
•	Type: Labelled network traffic dataset
•	Features: Statistical and flow-based network attributes
•	Labels: Binary classification (Benign / Attack)
The dataset is processed offline to avoid ethical and security risks associated with live traffic capture.
________________________________________
4. System Architecture
The project follows a modular pipeline architecture:
Raw Network Data (CSV)
        ↓
Data Preprocessing
        ↓
Machine Learning Training
        ↓
Explainable AI (SHAP & LIME)
        ↓
Streamlit SOC Dashboard
Each stage is implemented as an independent script to improve clarity, maintainability, and reproducibility.
________________________________________
5. Project Structure
NIDS_Project/
│
├── data/
│   └── processed/
│       └── processed_dataset.csv
│
├── models/
│   ├── xgb_model.joblib
│   └── label_encoder.joblib
│
├── outputs/
│   ├── shap_values_summary.png
│   └── lime_explanation.html
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── explain.py
│
├── app/
│   ├── app.py
│   ├── teesside.png
│   └── mdis.png
│
├── run_all.sh
├── requirements.txt
└── README.md
________________________________________
6. Technologies Used
•	Programming Language: Python 3
•	Machine Learning: XGBoost, Scikit-learn
•	Explainable AI: SHAP, LIME
•	Data Processing: Pandas, NumPy
•	Visualisation: Streamlit, Plotly
•	Environment: Kali Linux (Virtual Machine)
•	Version Control: Git & GitHub
________________________________________
7. Installation & Environment Setup (Kali Linux)
7.1 Clone the Repository
git clone https://github.com/<your-username>/ai-nids-cicids2017.git
cd ai-nids-cicids2017
7.2 Create Virtual Environment
python3 -m venv venv
source venv/bin/activate
7.3 Install Dependencies
pip install -r requirements.txt
________________________________________
8. Running the Artefact (Manual Execution)
Run each stage in sequence:
python src/preprocess.py
python src/train.py
python src/explain.py
streamlit run app/app.py
The dashboard will be available at:
http://localhost:8501
________________________________________
9. Automated Execution (Single Command)
For demonstration and assessment purposes, the entire pipeline can be executed using a single script.
chmod +x run_all.sh
./run_all.sh
This script sequentially runs:
1.	Data preprocessing
2.	Model training
3.	Explainability generation
4.	Dashboard launch
________________________________________
10. Dashboard Features
The Streamlit dashboard provides:
•	SOC-style Overview
o	Traffic statistics
o	Attack rate trends
o	Class distribution
•	Dataset Exploration
o	Feature summaries
o	Distribution plots
•	Model Analysis
o	Predictions
o	Feature importance
•	Explainability
o	SHAP global explanations
o	LIME local explanations
•	Live Monitoring (Simulated)
o	Attack probability gauge
________________________________________
11. Explainable AI Integration
•	SHAP:
Provides global feature importance, showing which attributes influence predictions across the dataset.
•	LIME:
Explains individual predictions by approximating model behaviour locally.
These techniques ensure that detection decisions are transparent, interpretable, and trustworthy.
________________________________________
12. Constraints & Limitations
•	Operates on offline datasets only (no real-time traffic).
•	Performance depends on dataset quality and representativeness.
•	Explainability techniques introduce additional computational overhead.
•	Not integrated with live SIEM or production environments.
________________________________________
13. Academic Declaration
This project artefact is submitted as part of the CIS3004-N Computing Project module.
All work presented is original and developed solely for academic purposes.
________________________________________
14. Conclusion
This project demonstrates a complete, working AI-enabled Network Intrusion Detection System that combines accurate detection with explainability and professional visualisation.
The system highlights how Explainable AI can enhance trust and usability in cybersecurity applications and provides a strong foundation for future extensions such as real-time detection and enterprise integration.
________________________________________
15. Contact
Student: Shrikant More
Programme: BSc (Hons) Cybersecurity & Networks
Institution: Teesside University (MDIS)
