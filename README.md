# CSE4063 Data Mining Project 2

## Dataset

The dataset used in this project can be found at [this link](https://www.kaggle.com/code/marcogdepinto/let-s-discover-more-about-the-olympic-games).

It was scraped from [www.sports-reference.com](https://www.sports-reference.com) in May 2018. The dataset contains 271,000 rows and 15 columns. Each row represents an individual athlete competing in a specific Olympic event.

## How to Run

1. Ensure you have the dataset file named "athlete_events.csv" in "data/" directory and all the required dependencies installed. You can install them using:

   ```sh
   pip install -r requirements.txt
   ```

2. Run the main script:

   ```sh
   python src/main.py
   ```

3. Follow the on-screen instructions to proceed through the program.

## Folder Structure

```
CSE4063-Data-Mining-Project2/
├── data/                   # Contains the dataset files (ignored by git since its size is large)
│   ├── athlete_events.csv  # Each row corresponds to an individual athlete competing in an individual Olympic event
│   ├── noc_regions.csv     # Contains NOC (National Olympic Committee 3 letter code) and region equivalents
├── src/                    # Source code files
│   ├── main.py             # Main script to run the project
├── .gitignore              # Gitignore file
├── requirements.txt        # List of dependencies
└── README.md               # Project overview and instructions
```

## Project Phases

### **Phase 1: Data Preprocessing (Deadline: Week 2)**

1. **Dataset Understanding**:
   - Explore the dataset (including data types, missing values, and distributions).
   - Summarize its characteristics (number of rows, columns, and unique features).
2. **Cleaning and Transformation**:
   - Handle missing values through imputation or removal of rows/columns.
   - Normalize or scale the data as needed for clustering algorithms.

---

### **Phase 2: Model Implementation (Deadline: Week 3)**

1. **Frequent Pattern Mining Algorithms**:
   - Implement **Apriori**, **FP-Growth**, and **ECLAT** algorithms.
   - Use the `mlxtend` library for Apriori and FP-Growth implementation.
   - Create a custom implementation for ECLAT if no library is available.
2. **Clustering Algorithms**:
   - Implement **K-Means**, **AGNES (Hierarchical Clustering)**, and **DBSCAN**.
   - Use the `scikit-learn` library for these clustering techniques.
3. **Document Each Step**:
   - Maintain detailed documentation and comments throughout implementation.

---

### **Phase 3: Evaluation & Comparison (Deadline: Week 4)**

1. **Frequent Pattern Mining**:
   - Evaluate performance using runtime, number of patterns discovered, and interpretability.
   - Discuss findings for each algorithm.
2. **Clustering**:
   - Evaluate clusters using silhouette score, inertia, and adjusted Rand index.
   - Compare clustering quality and runtime across methods.
3. **Comparison with Literature**:
   - Research papers relevant to your dataset and algorithms.
   - Compare your findings with published results.

---

### **Phase 4: Presentation Preparation (Deadline: Week 5)**

1. **Structure**:
   - **Problem Definition**: State the problem and dataset significance clearly.
   - **Dataset**: Present dataset details, preprocessing steps, and visualizations.
   - **Implementation**: Explain each algorithm and its implementation concisely.
   - **Results & Comparisons**: Highlight key findings and comparisons.
   - **Conclusion**: Summarize insights and recommendations.
2. **Demo**:
   - Prepare a walkthrough of your Python implementations.
   - Test all scripts to ensure smooth execution.

---

### **Phase 5: Submission (Deadline: 03.01.2025)**

1. **Files to Submit**:
   - Python scripts.
   - Presentation file (PPT or PDF).
   - `we_swear.txt` file with the required declaration.
2. **Submission Format**:
   - Zip all files as `GrRepStudentNumber_P2.zip`.
   - Verify the file integrity and submit via Google Classroom.

---

### **Milestone Deadlines**

| **Milestone**            | **Deadline** |
| ------------------------ | ------------ |
| Data Preprocessing       | 18.12.2024   |
| Model Implementation     | 25.12.2024   |
| Evaluation & Comparison  | 29.12.2024   |
| Presentation Preparation | 01.01.2025   |
| Submission               | 03.01.2025   |
