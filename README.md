# DefectDetect

### **Predict defects in C programs given various attributes about the code.**
This project employs machine learning techniques to predict defects in C programs. By analyzing code attributes such as complexity, size, and design, the project identifies potential trouble spots in the codebase. This predictive model aids in proactively addressing software defects, ultimately enhancing code quality and reliability.

**Data Source: [Kaggle](https://www.kaggle.com/datasets/semustafacevik/software-defect-prediction)</br></br>**
**⚠ Please read the text in `about JM1 Dataset` located in `dataset/raw/`**

---
**Some of the models and techniques used:**
- Logistic Regression
- KNN
- Random Forest Classifier
- LightGBM Classifier
- Xgboost Classifier
- CatBoost Classifier  
- Hill Climb Ensemble
- Voting Ensemble
#### The app is deployed --> [here](https://defectdetect-2hbgzrrmgjhrqm4wjlolbw.streamlit.app/) <-- on the Streamlit Community Cloud.

---

### Data Description:

**1. loc (McCabe's line count of code):**

- This feature represents the number of lines of code in a program, according to McCabe's complexity metric.
- It helps measure the program's size and complexity.

**2. v(g) (McCabe "cyclomatic complexity"):**

- Cyclomatic complexity is a software metric that measures the number of linearly independent paths through a program's source code.
- This feature quantifies the program's structural complexity.

**3. ev(g) (McCabe "essential complexity"):**

- Essential complexity is a measure of complexity in software, focusing on the essential parts that impact maintainability and comprehension.
- It helps identify essential parts of the codebase.

**4. iv(g) (McCabe "design complexity"):**

- Design complexity is a metric that measures the complexity of a program's design.
- It can provide insights into the overall design of the code.

**5. n (Halstead total operators + operands):**

- Halstead metrics are used to measure software complexity.
- This feature represents the total number of operators and operands in the code.

**6. v (Halstead "volume"):**

- Halstead's volume metric quantifies the size and complexity of a program.
- A higher volume suggests a more complex program.

**7. l (Halstead "program length"):**

- Program length in Halstead metrics measures the length or size of the program.
- It is related to the total number of operators and operands.

**8. d (Halstead "difficulty"):**

- Difficulty is a measure of how hard it is to understand a program.
- A higher difficulty indicates a more challenging codebase.

**9. e (Halstead "effort"):**

- Effort is a Halstead metric that quantifies the effort required to understand and maintain the program.
- It combines volume and difficulty.

**10. b (Halstead's time estimator):**

- Halstead's time estimator estimates the time required to understand and modify a program.
- It is a useful metric for project planning.

**11. t (Halstead's time estimator):**

- Halstead's time estimator is a metric that estimates the time required to understand and modify a program.
- It takes into account the program's volume, difficulty, and effort, providing an estimate of the time needed for software development and maintenance tasks.

**12. lOCode (Halstead's line count):**

- This feature represents the line count of code, according to Halstead metrics.
- It measures the total number of lines in the code.

**13. lOComment (Halstead's count of lines of comments):**

- This feature quantifies the number of lines containing comments in the code.
- It's useful for understanding code documentation.

**14. lOBlank (Halstead's count of blank lines):**

- Blank lines are lines in the code that do not contain any characters.
- This feature measures the number of such lines.

**15. lOCodeAndComment:**

- This feature represents the total number of lines containing both code and comments.
- It can help identify areas where code and comments coexist.

**16. uniq_Op (unique operators):**

- This feature quantifies the number of unique operators in the code.
- Unique operators represent distinct programming operations.

**17. uniq_Opnd (unique operands):**

- Unique operands refer to distinct data items or variables used in the code.
- This feature counts the number of unique operands.

**18. total_Op (total operators):**

- This feature represents the total number of operators in the code.
- It provides an overview of the types of operations performed.

**19. total_Opnd (total operands):**

- Total operands are the total number of data items or variables used in the code.
- This feature quantifies the number of operands.

**20. branchCount:**

- Branch count is a metric that measures the number of branches or decision points in the code.
- It helps assess the program's control flow complexity.
  
