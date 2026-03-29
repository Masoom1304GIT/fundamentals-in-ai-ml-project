 AI Loan Approval System

 Made by: Masoom Ali
 About the Project

This project is a simple AI Loan Approval System that I built using Machine Learning.
The main idea is to check whether a person’s loan should be approved or not based on some basic details like income, credit score, and employment status.

In real life, banks do this process manually, which takes time and can sometimes be unfair. So, this project shows how we can use AI to make the process faster and more efficient.

 How It Works

The system uses a machine learning model (Random Forest) to learn from the data and then make predictions.

Basically, the flow is:

Load the data
Process it a bit (add useful features)
Train the model
Test its performance
Show graphs for better understanding
Take user input and predict result
 What Data is Used

I created a small dataset myself just for learning purposes.

It includes:

Income → how much the person earns
Credit Score → financial reliability
Age → age of the person
Loan Amount → requested loan
Employment → job status (1 = yes, 0 = no)
Approved → final decision

I also added one extra feature:

 Income-to-Loan Ratio
This helps the model understand if the person can repay the loan easily or not.

 Tools & Technologies

I used:

Python
NumPy
Pandas
Matplotlib
Scikit-learn
  Model Info
Model used: Random Forest Classifier
Data split: 80% training / 20% testing
Performance checked using:
Accuracy
Confusion Matrix
 Graphs in the Project

The program shows some graphs to make things clear:

Income vs Loan Approval
Credit Score vs Loan Approval
Feature Importance

These graphs help in understanding how decisions are made.

 How to Run
1. Install libraries
pip install numpy pandas matplotlib scikit-learn
2. Run the file
python your_file_name.py
3. Enter details

You’ll be asked to enter:

Income
Credit Score
Age
Loan Amount
Employment status
4. Get result

The system will tell:

✅ Loan Approved
❌ Loan Rejected
 Example
Income: 60000  
Credit Score: 720  
Age: 28  
Loan Amount: 200000  
Employment: 1  

Output: Loan Approved
 What’s Good About This Project
Easy to understand
Good for beginners in ML
Shows real-life use of AI
Fast predictions
 Limitations
Dataset is small and not real
Real bank systems are much more complex
Fewer features used
 What Can Be Improved
Use real-world data
Add more features (like past loans, education, etc.)
Make a website or app
Use more advanced models
 What I Learned

While making this project, I learned:

How ML models work
Data preprocessing
Model training and testing
Creating graphs for analysis
 Final Note

This is a simple project, but it clearly shows how AI can be used in real-world problems like loan approval. With more improvements, this idea can be turned into a full real-world application.


