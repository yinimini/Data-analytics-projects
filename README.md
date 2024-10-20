# Data-analytics-projects
About me: Just a curious girl who tries to find answers in data, I mainly use Python and Power BI to collect, process, and visualize the data. Please click on any projects that catch your interest :)

Here are some of the projects that I have picked out:

# Menstrual cycle analysis
Tool: Python

I wondered: 

1) What symptoms am I getting during my cycles and when am I getting them in the cycles?

2) How long are my cycles and are the durations in the 'normal' range?

3) What's the probabilities of getting the symptoms again? 

I collected and modified the raw data, and visualized the answers to my questions with Python. The analysis focuses on the duration of cycles and periods, as well as the recurrence of symptoms. Probabilities of symptom occurrence are also calculated:

Code file 1:[https://github.com/yinimini/Data-analysis-projects/blob/main/Menstrual_cycle_analysis/Cycle_length.py]

Code file 2:[https://github.com/yinimini/Data-analysis-projects/blob/main/Menstrual_cycle_analysis/symp_analysis.py]

 <img src="https://github.com/yinimini/Data-analysis-projects/assets/32144515/2a6e4681-1a51-4b51-951a-bb8181fa56b9" width="500">
 <img src="https://github.com/yinimini/Data-analysis-projects/assets/32144515/9079f102-9149-48bb-b96b-a2d92b0fa582" width="500">
 <img src="https://github.com/yinimini/Data-analysis-projects/assets/32144515/2f8d5179-62d0-40fb-b715-b1a454b2750b" width="500">

# Credit Card Fraud Detection 
Tool: Python

I wondered: 

1) How do the finance bros detect fraudulent credit card transactions and can I do it?

2) Which library is faster and better for this task: Scikit-Learn or Snap ML?

3) How do different classification models compare in terms of accuracy and efficiency?

I used a real dataset from European cardholders' transactions in September 2013 and implemented and trained two classification models: Decision Tree and Support Vector Machine (SVM). To see how well the models did, I checked their training times, ROC-AUC scores, hinge loss metrics and confusion matrices. Spoiler alert: Snap ML was super fast in training time and just as accurate as Scikit-Learn!

Dataset:[https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud]

Python file: [https://github.com/yinimini/Data-analytics-projects/blob/main/Credit_card_fraud_detection/credit_card_fraud_detection.py]

<img src="https://github.com/yinimini/Data-analytics-projects/blob/main/Credit_card_fraud_detection/Evaluation_plots/ROC_curves.png" width="400">

<img src="https://github.com/yinimini/Data-analytics-projects/blob/main/Credit_card_fraud_detection/Evaluation_plots/training_times.png" width="400">

<img src="https://github.com/yinimini/Data-analytics-projects/blob/main/Credit_card_fraud_detection/Evaluation_plots/scikitlearn.png" width="400">

<img src="https://github.com/yinimini/Data-analytics-projects/blob/main/Credit_card_fraud_detection/Evaluation_plots/snnpml.png" width="400">

<img src="https://github.com/yinimini/Data-analytics-projects/blob/main/Credit_card_fraud_detection/Evaluation_plots/scikitlearn_svm.png" width="400">

<img src="https://github.com/yinimini/Data-analytics-projects/blob/main/Credit_card_fraud_detection/Evaluation_plots/snapml_svm.png" width="400">

# Mental Health Disorders Analysis
Tool: Power BI

I wonder: 

1) How are the mental health situations in China and Norway over the years?

2) What's the relationship between psychological disorders and STEM students' grades?

3) What are the factors that could contribute to the formation of a psychological disorder, and what further impact does it have on students' grades?

I collected three different datasets fra Kaggle, mind that these datasets are probably not big enough to present accurate results, but it does reflect some of the aspect in the problems. 

Power BI file:[https://github.com/yinimini/Data-analysis-projects/blob/main/Mental_health_disorder_analysis/Mental_health_analysis.pbix]

 <img src="https://github.com/yinimini/Data-analysis-projects/assets/32144515/74597600-5d17-40db-89f7-b1acc9cfc871" width="500">
 <img src="https://github.com/yinimini/Data-analysis-projects/assets/32144515/fbc4dfcc-a111-4085-bedf-d12c2f698316" width="500">

# Frequency Delta Sigma Modulator (FDSM) performance analysis
Tool: Python 

I wondered:

1)Does this modulator work at all, if so, does it behave like a oversampling converter as it should?

2)How well does this modulator perform, and how does its performance compare to that of other oversampling converters in the research field?

I collected the large data file from the modulator, rinsed the data that includes the offset, then visualized the results in both time and frequency domains. I also calculated and visualized performance parameters such as linearity, SQNR, SINAD, THD, and gate switching time.

Code: [https://github.com/yinimini/Data-analysis-projects/blob/main/FDSM_performance_analysis/FFT_1.py]

<img src="https://github.com/yinimini/Data-analysis-projects/assets/32144515/dd2ee323-a03d-4541-849b-a2ca76be7249" width="280"> 
<img src="https://github.com/yinimini/Data-analysis-projects/assets/32144515/503f5545-40ab-4f82-8c60-04e622527002" width="300">
<img src="https://github.com/yinimini/Data-analysis-projects/assets/32144515/f55befed-3441-4e3c-8c43-b2483786e2ca" width="300"> 
<img src="https://github.com/yinimini/Data-analysis-projects/assets/32144515/da50c3c8-fdd6-43e5-9f98-d7fcf940a127" width="500"> 






