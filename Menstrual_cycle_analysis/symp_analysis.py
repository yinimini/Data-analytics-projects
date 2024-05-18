import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


symp_file = pd.read_excel("Menstration_symp.xlsx")

df = pd.DataFrame(symp_file)

df.plot(x='Cycle', y=['Cramps','Tender_breasts','Low_energy','Headache', 'Abdominal_pain','Nausea'], kind='bar')
plt.xlabel('Cycle in days')
plt.ylabel('1=positive, 0=negative')
plt.title('Recurrence of symptoms')
plt.legend()
plt.show()

def calculate_symps_probability(symptom_list):
    positive_symp = []

    for i in symptom_list:
        if i == 1:
            positive_symp.append(i)

    probability = len(positive_symp) / len(symptom_list)
    return probability

symp_list = df.columns.tolist()[2:] 
print(symp_list)

symp_probability_1 = []
for i in symp_list:
    symp_probability = calculate_symps_probability(df[i])
    symp_probability_1.append(symp_probability)
    print('Probability of getting', i, 'during', len(df[i]), 'days/7 cycles is', round(symp_probability,3)*100, '%')

plt.bar(symp_list, symp_probability_1)
plt.xlabel('Symptoms')
plt.ylabel('Probability')
plt.title('Probabilities of symptom occurrence')
plt.show()

