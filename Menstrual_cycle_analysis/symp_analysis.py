import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


nr_of_cycle = [0, 1, 2, 3, 4, 5, 6]

symp_file = pd.read_excel("Menstration_symp.xlsx")

df = pd.DataFrame(symp_file)

df.plot(x='Cycle', y=['Cramps','Tender_breasts','Low_energy','Headache'], kind='bar')
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

for i in symp_list:
    symp_probability = calculate_symps_probability(df[i])
    print('Probability of getting', i, 'during', len(df[i]), 'days/7 cycles is', round(symp_probability,3)*100, '%')


