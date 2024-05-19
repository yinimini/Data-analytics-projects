import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


symp_file = pd.read_excel("Menstration_symp.xlsx")

df = pd.DataFrame(symp_file)

#Adding days_in_cycle column into DataFrame
cycles = df['Cycle'].tolist()
day = 1
#Create an empty list to store the day counts for each cycle
cycle_days = []  
prev_cycle = cycles[0]

for cycle in cycles:
    if prev_cycle == cycle:
        cycle_days.append(day)
    else:
        day = 1
        prev_cycle = cycle
        cycle_days.append(day)
    day += 1

df['Days_in_cycles'] = pd.DataFrame(cycle_days)


#Visualization of recurrence of symptoms
df.plot(x='Days_in_cycles', y=['Cramps','Tender_breasts','Low_energy','Headache', 'Abdominal_pain','Nausea'], kind='bar')
plt.xlabel('Day in each cycle, total cycles = 7')
plt.ylabel('1=YES, 0=NO')
plt.title('Recurrence of symptoms')
plt.legend()
plt.show()

#Calculate probabilities of symptom occurrence
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

#Visualization of probabilities of symptom occurence
plt.bar(symp_list, symp_probability_1)
plt.xlabel('Symptoms')
plt.ylabel('Probability')
plt.title('Probabilities of symptom occurrence')
plt.show()

Menstration_symp_1 = 'Menstration_symp_1.xlsx'
df.to_excel(Menstration_symp_1)




