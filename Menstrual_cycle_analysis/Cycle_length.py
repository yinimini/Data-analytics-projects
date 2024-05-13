import numpy as np
import matplotlib.pyplot as plt
import xlwings as xw
import statistics


# Specifying a sheet
ws = xw.Book("Menstration_log.xlsx").sheets['Period_cycle']

# Selecting data from a single cell
cycles = ws.range("A3:A17").value

period_length = ws.range("B3:B17").value
cycle_length = ws.range("C3:C17").value

average_period_length= statistics.mean(period_length)
average_cycle_length = statistics.mean(cycle_length)


print(average_period_length, average_cycle_length)

plt.plot(cycles, period_length, '-o', label='Period_length')
plt.plot(cycles, cycle_length,'-o',label = 'Cycle length')
plt.axhline(y = average_period_length, linestyle = 'dotted', color='b', label= 'Average_period_length')
plt.axhline(y = average_cycle_length, linestyle = 'dotted', color='orange', label= 'Average_cycle_length')
plt.axhspan(21, 35, alpha=0.3, color='orange', label='Normal menstrual cycle length')
plt.axhspan(2,7, alpha=0.3, color='b', label='Normal period length' )
plt.grid()
plt.xlabel('Cycles')
plt.ylabel('Days')
plt.legend()
plt.show()