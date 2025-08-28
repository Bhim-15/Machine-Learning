import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import math

#predicting movie genre
df = pd.read_excel(r"C:\Users\DELL-PC\Desktop\python\ml\classification_data.xlsx")
print(df.head())

import math

rating = np.array([8.0, 6.2, 7.2, 8.2])
duration = np.array([160, 170, 168, 155])

#data of barbie movie 
x1 = 7.4
y1 = 114


movie1 = math.sqrt((x1 - rating[0])**2 + (y1 - duration[0])**2)
print("Distance from first movie:", movie1)

movie2 = math.sqrt((x1 - rating[1])**2 + (y1 - duration[1])**2)
print("Distance from second movie:", movie2)

movie3 = math.sqrt((x1 - rating[2])**2 + (y1 - duration[2])**2)
print("Distance from third movie:", movie3)

movie4 = math.sqrt((x1 - rating[3])**2 + (y1 - duration[3])**2)
print("Distance from fourth movie:", movie4)

list1=[movie1,movie2,movie3,movie4]
nearest_index = np.argmin(list1)

print("Nearest distance:", list1[nearest_index])
print("Nearest neighbor:", (rating[nearest_index], duration[nearest_index]))

val=plt.scatter(rating,duration)
plt.show()

val1=plt.plot(list1)
plt.show()
