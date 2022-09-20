import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools as it
from sklearn import linear_model
from copy import deepcopy
%matplotlib inline

# Ground Criket Chirps
ground_cricket_data = {"Chirps/Second": [20.0, 16.0, 19.8, 18.4, 17.1, 15.5, 14.7,
                                         15.7, 15.4, 16.3, 15.0, 17.2, 16.0, 17.0,
                                         14.4],
                       "Ground Temperature": [88.6, 71.6, 93.3, 84.3, 80.6, 75.2, 69.7,
                                              71.6, 69.4, 83.3, 79.6, 82.6, 80.6, 83.5,
                                              76.3]}
df = pd.DataFrame(ground_cricket_data)
print(df)

ground_temp = df['Ground Temperature']
ground_temp = ground_temp.to_frame()
chirps_sec = df['Chirps/Second']

lin_reg = linear_model.LinearRegression()
lin_reg.fit(ground_temp,chirps_sec)

Intercept = lin_reg.intercept_
Coef = lin_reg.coef_
print(Intercept)
print(Coef)

plt.scatter(ground_temp,chirps_sec, color='blue')
plt.plot(ground_temp, lin_reg.predict(ground_temp), color='green')
plt.title('Ground Cricket Chirps/Sec by Ground Temperature')
plt.xlabel('Ground Temp (˚F)')
plt.ylabel('Chirps/Sec')
plt.show()

print("R2 value : ", lin_reg.score(ground_temp,chirps_sec))

lin_reg.predict([[95]])


#Brain vs Body Weight

df_bb = pd.read_fwf("F:\\GUVI\\Task 6\\brain_body.txt")

regr_bb = linear_model.LinearRegression()
body = df_bb[['Body']]
brain = df_bb['Brain']
regr_bb.fit(body, brain)

print('Linear Regression Equation: y = {:.4f} * x + {:.4f}'.format(regr_bb.coef_[0], regr_bb.intercept_))

plt.scatter(body, brain, color='m')
plt.plot(body, regr_bb.predict(body))
plt.title('Brain Weight by Body Weight')
plt.xlabel('Body Weight')
plt.ylabel('Brain Weight')
plt.show()

print('R^2 score for this equation: {:.4f}'.format(regr_bb.score(body, brain)))


