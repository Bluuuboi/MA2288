import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

df = pd.read_csv('../Data_Sets/SOCR-HeightWeight.csv')

#Separating Data
height_data =  df['Height(Inches)']
weight_data = df['Weight(Pounds)']
des_w = weight_data.describe()
des_h = height_data.describe()

fig, subs = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))

#--WEIGHT DISTRIBUTION--
#Weight stats
mean_w = des_w.loc['mean']
std_w = des_w.loc['std']
min_w = des_w.loc['min']
print(min_w)
max_w = des_w.loc['max']
print(max_w)

#Gaussian Distribution curve(Weight Distribution) with the info below, 100 values
mu=mean_w
sigma=std_w
minn=min_w
maxx=max_w
x=np.linspace(minn,maxx,100)
y=np.exp(-0.5*(x-mu)**2/sigma**2)/(sigma*np.sqrt(2*np.pi))
subs[0].plot(x,y,'k',linewidth=1,label='Normal Curve')

#Plotting Gaussian Distribution(Weight Distribution) curve
w_plot = sns.kdeplot(data=df, x='Weight(Pounds)', fill=True,ax=subs[0])
w_plot.set_title('Weight Distribution')
w_plot.set_xlabel(xlabel='Weight(Pounds)', fontsize=10)
w_plot.set_ylabel(ylabel='Density', fontsize=10)
w_plot.legend()

#--HEIGHT DISTRIBUTION--
#Height stats
mean_h = des_h.loc['mean']
std_h = des_h.loc['std']
min_h = des_h.loc['min']
print(min_h)
max_h = des_h.loc['max']
print(max_h)

#Gaussian Distribution curve(Height Distribution) with the info below, 100 values
muh=mean_h
sigmah=std_h
minnh=min_h
maxxh=max_h
xh=np.linspace(minnh,maxxh,100)
yh=np.exp(-0.5*(xh-muh)**2/sigmah**2)/(sigmah*np.sqrt(2*np.pi))
subs[1].plot(xh,yh,'k',linewidth=1,label='Normal Curve')

#Plotting Gaussian Distribution(Height Distribution) curve
h_plot = sns.kdeplot(data=df, x='Height(Inches)', fill=True,ax=subs[1])
h_plot.set_title('Height Distribution')
h_plot.set_xlabel(xlabel='Height(Inches)', fontsize=10)
h_plot.set_ylabel(ylabel='Density', fontsize=10)
h_plot.legend()


plt.show()