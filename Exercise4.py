# %% [markdown]
# # Bài tập thực hành tuần 2
# ## Bài tập 4
# * Phân tích hồi quy đa biến, cho dữ liệu về giá cổ phiếu demo-data-mul.xls.
# * Vẽ biểu đồ Scatter giữa giá cổ phiếu và các thuộc tính Interest Rate và Unemployment Rate
# * Tìm phương trình hồi quy của giá cổ phiếu theo hai thuộc tính (biến) Interest Rate và Unemployment Rate.

# %% [markdown]
# ### Gọi một số thư viện cần thiết
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
print('Import libraries successfully')


# %% 
data = pd.ExcelFile('demo_data_mul_2.xls')

# %% 
df = pd.read_excel(data, 0, header = 0)
df.head()

# %%
df.shape

# %%
unemployRate = df[['Unemployment_Rate']]
interestRate = df[['Interest_Rate']]
stockIdxPrice = df[['Stock_Index_Price']]

# %% [markdown]
# ### Biểu đồ Scatter liên hệ giữa giá cổ phiếu và Interest Rate
plt.scatter(stockIdxPrice, interestRate)
plt.xlabel('Stock Index Price') 
plt.ylabel('Interest_Rate')

# %% [markdown]
# ### Biểu đồ liên hệ giữa giá cổ phiếu và Unemployment_Rate
plt.scatter(stockIdxPrice, unemployRate)
plt.xlabel('Stock Index Price') 
plt.ylabel('Unemployment_Rate')


# %% [markdown]
# Lấy dữ liệu cho model
X = df[['Interest_Rate', 'Unemployment_Rate']]
X.head()

# %% 
y = df[['Stock_Index_Price']]
y.head()


# %% [markdown]
# ### Phương trình hồi quy tuyến tính
lin_reg = LinearRegression() 
lin_reg.fit(X,y) 
theta0, theta = lin_reg.intercept_, lin_reg.coef_ 
print(theta0)
print(theta) 

# %% [markdown]
# ### Vậy phương trình hồi quy tuyến tính tìm được là 
# # $y$ = $1798.4 + 345.54 {x}_1 - 250.147{x}_2$