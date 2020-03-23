# %% [markdown]
# # Bài tập thực hành tuần 2
# ## Bài tập 3
# * Thực hành với dữ liệu cho trong Sheet1 file Excel demo-data.xls.
# * Vẽ biểu đồ phân tán (biểu đồ Scatter) và nhận định về quan hệ giữa  X  và  Y .
# * Thay đổi một giá trị của Y sao cho thật khác biệt. Chạy chương trình và quan sát.

# %% [markdown]
# ### Gọi một số thư viện cần thiết
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
print('Import libraries successfully')

# %% 
data = pd.ExcelFile('demo_data.xls')

# %% 
df = pd.read_excel(data, 0, header = 0)
df.head()

# %%
df.shape

# %% [markdown]
# ### Biểu đồ Scatter và quan hệ giữa X với y
# #### Biểu đồ Scatter
X = df[['X']] #Get X column from dataframe
y = df[['y']] #Get y column from dataframe
plt.scatter(X,y)
plt.xlabel('x') 
plt.ylabel('y')

# %% [markdown]
# ### Nhận xét:
# #### - Ta thấy rằng biểu đồ có tăng dần, khi giá trị của x tăng thì giá trị của y cũng tăng theo

# %% [markdown]
# ### Sử dụng Linear Regression để tính toán $\hat{\theta}$
lin_reg = LinearRegression()
lin_reg.fit(X,y)
theta0, theta1 = lin_reg.intercept_, lin_reg.coef_ #Get theta0 and theta1
print('theta0: ',theta0)
print('theta1: ',theta1)


# %% [markdown]
# ### Thay đổi 1 giá trị y khác biệt, chạy lại chương trình để xem xét
randIdx = np.random.randint(1,y.shape[0])
y.at[randIdx, 'y'] = y.at[randIdx, 'y'] + 1000
y.at[randIdx, 'y']

# %% [markdown]
# ### Biểu đồ Scatter sau khi thay đổi 1 giá trị y
plt.scatter(X,y)
plt.xlabel('x') 
plt.ylabel('y')

# %% [markdown]
# ### Sử dụng Linear Regression để tính toán $\hat{\theta}$
lin_reg.fit(X,y)
theta0_new, theta1_new = lin_reg.intercept_, lin_reg.coef_ #Get theta0 and theta1
print('theta0: ',theta0_new)
print('theta1: ',theta1_new)

# %% [markdown]
# ### Nhận xét: 
# #### Vậy ta thấy được rằng nếu có một giá trị lớn bất thường xuất hiện trong dataset sẽ làm thay đổi kết quả trả về từ phương trình rất nhiều nên đây là điều cần đặc biệt quan tâm