# %% [markdown]
# # Bài tập thực hành tuần 2
# ## Bài tập 5
# #### Thực hành với dữ liệu cho trong Sheet2 file Excel demo-data.xls.
# #### Thực hiện các yêu cầu tương tự Bài tập 4.
# #### Bỏ bớt biến và quan sát sự thay đổi các thông số trong bảng phân tích hồi quy.
# #### Xác định phương trình hồi quy có chứa  $X^{2}_2$  và nhận xét các thông số (Tạo thêm 1 cột $X^{2}_2$).

# %% [markdown]
# #### Thêm một số thư viện cần thiết vào chương trình
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
print('Import libraries successfully')

# %% [markdown]
# #### Đọc file và chuyển về dataframe
data = pd.ExcelFile('demo_data.xls')
df = pd.read_excel(data, 1, header = 0)
df.head() 

# %% [markdown]
# #### Tách dữ liệu cho phương trình
X = df[['X1', 'X2']]
y = df[['y']]

# %% 
X.head()

# %% 
y.head()

# %% [markdown]
# ### Đồ thị biểu hiện sự tương quan giữa các thành phần
plt.scatter(df[['X1']],df[['y']])
plt.xlabel('X1')
plt.ylabel('y')

# %% 
plt.scatter(df[['X2']],df[['y']])
plt.xlabel('X2')
plt.ylabel('y')

# %% [markdown]
# ### Correlation giữa các thành phần
sns.pairplot(data = df, kind = 'scatter')
plt.show()


# %% [markdown]
# ### Linear Regression 

# %%
lin_reg = LinearRegression()
lin_reg.fit(X,y)

# %% [markdown]
# #### Kết quả $\hat{\theta}$ 
lin_reg.intercept_, lin_reg.coef_

# %% [markdown]
# ### Bảng phân tích hồi quy 
import statsmodels.formula.api as smf 
result = smf.ols('y ~ X', data = df).fit() 
print(result.summary())

# %% [markdown]
# #### Ta sẽ bỏ bớt X2 để xem sự thay đổi của $\hat{\theta}$  và bảng hồi quy

# %% 
lin_reg.fit(df[['X1']],y)
lin_reg.intercept_, lin_reg.coef_

# %% 
resultDropX2 = smf.ols('y ~ X1', data = df[['X1','y']]).fit() 
print(resultDropX2.summary())

# %% [markdown]
# ### Thêm vào cột $X^{2}_2$, xác định phương trình hồi quy và nhận xét các thông số 

# %% 
x22_vector = np.full((10,1),7.0) #x22 = 7.0
copy_df = df.copy() #Copy exists df to new_df
copy_df['x22'] = x22_vector #add x22 column
copy_df.head()

# %% [markdown]
# #### Biểu đồ thể hiện sự tương quan, thay đổi khi thêm cột $X^{2}_2$ 
sns.pairplot(data = copy_df, kind = 'scatter')
plt.show()

# %% [markdown]
# #### Tính toán phương trình hồi quy
lin_reg.fit(copy_df,y)
lin_reg.intercept_, lin_reg.coef_

# %% [markdown]
# ### Nhận xét: 
# #### * Khi thêm cột X22 vào, phương trình hồi quy xuất hiện thêm 2 theta tuy nhiên lại nhận được giá trị 1 và 0
# #### * Giá trị của ${\theta}_0$ thay đổi rất nhiều, chênh lệch lớn so với ban đầu