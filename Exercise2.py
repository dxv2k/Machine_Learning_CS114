# %% [markdown]
# # Bài tập thực hành tuần 2
# ## Bài tập 2
#  * Sử dụng chương trình Python trang 114 - 117 sách Hands-on Machine Learning with Scikit-Learn, Keras & TensorFlow, 2nd Edition tính lại ví dụ mô phỏng ở slides 11 thuộc Tuần 2.

# %% [markdown]
# ### Dữ liệu từ đề bài 
# ### $(x_i,y_i) = (147,49), (150,53), (153,51), (160,54)$

# %% [markdown]
# ### C1: Không áp dụng thư viện Scikit-learn

# %% [markdown]
# #### Gọi thư viện numpy
import numpy as np
print('Import numpy as np successfully')

# %% [markdown]
# #### Khởi tạo X và y
X = np.array([147,150,153,160])
y = np.array([49,53,51,54])
print(X,y)

# %% [markdown]
# #### Đồ thị thể hiện giữa X và y 
import matplotlib.pyplot as plt
plt.scatter(X,y)

# %% [markdown]
# #### Thêm cột 1 vào X và tính toán $\hat{\theta}$ 
X_b = np.c_[np.ones((4,1)),X]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y) 
print(theta_best) #Return theta0 and theta1

# %% [markdown]
# #### Vậy phương trình hồi quy ta tìm được là 
# #### $y = 5.02x + 0.31$

# %% [markdown]
# ### C2: Áp dụng thư viện Scikit-learn 
from sklearn.linear_model import LinearRegression 
print('Import Linear Regression from Sklearn successfully')

# %% 
lin_reg = LinearRegression() 
lin_reg.fit(X.reshape(-1,1),y)
lin_reg.intercept_, lin_reg.coef_ #intecept_ theta0, coef_ theta1

# %% [markdown]
# #### Vậy phương trình hồi quy ta tìm được sau khi áp dụng thư viện: 
# #### $y = 5.02x + 0.31$


# %% [markdown]
# #### Vậy cả 2 cách đều cho kết quả như nhau
