# %% [markdown]
#  # Bài tập thực hành tuần 2
#  ## Bài tập 1
#  * Thực hành trên Python tính phương trình hồi quy đơn (trang 114 - 117 sách Hands-on Machine Learning with Scikit-Learn, Keras & TensorFlow, 2nd Edition).
# %% [markdown]
#  # 1. Tạo data-set có dạng tuyến tính
#  Import thư viện

# %%
import numpy as np
import matplotlib.pyplot as plt
print('Import numpy successfully')

# %% [markdown]
#  Tạo data-set

# %%
X = 2 * np.random.rand(100,1)
y = 4 + 3 * X + np.random.randn(100,1) 

# %% [markdown]
#  Xem sơ lượt về dataset đã được tạo ra
#  5 phần tử đầu tiên của X

# %%
for i in range(5):  
    print(X[i])

# %% [markdown]
#  5 phần tử đầu tiên cua của y

# %%
for i in range(5): 
    print(y[i])

# %% [markdown]
#  Vẽ đồ thị thể hiện sự tương quan giữa X và y

# %%
plt.scatter(X,y)

# %% [markdown]
#  # 2.Bắt đầu từ đây, ta sẽ tiến hành tính toán $\hat{\theta}$ sử dụng Normal Equation
# %% [markdown]
#  Tiến hành bổ sung x0 = 1 vào ma trận X

# %%
X_b = np.c_[np.ones((100,1)),X]
for i in range(5): 
    print(X_b[i])

# %% [markdown]
#  Tiến hành tính toán theo công thức
#  $\hat{\theta}$ = $(X^T X)^{-1} X^T y $

# %%
min_theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print(min_theta)

# %% [markdown]
#  Tiến hành tính toán theo công thức
#  $\hat{\theta}$ = $(X^T X)^{-1} X^T y $

# %%
min_theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print(min_theta)

# %% [markdown]
#  Từ đây ta có thể làm tròn ${\theta}_0$, ${\theta}_1$ và
#  thầy rằng ${\theta}_0$ = 4 và ${\theta}_1$ = 3 đã dự đoán
#  gần đúng phương trình mà ta đã sử dụng ban đầu
# %% [markdown]
#  Ta sẽ tiến hành sử dụng $\hat{\theta}$ để kiểm nghiệm
#  tính đúng đắn với bộ X = {0,2}

# %%
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new] # add x0 = 1 column  
y_predict = X_new_b.dot(min_theta) 
y_predict 

# %% [markdown]
#  Vẽ đồ thị thể hiện dự đoán của phương trình hồi quy
#  tuyến tính

# %%
plt.plot(X_new, y_predict, "-r")
plt.plot(X,y, "b.")
plt.axis([0,2,0,15])
plt.show() 

# %% [markdown]
# # 3.Sử dụng thư viện Scikit-learn
# %% [markdown]
# Ta sẽ sử dụng thư viên Scikit-learn để tính toán và so sánh kết quả giữa việc sử dụng thư viện và code 

# %%
from sklearn.linear_model import LinearRegression 
print('Import Linear Regression successfully')

# %% [markdown]
# Tiến hành tính toán ${\theta}_0$ và ${\theta}_1$ trong trường hợp ở đây sẽ là kết quả trả về lần lượt của 2 hàm coef_ và intercept_

# %%
lin_reg = LinearRegression() 
lin_reg.fit(X,y) #Perform fitting dimension of the X,y matrices
lin_reg.intercept_, lin_reg.coef_ #coeff_ estimated coefficient for theta, in this case theta1 
                                  #intercept_ estimated the intercept, in this case theta0

# %% [markdown]
# Sau khi tính toán được ${\theta}_0$ và ${\theta}_1$, tiến hành tính toán kết quả sẽ nhận được từ dự đoán bằng bộ thử X_new = {0,2}

# %%
y_new = lin_reg.predict(X_new) 

# %% [markdown]
# Vẽ đồ thị thể hiện prediction của model 
# 

# %%
plt.plot(X_new, y_new, "r-")
plt.plot(X,y,"b.")

# %% [markdown]
# Ta có thể thấy được rằng kết quả giữa việc xây dựng lại toàn bộ và việc áp dụng thư viện cho kết quả không chênh lệch nhau nhiều. Tuy nhiên việc áp dụng thư viện lại khiến xây dựng Linear Regression model nhanh và thuận tiện hơn nhiều 
