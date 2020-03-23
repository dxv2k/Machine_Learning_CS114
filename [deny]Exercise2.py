# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Bài tập thực hành tuần 2
# ## Bài tập 2
#  * Sử dụng chương trình Python trang 114 - 117 sách Hands-on Machine Learning with Scikit-Learn, Keras & TensorFlow, 2nd Edition tính lại ví dụ mô phỏng ở slides 11 thuộc Tuần 2.
# %% [markdown]
#  Ở bài tập 1 ta nhận thấy rằng chỉ xét đối với ${\theta}_0$ và ${\theta}_1$
#  vì vậy ta sẽ trường hợp đối với hàm tồn tại nhiều ${\theta}$ hơn
#  ta sẽ sử dụng $y(x,$${\theta}$$)$ = ${\theta}_0$ + ${\theta}_1$$x_1$ + ${\theta}_2$$x_2$ + ${\theta}_3$$x_3$ + ${\mu}$.
#  Lý do chỉ sử dụng 3 tham số ${\theta}$ để ta có thể dể dàng biểu thị đồ thị.
#  Tương tự ta cũng sẽ sử dụng 2 cách: áp dụng thư viện và tự viết thuật toán.
# %% [markdown]
#  ### Import thư viện

# %%
import numpy as np
print('Import numpy as np successfully')

# %% [markdown]
#  ### Tạo dataset
#  #### Ta sẽ thử tạo dataset từ phương trình
#  #### $y(x,$${\theta}$$)$ = 4 + 3$x_1$ + 5$x_2$ + 2$x_3$ + ${\mu}$
#  #### ${\mu}$ = Gaussian noise sẽ được tạo ra từ hàm random trong thư viện numpy

# %%
x1 = 2 * np.random.rand(100,1)
x2 = 3 * np.random.rand(100,1)
x3 = 4 * np.random.rand(100,1)
y = 4 + 3*x1 + 5*x2 + 2*x3 + np.random.randn(100,1)

# %% [markdown]
#  ### Tổng quát về dataset vừa được tạo ra
#  #### x1,x2,x3 vừa được tạo ra

# %%
print('x1           x2              x3')
for i in range(5): print(x1[i],x2[i],x3[i])

# %% [markdown]
#  #### y vừa được tạo ra

# %%
for i in range(5): print(y[i])

# %% [markdown]
#  ### Đồ thị Scatter quan hệ giữa x1, x2, x3 và y

# %%
import matplotlib.pyplot as plt
print('Import pyplot as plt successfully') 

# %% [markdown]
#  #### Đồ thị Scatter giữa x1 và y

# %%
plt.scatter(x1,y)

# %% [markdown]
#  #### Đồ thị Scatter giữa x2 và y

# %%
plt.scatter(x2,y)

# %% [markdown]
# #### Đồ thị Scatter giữa x3 và y

# %%
plt.scatter(x3,y)


# %% [markdown]
# ### Tính toán $\hat{\theta}$ sử dụng Normal Equation

# %% [markdown]
# ### Tiến hành tính toán $\hat{\theta}$ 
# #### Ta sẽ lần lượt tính toán $\hat{\theta}_0$, $\hat{\theta}_1$, $\hat{\theta}_2$ và $\hat{\theta}_3$  
X = np.array([x1,x2,x3]) #easier to iterate
min_theta = [[0,0]] * 3 #for every [0,0] contains theta0 and theta[i]
                        # Further explanation below 
for i in range(3): #calculate theta 
    X_b = np.c_[np.ones((100,1)),X[i]]
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y) #return theta0 and theta[i]
    min_theta[i] = theta_best

# %% [markdown]
# ### Xem kết quả vừa tính toán được 
# #### Giải thích thêm về min_theta
# #### min_theta là mảng 2 chiều, mỗi phần tử trong mảng chứa kết quả theta0 và theta[i]
for i in range(3): 
    print(min_theta[i])

# %% [markdown] 
# ### Nhận xét: 
# #### * Phương trình ban đầu: $y(x,$${\theta}$$)$ = 4 + 3$x_1$ + 5$x_2$ + 2$x_3$ + ${\mu}$ 
# #### * Ta nhận thấy được rằng kết quả trả về có sai số khá lớn đặc biệt là theta0 và tương tự đối với các theta còn lại. 
# #### * Đồng thời ta còn nhận được 2 kết quả theta0 khác nhau vì vậy ta sẽ không tiến hành kiểm tra kết quả (Prediction) của model

# %% [markdown]
# ## Sử dụng Linear Regression trên thư viện Scikit-learn 
# ### Thêm thư viện
from sklearn.linear_model import LinearRegression
print('Import Linear Regression from SKlearn successfully')
lin_reg = LinearRegression()


# %% [markdown]
# ## 