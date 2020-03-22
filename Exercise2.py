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
X1 = 2 * np.random.rand(100,1)
X2 = 3 * np.random.rand(100,1)
X3 = 4 * np.random.rand(100,1)
y = 4 + 3*X1 + 5*X2 + 2*X3 + np.random.randn(100,1)

# %% [markdown]
#  ### Tổng quát về dataset vừa được tạo ra
#  #### X1,X2,X3 vừa được tạo ra

# %%
print('X1           X2              X3')
for i in range(5): print(X1[i],X2[i],X3[i])

# %% [markdown]
#  #### y vừa được tạo ra

# %%
for i in range(5): print(y[i])

# %% [markdown]
#  ### Đồ thị Scatter quan hệ giữa X1, X2, X3 và y

# %%
import matplotlib.pyplot as plt
print('Import pyplot as plt successfully') 

# %% [markdown]
#  #### Đồ thị Scatter giữa X1 và y

# %%
plt.scatter(X1,y)

# %% [markdown]
#  #### Đồ thị Scatter giữa X2 và y

# %%
plt.scatter(X2,y)

# %% [markdown]
# #### Đồ thị Scatter giữa X3 và y

# %%
plt.scatter(X3,y)


# %% [markdown]
# ### Tính toán $\hat{\theta}$ sử dụng Normal Equation


# %% [markdown]
# #### Gom nhóm X1,X2,X3 thành một ma trận và thêm vào 1 trước mỗi dòng 
X = np.array([X1,X2,X3]) #Transfer 3 matrices into X
X_b = np.insert(X,0,1.0,axis = 1) #Add one to each instance

# %% [markdown]
# #### Tiến hành tính toán $\hat{\theta }$ 
theta_min = 