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

