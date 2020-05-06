# %% [markdown]
# ### Bài tập 1.
#  - <ins>Yêu cầu</ins>: Trình bày sự khác biệt cơ bản của mô hình hồi quy tuyến tính và mô hình hồi quy logistic.
# - Khác biệt: 
#   * Kết quả (outcome) của `Linear Regression` là liên tục trong khi `Logistic Regression` 
#   lại là rời rạc  
#   * Để sử dụng `Linear Regression` ta cần phải có mối liên hệ tuyến tính giữa các biến phụ thuộc 
#   và biến không phụ thuộc. Nhưng khi sử dụng `Logistic Regression` ta không cần điều đó
#   * `Linear Regression` ta cố gắng fit đường thẳng vào dataset nhưng đối với `Linear Regression` 
#   ta cố gắng fit đường cong vào trong dataset 
#    * `Linear Regression` là thuật toán hồi quy (regression), `Logistic Regression` là thuật toán phân loại 
#    (classification)
#    * `Linear Regression` sử dụng phân gối Gaussian (Gaussian distribution) hoặc phân phối chuẩn 
#    (normal distribution). `Logistic Regression` sử dụng phân phối nhị thức (binomial distribution) 



# %% [markdown]
# ### Bài tập 2.
# 
# |    |   X | Y   |
# |---:|----:|:----|
# |  0 |  60 | Yes |
# |  1 |  55 | No  |
# |  2 |  61 | No  |
# |  3 |  70 | Yes |
# |  4 |  59 | Yes |
# |  5 |  65 | Yes |
# |  6 |  80 | Yes |
# |  7 |  63 | No  |
# |  8 |  50 | No  |
# |  9 |  75 | Yes |
# | 10 |  73 | Yes |
# | 11 |  51 | No  |
# 
# - <ins>Mô tả</ins>: Bài toán dự đoán sở thích học Machine Learning thông qua số đo cân nặng.
# - <ins>Dữ liệu</ins>: Tập tin ``ML_Learning_Hobby.csv``.
# - <ins>Yêu cầu</ins>:
#     - Xây dựng mô hình hồi quy logistic (sử dụng thư viện có sẵn hoặc ``build from scratch`` đều được) 
#     với thuộc tính tự do là ``X`` (cân nặng, là biến liên tục) và thuộc tính phụ thuộc ``Y`` 
#     (chỉ có hai giá trị ``Yes`` và ``No``, là biến rời rạc).
#     - Dự đoán rằng sinh viên có cân nặng ``X_new`` = ``62`` kg có thích học môn Machine Learning 
#     hay không?

# %%
# Import libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import classification_report,confusion_matrix
print('Import libraries successfully')

# %%
log_reg = LogisticRegression()

# %%
# Read data 
df = pd.read_csv('ML_Learning_Hobby.csv')
df.head()

# %% 
# Split data 
# This line below may cause error if re-run this sector multiple times
# In order to fix this, this line below 
df = df.replace({'Yes':1,'No':0}) # COMMENT IF ERROR 
X = np.array(df['X'].copy()).reshape(-1,1)
y = np.array(df['Y'].copy())

# %%
print('X columns: ',X[0:5])
print('y columns: ',y[0:5])

# %%
# Logistic Regression 
log_reg.fit(X,y)
y_predict = log_reg.predict(X)

# %% 
# Result 
x_test = np.array([62.0]).reshape(-1,1)
log_reg.predict(x_test)

# %% [markdown]
# Vậy đối với người có cân nặng 62kg sẽ có hứng thú vào môn học Machine Learning 

# %% [markdown]
# ### Bài tập 3.
# 
# |    |   Day | Outlook_Cat   |   Outlook_Cont | Temp_Cat   |   Temp_Cont | Humidity_Cat   |   Humidity_Cont | Wind_Cat   |   Wind_Cont | Play_Tennis   |
# |---:|------:|:--------------|---------------:|:-----------|------------:|:---------------|----------------:|:-----------|------------:|:--------------|
# |  0 |     1 | Sunny         |             15 | Hot        |          36 | High           |              71 | Weak       |         0.5 | No            |
# |  1 |     2 | Sunny         |             17 | Hot        |          35 | High           |              80 | Strong     |         4   | No            |
# |  2 |     3 | Overcast      |             57 | Hot        |          30 | High           |              75 | Weak       |         0.7 | Yes           |
# |  3 |     4 | Rain          |             90 | Mild       |          26 | High           |              73 | Weak       |         0.8 | Yes           |
# |  4 |     5 | Rain          |             92 | Cool       |          25 | Normal         |              61 | Weak       |         1   | Yes           |
# |  5 |     6 | Rain          |             91 | Cool       |          23 | Normal         |              50 | Strong     |         3   | No            |
# |  6 |     7 | Overcast      |             60 | Cool       |          25 | Normal         |              52 | Strong     |         4   | Yes           |
# |  7 |     8 | Sunny         |              8 | Mild       |          27 | High           |              65 | Weak       |         0.9 | No            |
# |  8 |     9 | Sunny         |             10 | Cool       |          20 | Normal         |              53 | Weak       |         1   | Yes           |
# |  9 |    10 | Rain          |             95 | Mild       |          27 | Normal         |              52 | Weak       |         1.1 | Yes           |
# | 10 |    11 | Sunny         |              7 | Mild       |          29 | Normal         |              57 | Strong     |         3.5 | Yes           |
# | 11 |    12 | Overcast      |             49 | Mild       |          28 | High           |              66 | Strong     |         7   | Yes           |
# | 12 |    13 | Overcast      |             62 | Hot        |          31 | Normal         |              57 | Weak       |         0.5 | Yes           |
# | 13 |    14 | Rain          |             94 | Mild       |          28 | High           |              66 | Strong     |         6   | No            |
# 
# 
# - <ins>Mô tả</ins>: Tìm mô hình hồi quy trên tập dữ liệu huấn luyện bao gồm các thuộc tính ``Outlook``, ``Temp``, ``Humidity``, ``Wind`` để dự đoán thuộc tính ``Play_Tennis``.
# - <ins>Dữ liệu</ins>: Tập tin ``Play_Tennis.csv``.
# - <ins>Lưu ý</ins>: Mỗi thuộc tính đã nêu ở phần mô tả sẽ có hai cột dữ liệu rời rạc và liên tục. Cụ thể là ``Outlook`` sẽ được biểu diễn bởi ``Outlook_Cat`` (Categorical Feature) và ``Outlook_Cont`` (Continuous Variable), và tương tự cho các thuộc tính còn lại. Thuộc tính ``Play_Tennis`` chỉ có hai giá trị rời rạc ``Yes`` và ``No``.
# - <ins>Yêu cầu</ins>:
#     - Xây dựng mô hình hồi quy logistic dựa vào nhóm các thuộc tích liên tục ``FeatureName_Cont``.
#     - Xây dựng mô hình hồi quy logistic dựa vào nhóm các thuộc tích liên tục ``FeatureName_Cont`` và rời rạc ``FeatureName_Cat`` (có thể sử dụng thư viện ``LabelEncoder`` để  chuyển thuộc tính rời rạc sang liên tục).
#     - Nhận xét kết quả đạt được từ hai mô hình.


# %% 
# Import and process data
# Import libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import classification_report,confusion_matrix
print('Import libraries successfully')


# %% 
log_reg = LogisticRegression()
labelEncoder = preprocessing.LabelEncoder()



# %% 
# Receive data 
data = pd.read_csv('Play_Tennis.csv')
data.head()

# %% 
# Get output (Play_Tennis feature)
y = data['Play_Tennis'].copy()
y.head()

# %% 
# Label encoding for output 
labelEncoder.fit(y)
y = labelEncoder.transform(y)
y


# %%  
# Seperate feature names 
FeatureName_Cont = ['Outlook_Cont','Temp_Cont','Humidity_Cont','Wind_Cont']
FeatureName_Cat = ['Outlook_Cat','Temp_Cat','Humidity_Cat','Wind_Cat']
print(FeatureName_Cont) 
print(FeatureName_Cat) 


# %% [markdown]
# Xây dựng mô hình hồi quy logistic dựa vào nhóm các thuộc tính 
# liên tục `FeatureName_Cont` 


# %% 
FeatureName_Cont = ['Outlook_Cont','Temp_Cont','Humidity_Cont','Wind_Cont']
X_Cont = np.array(data[FeatureName_Cont].copy())
X_Cont


# %% 
log_reg.fit(X_Cont, y)
y_predict_Cont = log_reg.predict(X_Cont)
y_predict_Cont


# %%
# Report compare y and y_predict_cont
print(classification_report(y,y_predict_Cont))

# %% [markdown]
# * Xây dựng mô hình hồi quy logistic dựa vào nhóm các thuộc tính liên tục `FeatureName_Cont` và 
# rời rạc `FeatureName_Cat` 

# %%
# Get all features and values from dataset
X_All = data.drop('Play_Tennis',axis=1).copy()
X_All


# %%
# Label encoding for labeled columns
# Demo label encoding for multiple columns
X_All = data.copy().drop("Play_Tennis",axis=1)
FeatureName_Cat = ['Outlook_Cat','Temp_Cat','Humidity_Cat','Wind_Cat']
# Label encoding for FeaturesName_Cat columns 
for feature in FeatureName_Cat: 
    labelEncoder.fit(X_All[feature])
    X_All[feature] = labelEncoder.transform(X_All[feature])
X_All = np.array(X_All) # transform to np array
X_All[0:5]

# %%
# Training
log_reg.fit(X_All,y)
y_predict_All = log_reg.predict(X_All) 
y_predict_All


# %% [markdown] 
# Report cuả mô hình sử dụng biến liên tục `FeatureName_Cont` 
print(classification_report(y_predict_Cont,y))

# %%
# Print out report
print(classification_report(y_predict_All,y))


# %% [markdown]
# Nhận xét
# * Từ bảng trên ta nhận được `accuracy` của mô hình 
# khi sử dụng các đại lượng liên tục `FeatureName_Cont` tương đối thấp $0.57$ 
# * Khi sử dụng hết tất cả các features có trong dataset 
# thì `accuracy` mà ta nhận được từ mô hình cho kết quả cao hơn đáng kể là $0.79$
# * Tuy nhiên `accuracy` này vẫn còn rất thấp 
