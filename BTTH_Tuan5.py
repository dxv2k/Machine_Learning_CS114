# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ### Bài tập 1.
#  - <ins>Yêu cầu</ins>: Trình bày sự khác biệt cơ bản của mô hình hồi quy tuyến tính và mô hình hồi quy logistic.

# %% 
# 

# %%
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
#     - Xây dựng mô hình hồi quy logistic (sử dụng thư viện có sẵn hoặc ``build from scratch`` đều được) với thuộc tính tự do là ``X`` (cân nặng, là biến liên tục) và thuộc tính phụ thuộc ``Y`` (chỉ có hai giá trị ``Yes`` và ``No``, là biến rời rạc).
#     - Dự đoán rằng sinh viên có cân nặng ``X_new`` = ``62`` kg có thích học môn Machine Learning hay không?
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
