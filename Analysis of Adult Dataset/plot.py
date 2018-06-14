# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.misc import factorial
from scipy.stats import poisson

# 年龄
def show_age(df):
    print('最大值：', df['年龄'].max())
    print('最小值：', df['年龄'].min())
    print('均值：', df['年龄'].mean())
    print('中位数：', df['年龄'].median())
    print('标准差：', df['年龄'].std())
    plt.figure(figsize = (8, 5))
    ax = df['年龄'].value_counts(normalize = True).sort_index().plot(kind = 'line', xlim = (0, 100), secondary_y = True)
    ax.set(ylabel = "Proportion")
    ax = df['年龄'].value_counts().sort_index().plot(kind = 'line', xlim = (0, 100))
    ax.set(xlabel = "Age", ylabel = "Occurance")
    plt.savefig('figures/age.png', dpi = 300)

# 工作类别
def show_workclass(df):
    print(df['工作类别'].value_counts())
    print(df['工作类别'].value_counts(normalize = True))
    plt.figure(figsize = (8, 5))
    ax = df['工作类别'].value_counts(normalize = True).plot(kind = 'pie')
    ax.set(ylabel = 'Workclass')
    plt.savefig('figures/workclass.png', dpi = 300)

# 最终权重
def show_weight(df):
    print('最大值：', df['最终权重'].max())
    print('最小值：', df['最终权重'].min())
    print('均值：', df['最终权重'].mean())
    print('中位数：', df['最终权重'].median())
    print('标准差：', df['最终权重'].std())
    plt.figure(figsize = (8, 5))
    x = pd.Series([0] * 999999, index = range(1, 1000000))
    ax = x.add(df['最终权重'].value_counts(), fill_value=0).plot(kind = 'area', xlim = (0, 1000000), ylim = (0, 10))
    ax.set(xlabel = "Final weight", ylabel = "Occurance")
    plt.savefig('figures/weight.png', dpi = 300)
    # normalized_data = df['最终权重'].value_counts(normalize = True).sort_index().to_frame()
    # print(list(normalized_data))
    # print(pd.Series(range(1, 1000000)).to_frame().apply(poisson.pmf, args = (df['最终权重'].mean(),)))
    # pd.Series(range(1, 1000000)).to_frame().apply(poisson.pmf, args = (df['最终权重'].mean(),)).plot()
    # plt.show()

# 教育程度
def show_education(df):
    plt.figure(figsize = (8, 5))
    print(df['教育程度'].value_counts(normalize = True))
    ax = df['教育程度'].value_counts(normalize = True).plot(kind = 'pie')
    ax.set(ylabel = 'Education')
    plt.savefig('figures/education.png', dpi = 300)

# 受教育时间
def show_education_time(df):
    print('最大值：', df['受教育时间'].max())
    print('最小值：', df['受教育时间'].min())
    print('均值：', df['受教育时间'].mean())
    print('中位数：', df['受教育时间'].median())
    print('标准差：', df['受教育时间'].std())
    plt.figure(figsize = (8, 5))
    ax = df['受教育时间'].value_counts(normalize = True).sort_index().plot(kind = 'line', xlim = (0, 100), secondary_y = True)
    ax.set(ylabel = "Proportion")
    ax = df['受教育时间'].value_counts().sort_index().plot(kind = 'line', xlim = (0, 20))
    ax.set(xlabel = "Education time / years", ylabel = "Occurance")
    plt.savefig('figures/education_time.png', dpi = 300)

# 婚姻状况
def show_marriage(df):
    plt.figure(figsize = (8, 5))
    ax = df['婚姻状况'].value_counts(normalize = True).sort_index().plot(kind = 'pie')
    ax.set(ylabel = 'Marital Status')
    plt.savefig('figures/marriage.png', dpi = 300)

# 职业
def show_specialization(df):
    plt.figure(figsize = (9, 5))
    ax = df['职业'].value_counts(normalize = True).sort_index().plot(kind = 'pie')
    ax.set(ylabel = '')
    plt.savefig('figures/specialization.png', dpi = 300)

# 家庭关系
def show_home_relationship(df):
    plt.figure(figsize = (8, 5))
    ax = df['家庭关系'].value_counts(normalize = True).sort_index().plot(kind = 'pie')
    ax.set(ylabel = '')
    plt.savefig('figures/home_relationship.png', dpi = 300)
    
# 种族
def show_race(df):
    print(df['种族'].value_counts(normalize = True))
    plt.figure(figsize = (10, 5))
    ax = df['种族'].value_counts(normalize = True).sort_index().plot(kind = 'pie')
    ax.set(ylabel = '')
    plt.savefig('figures/race.png', dpi = 300)

# 性别
def show_gender(df):
    print(df['性别'].value_counts(normalize = True))
    plt.figure(figsize = (8, 6))
    ax = df['性别'].value_counts(normalize = True).sort_index().plot(kind = 'bar', secondary_y = True)
    ax.set(ylabel = "Proportion")
    ax = df['性别'].value_counts().sort_index().plot(kind = 'bar')
    ax.set(xlabel = "Gender", ylabel = "Occurance")
    plt.savefig('figures/gender.png', dpi = 300)

# 资本收益
def show_profit(df):
    print('有效样本数：', df[df['资本收益'] > 0].shape[0])
    print('最大值：', df[df['资本收益'] > 0]['资本收益'].max())
    print('最小值：', df[df['资本收益'] > 0]['资本收益'].min())
    print('均值：', df[df['资本收益'] > 0]['资本收益'].mean())
    print('中位数：', df[df['资本收益'] > 0]['资本收益'].median())
    print('标准差：', df[df['资本收益'] > 0]['资本收益'].std())
    plt.figure(figsize = (8, 5))
    x = pd.Series([0] * 119999, index = range(1, 120000))
    ax = x.add(df[df['资本收益'] > 0]['资本收益'].value_counts(), fill_value=0).plot(kind = 'area', xlim = (0, 120000))
    ax.set(xlabel = "Capital gains", ylabel = "Occurance")
    plt.savefig('figures/profit.png', dpi = 300)

# 资本亏损
def show_loss(df):
    print('有效样本数：', df[df['资本亏损'] > 0].shape[0])
    print('最大值：', df[df['资本亏损'] > 0]['资本亏损'].max())
    print('最小值：', df[df['资本亏损'] > 0]['资本亏损'].min())
    print('均值：', df[df['资本亏损'] > 0]['资本亏损'].mean())
    print('中位数：', df[df['资本亏损'] > 0]['资本亏损'].median())
    print('标准差：', df[df['资本亏损'] > 0]['资本亏损'].std())
    plt.figure(figsize = (8, 5))
    x = pd.Series([0] * 3999, index = range(1, 4000))
    ax = x.add(df[df['资本亏损'] > 0]['资本亏损'].value_counts(), fill_value=0).plot(kind = 'area', xlim = (0, 4000))
    ax.set(xlabel = "Capital loss", ylabel = "Occurance")
    plt.savefig('figures/loss.png', dpi = 300)
    
# 每周工作小时数
def show_working_hours(df):
    print('最大值：', df['每周工作小时数'].max())
    print('最小值：', df['每周工作小时数'].min())
    print('均值：', df['每周工作小时数'].mean())
    print('中位数：', df['每周工作小时数'].median())
    print('标准差：', df['每周工作小时数'].std())
    plt.figure(figsize = (8, 5))
    x = pd.Series([0] * 99, index = range(1, 100))
    ax = x.add(df[df['每周工作小时数'] > 0]['每周工作小时数'].value_counts(), fill_value=0).plot(kind = 'area', xlim = (0, 100))
    ax.set(xlabel = "Working hours per week", ylabel = "Occurance")
    plt.savefig('figures/working_hours.png', dpi = 300)

# 祖国
def show_nationality(df):
    plt.figure(figsize = (8, 7))
    ax = df['祖国'].value_counts(normalize = True).sort_index().plot(kind = 'bar', ylim = (0, 1))
    ax.set(ylabel = '')
    plt.subplots_adjust(bottom = 0.4)
    plt.savefig('figures/nationality.png', dpi = 300)

# 收入
def show_income(df):
    plt.figure(figsize = (8, 6))
    print(df['收入'].value_counts())
    ax = df['收入'].value_counts(normalize = True).sort_index().plot(kind = 'bar', secondary_y = True)
    ax.set(ylabel = "Proportion")
    ax = df['收入'].value_counts().sort_index().plot(kind = 'bar')
    ax.set(xlabel = "Income", ylabel = "Occurance")
    plt.savefig('figures/income.png', dpi = 300)

# 工作类别-堆积图
def show_workclass_stacked(df):
    plt.figure(figsize = (8, 5))
    ax = pd.crosstab(df['工作类别'], df['收入']).plot(kind = 'bar', stacked = True)
    ax.set(xlabel = 'Workclass')
    ax.get_legend().set_title('Income')
    plt.tight_layout()
    plt.savefig('figures/workclass_stacked.png', dpi = 300)

# 受教育时间-堆积图
def show_education_time_stacked(df):
    plt.figure(figsize = (8, 5))
    ax = pd.crosstab(df['受教育时间'], df['收入']).plot(kind = 'area', stacked = True)
    ax.set(xlabel = 'Education time')
    ax.get_legend().set_title('Income')
    plt.tight_layout()
    plt.savefig('figures/education_time_stacked.png', dpi = 300)

# 婚姻状况-堆积图
def show_marriage_stacked(df):
    plt.figure(figsize = (8, 5))
    ax = pd.crosstab(df['婚姻状况'], df['收入']).plot(kind = 'bar', stacked = True)
    ax.set(xlabel = 'Marital Status')
    ax.get_legend().set_title('Income')
    plt.tight_layout()
    plt.savefig('figures/marriage_stacked.png', dpi = 300)

# 性别-堆积图
def show_gender_stacked(df):
    plt.figure(figsize = (8, 5))
    ax = pd.crosstab(df['性别'], df['收入']).plot(kind = 'bar', stacked = True)
    ax.set(xlabel = 'Gender')
    ax.get_legend().set_title('Income')
    plt.tight_layout()
    plt.savefig('figures/gender_stacked.png', dpi = 300)

# 每周工作小时数-堆积图
def show_working_hours_stacked(df):
    plt.figure(figsize = (8, 5))
    ax = pd.crosstab(df['每周工作小时数'], df['收入']).plot(kind = 'area', stacked = True)
    ax.set(xlabel = 'Working hours')
    ax.get_legend().set_title('Income')
    plt.tight_layout()
    plt.savefig('figures/working_hours_stacked.png', dpi = 300)

# 读取数据文件
headers = ['年龄', '工作类别', '最终权重', '教育程度', '受教育时间', '婚姻状况', '职业', '家庭关系', '种族', '性别', '资本收益', '资本亏损', '每周工作小时数', '祖国', '收入']
df = pd.read_csv('adult.txt', names = headers)

show_working_hours_stacked(df)