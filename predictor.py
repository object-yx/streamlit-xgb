#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


# In[4]:


model = joblib.load("xgb.pkl")


# In[5]:


x_test = pd.read_csv("x_val_scaler.csv")


# In[6]:


feature_names = ["年龄","BMI","喝酒","抽烟","胎次","阴道分娩","剖宫产","流产",
                 "高血压","糖尿病","子宫肌瘤","陈旧性会阴裂伤","压力性尿失禁","绝育手术","阴道松弛","冠心病","体重(kg)"]


# In[7]:


import warnings


# In[8]:


st.title("盆腔器官脱垂预测器")


# In[9]:


age = st.number_input("年龄:",min_value=0,max_value=100,value=55)
BMI = st.number_input("BMI:",min_value=0,max_value=100,value=20)
parity = st.number_input("胎次:",min_value=0,max_value=100,value=2)
vd = st.number_input("阴道分娩:",min_value=0,max_value=100,value=1)
cs = st.number_input("剖宫产:",min_value=0,max_value=100,value=1)
mi = st.number_input("流产:",min_value=0,max_value=100,value=55)
weight = st.number_input("体重:",min_value=0,max_value=200,value=60)


# In[10]:


drink = st.selectbox("喝酒:",options=[0,1],format_func=lambda x:"是" if x==1 else "否")
smoke = st.selectbox("抽烟:",options=[0,1],format_func=lambda x:"是" if x==1 else "否")
htn = st.selectbox("高血压:",options=[0,1],format_func=lambda x:"是" if x==1 else "否")
dm = st.selectbox("糖尿病:",options=[0,1],format_func=lambda x:"是" if x==1 else "否")
zgjl = st.selectbox("子宫肌瘤:",options=[0,1],format_func=lambda x:"是" if x==1 else "否")
opl = st.selectbox("陈旧性会阴裂伤:",options=[0,1],format_func=lambda x:"是" if x==1 else "否")
sui = st.selectbox("压力性尿失禁:",options=[0,1],format_func=lambda x:"是" if x==1 else "否")
St = st.selectbox("绝育手术:",options=[0,1],format_func=lambda x:"是" if x==1 else "否")
vl = st.selectbox("阴道松弛:",options=[0,1],format_func=lambda x:"是" if x==1 else "否")
gxb = st.selectbox("冠心病:",options=[0,1],format_func=lambda x:"是" if x==1 else "否")


# In[11]:


feature_values = [age, BMI, parity, vd, cs, mi, weight, drink, smoke, htn, dm, zgjl, opl, sui, St, vl, gxb]


# In[12]:


features = np.array([feature_values])


# In[19]:
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.family"] = "Microsoft YaHei"

if st.button("预测"):
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    st.write(f"**预测结果为:**{predicted_class}(1:脱垂,0:正常)")
    st.write(f"**预测概率:**{predicted_proba}")
    probability = predicted_proba[predicted_class]*100
    if predicted_class == 1:
        advice = (f"高风险"
                  f"盆腔器官脱垂的概率为{probability:1f}%")
    else:
        advice = (f"低风险"
                  f"盆腔器官脱垂的概率为{probability:1f}%")
    st.write(advice)
    st.subheader("SHAP 解释")
    explainer_shap = shap.TreeExplainer(model)
    shap_values = explainer_shap.shap_values(pd.DataFrame([feature_values], columns = feature_names))
    shap.force_plot(explainer_shap.expected_value, shap_values[0],pd.DataFrame([feature_names], columns = feature_names),matplotlib=True)
    plt.savefig("shap_force_plot.png",bbox_inches="tight",dpi=1200)
    st.image("shap_force_plot.png",caption="SHAP Force Plot Explanation")


# In[ ]:




