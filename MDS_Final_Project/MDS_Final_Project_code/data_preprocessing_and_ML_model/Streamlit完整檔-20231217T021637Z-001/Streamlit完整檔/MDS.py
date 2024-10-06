# app.py
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
#from sklearn.ensemble import RandomForestClassifier
#from imblearn.over_sampling import SMOTE
#from sklearn.model_selection import StratifiedKFold, GridSearchCV
#from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from prediction import predict, predict_lr, predict_xgb

# Load the data
df = pd.read_csv('model_training.csv')
df_mapping = pd.read_excel('氣候資料.xlsx')
df_ols =  pd.read_csv('ols_top10_features.csv')
df_elastic =  pd.read_csv('elastic_top10_features.csv')
df_rf =  pd.read_csv('rf_top10_features.csv')
df_bridge_data = pd.read_excel('橋梁基本資料(北分局).xlsx')
df_bridge_variables = pd.read_excel('bridge_variable.xlsx')

def run():
    # Streamlit app
    col1, col2 = st.columns([0.15,0.85])
    with col2:
        st.title('橋梁劣化速度分析與預測')
    with col1:    
        st.image('photo_mds1.png', width=110)
    white_space = 16
    list_tabs = ['  各橋梁資料  ', '  資料視覺化  ', '  重要因子分析  ', '   模型預測   ']
    tab1, tab2,tab3,tab4 = st.tabs([s.center(white_space,"\u2001") for s in list_tabs])

    with tab1:
        st.header('各橋梁資料')
        df_show = pd.DataFrame()
        id_to_match = st.number_input("請輸入橋梁ID:", min_value=1, max_value=max(df_bridge_data['系統ID']))
        search_button = st.button("搜尋")
        if search_button:
            if id_to_match in df_bridge_data['系統ID'].values:
                matched_df = df_bridge_data[df_bridge_data['系統ID'] == id_to_match]
                df_show = pd.concat([df_show, matched_df], ignore_index = True)
                st.dataframe(df_show)
            else:
                st.text("查無橋梁，請再輸入一次")


    with tab2:
        st.header('重要變數視覺化')
        variable_selection = ['平均月相對濕度', '年平均每日交通量', '是否為跨水橋', '橋梁總長(M)', '橋下有無租用', '離海岸距離(公尺)',	'橋梁最接近斷層距離', '主梁材質_預力混凝土', '設計垂直地震力係數']
        variable_choice = st.selectbox('重要變數', variable_selection)

        # 製圖（散佈圖）
        if variable_choice == '平均月相對濕度':
            color_values = df_bridge_variables['平均月相對濕度']
            fig = go.Figure()
            fig.add_trace(go.Scatter(x = df_bridge_variables[variable_choice], y=df_bridge_variables['U_change_ave'], mode='markers', text=[f"橋梁ID: {id_val}<br>平均月相對濕度: {x_val}<br>U_change_ave: {y_val}" for id_val, x_val, y_val in zip(df_bridge_variables['橋梁 系統ID'], df_bridge_variables[variable_choice], df_bridge_variables['U_change_ave'])],hoverinfo='text', marker=dict(size=10,color=color_values)))
            fig.update_layout(title= variable_choice + '與劣化速度關係圖', xaxis=dict(title=''), yaxis=dict(title=''))
            fig.update_xaxes(title_text = variable_choice, tickformat="d")
            st.plotly_chart(fig)

        if variable_choice == '年平均每日交通量':
            color_values = df_bridge_variables['年平均每日交通量']
            fig = go.Figure()
            fig.add_trace(go.Scatter(x = df_bridge_variables[variable_choice], y=df_bridge_variables['U_change_ave'], mode='markers', text=[f"橋梁ID: {id_val}<br>年平均日交通量: {x_val}<br>U_change_ave: {y_val}" for id_val, x_val, y_val in zip(df_bridge_variables['橋梁 系統ID'], df_bridge_variables[variable_choice], df_bridge_variables['U_change_ave'])],hoverinfo='text', marker=dict(size=10,color=color_values)))
            fig.update_layout(title= variable_choice + '與劣化速度關係圖', xaxis=dict(title=''), yaxis=dict(title=''))
            fig.update_xaxes(title_text = variable_choice, tickformat="d")
            st.plotly_chart(fig)

        if variable_choice == '橋梁總長(M)':
            color_values = df_bridge_variables['橋梁總長(M)']
            fig = go.Figure()
            fig.add_trace(go.Scatter(x = df_bridge_variables[variable_choice], y=df_bridge_variables['U_change_ave'], mode='markers', text=[f"橋梁ID: {id_val}<br>橋梁總長(M): {x_val}<br>U_change_ave: {y_val}" for id_val, x_val, y_val in zip(df_bridge_variables['橋梁 系統ID'], df_bridge_variables[variable_choice], df_bridge_variables['U_change_ave'])],hoverinfo='text', marker=dict(size=10,color=color_values)))
            fig.update_layout(title= variable_choice + '與劣化速度關係圖', xaxis=dict(title=''), yaxis=dict(title=''))
            fig.update_xaxes(title_text = variable_choice, tickformat="d")
            st.plotly_chart(fig)

        if variable_choice == '離海岸距離(公尺)':
            color_values = df_bridge_variables['離海岸距離(公尺)']
            fig = go.Figure()
            fig.add_trace(go.Scatter(x = df_bridge_variables[variable_choice], y=df_bridge_variables['U_change_ave'], mode='markers', text=[f"橋梁ID: {id_val}<br>離海岸距離(公尺): {x_val}<br>U_change_ave: {y_val}" for id_val, x_val, y_val in zip(df_bridge_variables['橋梁 系統ID'], df_bridge_variables[variable_choice], df_bridge_variables['U_change_ave'])],hoverinfo='text', marker=dict(size=10,color=color_values)))
            fig.update_layout(title= variable_choice + '與劣化速度關係圖', xaxis=dict(title=''), yaxis=dict(title=''))
            fig.update_xaxes(title_text = variable_choice, tickformat="d")
            st.plotly_chart(fig)

        if variable_choice == '橋梁最接近斷層距離':
            color_values = df_bridge_variables['橋梁最接近斷層距離']
            fig = go.Figure()
            fig.add_trace(go.Scatter(x = df_bridge_variables[variable_choice], y=df_bridge_variables['U_change_ave'], mode='markers', text=[f"橋梁ID: {id_val}<br>橋梁最接近斷層距離: {x_val}<br>U_change_ave: {y_val}" for id_val, x_val, y_val in zip(df_bridge_variables['橋梁 系統ID'], df_bridge_variables[variable_choice], df_bridge_variables['U_change_ave'])],hoverinfo='text', marker=dict(size=10,color=color_values)))
            fig.update_layout(title= variable_choice + '與劣化速度關係圖', xaxis=dict(title=''), yaxis=dict(title=''))
            fig.update_xaxes(title_text = variable_choice, tickformat="d")
            st.plotly_chart(fig)   
        
        # 製圖（長條圖）
        if variable_choice == '是否為跨水橋':
            st.markdown('註：0為否，1為是')
            value_counts = df_bridge_variables[variable_choice].value_counts()
            count_df = pd.DataFrame({'類別': value_counts.index, '次數': value_counts.values})
            fig = px.bar(count_df, x='類別', y='次數', text='次數', title= variable_choice + '長條圖',color = '類別')
            st.plotly_chart(fig)  

        if variable_choice == '橋下有無租用':
            st.markdown('註：0為否，1為是')
            value_counts = df_bridge_variables[variable_choice].value_counts()
            count_df = pd.DataFrame({'類別': value_counts.index, '次數': value_counts.values})
            fig = px.bar(count_df, x='類別', y='次數', text='次數', title= variable_choice + '長條圖',color = '類別')
            st.plotly_chart(fig)  

        if variable_choice == '主梁材質_預力混凝土':
            st.markdown('註：0為否，1為是')
            value_counts = df_bridge_variables[variable_choice].value_counts()
            count_df = pd.DataFrame({'類別': value_counts.index, '次數': value_counts.values})
            fig = px.bar(count_df, x='類別', y='次數', text='次數', title= variable_choice + '長條圖',color = '類別')
            st.plotly_chart(fig)

        if variable_choice == '設計垂直地震力係數':
            value_counts = df_bridge_variables[variable_choice].value_counts()
            count_df = pd.DataFrame({'類別': value_counts.index, '次數': value_counts.values})
            fig = px.bar(count_df, x='類別', y='次數', text='次數', title= variable_choice + '長條圖',color = '類別')
            st.plotly_chart(fig)

    with tab3:
        st.header('影響橋梁劣化速度重要因子分析')
        st.text('請注意：我們使用的依變數(y)為每月平均劣化速度')
        num_top_features = st.slider('請滑動以選擇想要看的前幾名重要變數', 1, 10, 1)
        fig_ols = px.bar(df_ols.head(num_top_features), x='feature', y='coef', title='Top Features from OLS', color = 'feature')
        st.plotly_chart(fig_ols)
        fig_rf = px.bar(df_rf.head(num_top_features), x='Feature', y='Importance', title='Top Features from Random Forest',color = 'Feature')
        st.plotly_chart(fig_rf)
        fig_elastic = px.bar(df_elastic.head(num_top_features), x='Feature', y='Importance', title='Top Features Elastic Net',color = 'Feature')
        st.plotly_chart(fig_elastic)

    with tab4:    
        st.header('預測橋梁是否可能有快速劣化之風險')
        col1, col2 = st.columns(2)
        with col1:
            br_trafic = st.slider('年平均每日交通量預估', 5000, 150000, 100)
            br_coast = st.slider('離海岸距離(公尺)', 100, 30000, 100)
            br_earthquack_coef = st.slider('設計垂直地震力係數', 0.03, 0.5, 0.01)
            br_loc = st.selectbox('橋梁所在地', df_mapping.iloc[:,0])
            br_loc = df_mapping.loc[df_mapping.iloc[:,0] == br_loc, '6 年平均相對濕度'].values[0]
        with col2:
            br_long = st.slider('橋梁總長(公尺)', 5, 1000, 5)
            br_fault = st.slider('橋梁最接近斷層距離(公尺)', 0, 60000, 100)
            br_thruwater= st.checkbox('是否為跨水橋')
            br_thruwater = 1 if br_thruwater else 0
            br_material= st.checkbox('主梁材質為預力混凝土')
            br_material = 1 if br_material else 0
            
            br_rent= st.checkbox('橋下有無租用')
            br_rent = 1 if  br_rent else 0
        if st.button('Press to Predict'):
            st.markdown('註：我們依據三種模型投票進行分類，每個模型各擁有一票，若票數>=2票則可能為快速劣化橋梁')
            result_rf = predict(np.array([[br_trafic, br_material, br_loc, br_long,br_coast,br_earthquack_coef,br_thruwater,br_fault,br_rent]]))
            result_xgb = predict_xgb(np.array([[br_trafic, br_material, br_loc, br_long,br_coast,br_earthquack_coef,br_thruwater,br_fault,br_rent]]))
            result_lr = predict_lr(np.array([[br_trafic, br_material, br_loc, br_long,br_coast,br_earthquack_coef,br_thruwater,br_fault,br_rent]]))

            col1, col2, col3 = st.columns(3)
            with col1:
                new_title = '<p style="font-family:sans-serif; color:#008000; font-size: 22px;">Random Forest</p>'
                st.markdown(new_title, unsafe_allow_html=True)
                st.markdown(result_rf[0])
            with col2:
                new_title2 = '<p style="font-family:sans-serif; color:#FF6600; font-size: 22px;">XGBoost </p>'
                st.markdown(new_title2, unsafe_allow_html=True)
                st.markdown(result_xgb[0])
            with col3:
                new_title3 = '<p style="font-family:sans-serif; color:#800080; font-size: 22px;">Logistic Regression </p>'
                st.markdown(new_title3, unsafe_allow_html=True)
                st.markdown(result_lr[0])

            if result_rf[0] + result_lr[0] + result_xgb[0] >1.5:
                new_title4 = '<h1 style="text-align: center; font-family:sans-serif; color:red; font-size: 30px;"><span style="text-decoration:underline;">預測結果：可能為快速劣化之橋梁 </p>'
                st.markdown(new_title4, unsafe_allow_html=True)
            else:
                new_title5 = '<h1 style="text-align: center; font-family:sans-serif; color:black; font-size: 30px;"><span style="text-decoration:underline;">預測結果：為快速劣化之橋梁風險較低 </p>'
                st.markdown(new_title5, unsafe_allow_html=True)
    

    


if __name__ == "__main__":
    run()

