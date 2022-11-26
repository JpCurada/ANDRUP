
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import collections
import math as m
import scipy.stats as stats
import matplotlib.pyplot as plt
# import xlsxwriter
# from io import BytesIO

st.set_page_config(
   page_title="ANDRUP",
   page_icon=":bar_chart",
   layout="wide"
#    initial_sidebar_state="expanded",
)

plt.style.use('ggplot')

st.title('ANDRUP')
st.header('A Web Application for Automated Normal Distribution Ranking Using Python ')
st.write("Many people, especially students and teachers, experience difficulty in applications or problems involving normal distribution. Time is valuable for statisticians, students, teachers, and other professionals. Every detail needs to be correct and accurate. Manual computations often result in errors that might lead to false or wrong results. ")

st.write("Therefore, this web application is created to automate the manual processing of various data for computing, ranking, and visualizing numerical data provided by users. This endeavor aims to lessen the time allocated to manual computations and make their day productive, allowing them to dedicate their time to other workloads and get the chance to do various works. ") 
       
with st.expander('How to Use?'):
    st.write("""
        The chart above shows some numbers I picked for you.
        I rolled actual dice for these, so they're *guaranteed* to
        be random.
    """)

st.text('(May video tutorial na lalabas dito, gagawin natin yon soon)')

st.subheader('Features')

feature1, feature2, feature3, feature4, feature5 = st.tabs(['[1] Insert Data','[2] Normal Probability Distribution Table','[3] Z-Score Mapping','[4] Percentile Ranking', 'About Us'])

with feature1:
   st.header("Insert Data")
   data_type = st.radio("How will you insert your data?",('Excel', 'Google Sheets', 'Input separated by space', 'Input separated by comma'))

   def acquire_data(data_type):

        if data_type == 'Excel':
            st.write('Excel sample')
            try:
                uploaded_excel_file = st.file_uploader("Choose an Excel file", accept_multiple_files=False)
                bytes_data = uploaded_excel_file.read()
                st.write("filename:", uploaded_excel_file.name)
                df_excel = pd.read_excel(bytes_data)
                print(df_excel)
                st.dataframe(df_excel)
                data = np.array([round(X,2) for X in df_excel[df_excel.columns[1]].values])
                return data
            except AttributeError:
                pass  

        elif data_type == 'Google Sheets':
            st.write("Google Sheet Sample")
            pass

        elif data_type == 'Input separated by space':
            st.write("Separate the number by a space(  ) to be readable")
            st.caption('Example: 50 51.20 53 54 55.55 57 58 59 60')
            try:
                space_data = (st.text_input('Input your data:'))
                space_raw_data = [float(x) for x in space_data.split(' ')]
                st.caption(space_raw_data)
                space_rounded_data = [round(val, 2) for val in space_raw_data]
                df_space = pd.DataFrame({'X':np.array(space_rounded_data)})
                st.dataframe(df_space)
                data = space_rounded_data
                return data
            except ValueError:
                pass


        elif data_type == 'Input separated by comma':
            st.write("Separate the number by a comma ( , ) to be readable")
            st.caption('Example: 51.50,52.50,53.50,54.50,54,57,59,61,62.30')
            try:
                comma_data = (st.text_input('Input your data:'))
                comma_raw_data = [float(x) for x in comma_data.split(',')]
                comma_rounded_data = [round(val, 2) for val in comma_raw_data]
                df_comma = pd.DataFrame({'X':np.array(comma_rounded_data)})
                st.dataframe(df_comma)
                data = comma_rounded_data
                return data
            except ValueError:
                pass
   
   try:
        data = list(acquire_data(data_type))
   except TypeError:
        pass

with feature2:

    def create_distribution_table(data):
        # Create df based on the data
        df = pd.DataFrame(data)
        df = df.rename(columns={df.columns[0]:"X"})

        # drop duplicate x
        df = df.drop_duplicates(keep='first')

        # get the frequency of each values (x) of the data
        frequency=collections.Counter(data).values()
        
        # Insert the column for frequency
        df.insert(1,'Frequency', frequency)
        
        # Sort values by least to greatest, reset index, and set column name to 'X'
        df = df.sort_values(by=['X']).reset_index(drop=True)
        
        # get P(x) or the probability of x and insert
        df.insert(2, 'P(X)', np.array([f / len(data) for f in df['Frequency'].values]))
        
        # multiply the x values to the p(x) values
        df.insert(3, 'X*P(X)', df['X'].values * df['P(X)'].values)
        
        # get the mean of the data or summation of x*p(x)
        mean = (df['X'].values * df['P(X)'].values).sum()
        
        # subtract the value of x to mean
        df.insert(4,'X-μ', np.array([val-mean for val in df['X'].values]))
        
        # raise to the power of the difference between the value of x and mean
        df.insert(5,'(X-μ)^2', np.array([val**2 for val in df['X-μ'].values]))
        
        # multiply the '(x-μ)^2' values to p(x) values
        df.insert(6, '(X-μ)^2*P(X)', (df['(X-μ)^2'].values * df['P(X)'].values))
        
        # get the variance by getting the sum of '(x-μ)^2' column and subtracting it to the number of 'x'
        variance = sum(df['(X-μ)^2*P(X)'].values)
        
        # get the standard deviation by getting the square root of variance
        st_dev = m.sqrt(variance)
        
        return df, mean, variance, st_dev
    
    try:
        ndtcol1, ndtcol2= st.columns(2)
        with ndtcol1:
            st.header("Normal Probability Distribution Table")
            nd_table, mean, variance, st_dev = create_distribution_table(data)
            st.dataframe(nd_table.style.format({'X': '{:.2f}', 'X*P(X)': '{:.2f}', 'P(X)' : '{:.2f}', 'X-μ' : '{:.2f}', '(X-μ)^2': '{:.2f}', '(X-μ)^2*P(X)' : '{:.2f}'}))
            @st.cache
            def convert_df(nd_table):
                # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return nd_table.to_csv().encode('utf-8')

            csv = convert_df(nd_table)

            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='normal_dist_table.csv',
                mime='text/csv',
            )
        
        with ndtcol2:
            st.header('Mean, Variance and Standard Deviation')
            st.metric("Mean", f"{mean}")
            st.write('''
                    _The mean is the central tendency of the normal distribution. It defines the location of the peak for the bell curve._
                    ''')
            st.metric("Variance", f"{variance}")
            st.write('''
                    _The variance measures the average degree to which each point differs from the mean. While standard deviation is the square root of the variance, variance is the average of all data points within a group._
                    ''')
            st.metric("Standard Deviation", f"{st_dev}")
            st.write('''
                    _The standard deviation is the measure of how spread out a normally distributed set of data is. It is a statistic that tells you how closely all of the examples are gathered around the mean in a data set._
                    ''')

    except (TypeError, NameError) as e:
        pass

with feature3:
    st.header("Data Analysis")
    st.write('''
            _The value of the **z-score** tells you how many standard deviations you are away from the mean. If a z-score is equal to 0, it is on the mean. A positive z-score indicates the raw score is higher than the mean._ 
            ''')

    def get_z_deets(X, mean, st_dev):
        
        # Determine the value of z-score
        z = round((X - mean) / st_dev,2)
        
        # area of z-score value based on the z-table
        z_area_based_on_z_s_table = abs(round(0.50 - stats.norm.cdf(z),4)) 
        return z, z_area_based_on_z_s_table

    def get_z_left_right_deets(X_given, z, mean):
        if X_given > mean:
           # get the left side's area of z-score value on normal curve 
            z_left_area = round(0.5000 + z,4)
            z_l_area_percent = round(z_left_area * 100,2)
            # get the rides side's area of z-score value on normal curve
            z_right_area = round(0.5000 - z,4)
            z_r_area_percent = round(z_right_area * 100,2)
            return z_left_area, z_l_area_percent,  z_right_area,  z_r_area_percent
        elif X_given < mean:
            # get the left side's area of z-score value on normal curve 
            z_left_area = round(0.5000 - z,4)
            z_l_area_percent = round(z_left_area * 100,2)
            # get the rides side's area of z-score value on normal curve
            z_right_area = round(0.5000 + z,4)
            z_r_area_percent = round(z_right_area * 100,2)
            return z_left_area, z_l_area_percent,  z_right_area,  z_r_area_percent
        else:
            # get the left side's area of z-score value on normal curve 
            z_left_area = round(0.5000 - z,4)
            z_l_area_percent = round(z_left_area * 100,2)
            # get the rides side's area of z-score value on normal curve
            z_right_area = round(0.5000 + z,4)
            z_r_area_percent = round(z_right_area * 100,2)
            return z_left_area, z_l_area_percent,  z_right_area,  z_r_area_percent

    def draw_z_score(x, cond, s_mean, s_st_dev, title):
        
        plt.figure(figsize=(15,7))
        y = stats.norm.pdf(x, s_mean, s_st_dev)
        z = x[cond]
        plt.plot(x, y, color='black', lw=5)
        plt.fill_between(z, 0, stats.norm.pdf(z, s_mean, s_st_dev))
        
        # change standard x-sticks 
        plot_xticks = [-3,-2,-1,0,1,2,3]
        # set new ticks (mean and st_dev variations)
        plt.xticks(plot_xticks, [round(mean-(3*st_dev),2), round(mean-(2*st_dev),2), round(mean-st_dev,2), round(mean,2), round(mean+st_dev,2), round(mean+(2*st_dev),2), round(mean+(3*st_dev),2)])
        
        plt.title(title)
        return plt

    try:

        with st.container():
            st.header('Single Data Point Analysis')
            X_given = float(st.number_input('Input the data that you want to analyze: '))
            # Show X value, Z score of X, Area from the mean
            if X_given != 0:
                z, z_area_based_on_z_s_table = get_z_deets(X_given, mean, st_dev)
                zcol1, zcol2, zcol3= st.columns(3)
                with zcol1:
                    st.metric("X Value",f"{X_given}")
                with zcol2:
                    st.metric("Z Score", f"z = {z}")
                with zcol3:
                    st.metric("Area", f"{z_area_based_on_z_s_table}")

                z_left_area, z_l_area_percent,  z_right_area,  z_r_area_percent = get_z_left_right_deets(X_given, z_area_based_on_z_s_table, mean)
                standard_x_norm = np.arange(-3,3,0.001)

                st.markdown('---')
                left_z, right_z = st.columns(2)

                left_subheader, right_subheader = st.columns(2)
                left_subheader.subheader("Left side of the Z-score's details")
                right_subheader.subheader("Right side of the Z-score's details")

                left_area, left_percent, right_area, right_percent = st.columns(4)
                left_area.metric("Left Side Area", f"{z_left_area}")
                left_percent.metric("Area Percentage", f"{z_l_area_percent}%")
                right_area.metric("Right Side Area", f"{z_right_area}")
                right_percent.metric("Area Percentage", f"{z_r_area_percent}%")

                left_g, right_g = st.columns(2)
                with left_g:
                    # left_area_col, left_a_percentage_col = st.columns(2)
                    # left_area_col.metric("Left Side Area", f"{z_left_area}")
                    # left_a_percentage_col.metric("Area Percentage", f"{z_l_area_percent}%")
                    z_left_graph = draw_z_score(standard_x_norm, standard_x_norm<z, 0, 1, f'z < {z}')
                    st.pyplot(fig=z_left_graph )

                with right_g:
                    # right_area_col, right_a_percentage_col = st.columns(2)
                    # right_area_col.metric("Right Side Area", f"{z_right_area}")
                    # right_a_percentage_col.metric("Area Percentage", f"{z_r_area_percent}%")
                    z_right_graph = draw_z_score(standard_x_norm, standard_x_norm>z, 0, 1, f'z > {z}')
                    st.pyplot(fig=z_right_graph )

                z_graph, analysis = st.columns(2)
                with z_graph:
                    st.subheader("Z-score's details with respect to mean")

                    def sda_respect_to_mean(z):
                        if z < 0:
                            z0 = z
                            z1 = 0
                            title = f'0 < z < {z}'
                        
                        elif z > 0:
                            z1 = z
                            z0 = 0
                            title = f'0 > z > {z}'

                        return z0, z1, title

                    z0, z1, title =  sda_respect_to_mean(z)
                    z_mean_graph = draw_z_score(standard_x_norm, (z0 < standard_x_norm) & (standard_x_norm < z1), 0, 1, title=title)
                    st.pyplot(fig=z_mean_graph)
                with analysis:
                    st.subheader('Interpretation')

            else:
                st.caption('Enter a data point')

    except (TypeError, NameError) as e:
        pass

    try:

        with st.container():
            st.markdown('---')
            st.header('Double Data Point Analysis')
        

            dp1_col, dp2_col = st.columns(2)
            dp1 = dp1_col.number_input('Enter a data point', key='dp1')
            dp2 = dp2_col.number_input('Enter a data point', key='dp2')
            if dp1 != 0 and dp2 != 0:
                dp1_z, dp1_area, dp2_z, dp2_area = st.columns(4)
                dp1_z.metric("Z-score", f'{get_z_deets(dp1, mean, st_dev)[0]}')
                dp1_area.metric("Area", f'{get_z_deets(dp1, mean, st_dev)[1]}')
                dp2_z.metric("Z-score", f'{get_z_deets(dp2, mean, st_dev)[0]}')
                dp2_area.metric("Area", f'{get_z_deets(dp2, mean, st_dev)[1]}')

                st.subheader('Between the two data points')

                dp1_z_score, dp1_z_area = get_z_deets(dp1 ,mean, st_dev)
                dp2_z_score, dp2_z_area = get_z_deets(dp2 ,mean, st_dev)

                def dda_z(dp1_z_score, dp2_z_score):
                    if dp1_z_score < dp2_z_score:
                        z_0 = dp1_z_score
                        z_1 = dp2_z_score
                        title = f'{z_0} < z < {z_1}'
                    
                    elif dp1_z_score > dp2_z_score:
                        z_0 = dp2_z_score
                        z_1 = dp1_z_score
                        title = f'{z_1} > z > {z_0}'

                    return z_0, z_1, title

                z_0, z_1, title_g =  dda_z(dp1_z_score, dp2_z_score)

                z_area_graph, z_area_perccentage = st.columns(2)
                z_area_graph.metric("Area", f'{dp1_z_area + dp2_z_area}')
                z_area_perccentage.metric("Percentage", f'{round((dp1_z_area + dp2_z_area)*100,2)}%')

                z_between_graph = draw_z_score(standard_x_norm, (z_0 < standard_x_norm) & (standard_x_norm < z_1), 0, 1, title_g)
                st.pyplot(fig=z_between_graph)

            else:
                st.caption('Enter two data points')

    except (TypeError, NameError) as e:
        pass

with feature4:
   st.header("Ranking Analysis")

with feature5:
#     from PIL import Image
#     from researchers import alf, jeson, bem, jp, sh, yv
#     alf_img = Image.open('alf.png')
#     jeson_img = Image.open('jeson.png')
#     bem_img = Image.open('bem.png')
#     jp_img = Image.open('jp.png')
#     sh_img = Image.open('sh.png')
#     yv_img = Image.open('yv.png')

      st.header("Researchers")
#     pic_alf, pic_jes, pic_bem, pic_jp, pic_yv,pic_sh= st.columns(6)
#     with pic_alf:
#         st.image(alf_img, caption='Alfredo L. Agulto III')
#     with pic_jes:
#         st.image(jeson_img, caption='Romnel Jesuron B. Burnot')
#     with pic_bem:
#         st.image(bem_img, caption='Bem Eiffel C. Castillo')
#     with pic_jp:
#         st.image(jp_img, caption='John Paul M. Curada')
#     with pic_yv:
#         st.image(yv_img, caption='Yvonne R. Mangulabnan')
#     with pic_sh:
#         st.image(sh_img, caption='Sharmaine Carl M. Perez')

    

st.markdown('---')
