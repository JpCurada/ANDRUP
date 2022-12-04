
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import collections
import math as m
import scipy.stats as stats
import matplotlib.pyplot as plt
import gspread as gs

# import xlsxwriter
# from io import BytesIO

st.set_page_config(
   page_title="ANDRUP",
   page_icon=":bar_chart",
#    initial_sidebar_state="expanded",
)

st.write(
    """
    <style>
    [data-testid="stMetricDelta"] svg {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
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

st.subheader('DATA ANALYSIS')

feature1, feature2, feature3, feature4 = st.tabs(['[1] Insert Data','[2] Normal Probability Distribution Table','[3] Z-Score Mapping','[4] Ranking'])

with feature1:
   st.header("Insert Data")
   data_type = st.radio("How will you insert your data?",('Excel', 'Google Sheets', 'Values separated by space', 'Values separated by comma'))

   def acquire_data(data_type):

        if data_type == 'Excel':
            st.subheader('Microsoft Excel')
            try:
                uploaded_excel_file = st.file_uploader("Choose an MS Excel file", accept_multiple_files=False)
                bytes_data = uploaded_excel_file.read()
                st.write("Filename: ", uploaded_excel_file.name)
                df_excel = pd.read_excel(bytes_data)
                print(df_excel)
                st.dataframe(df_excel)
                data = np.array([X for X in df_excel[df_excel.columns[1]].values])
                return data
            except AttributeError:
                pass  

        elif data_type == 'Google Sheets':
            st.subheader('Google Sheet')
            gc = gs.service_account(filename='youtube-api-project-364001-624c994c3ce4.json')
            try:
                gsheet_link = st.text_input('Google Sheet Link')
                sh = gc.open_by_url(gsheet_link)
                ws_name = st.text_input('Worksheet Name')
                ws = sh.worksheet(ws_name)
                df_gs = pd.DataFrame(ws.get_all_records())
                st.dataframe(df_gs)
                num_columns = [col for col in df_gs.columns if df_gs[col].dtype == 'float64' or df_gs[col].dtype == 'int64']
                chosen_column = st.selectbox('Choose the column that you want to analyze', tuple(num_columns))
                data = pd.DataFrame(df_gs[chosen_column])
                st.code(np.array(data[chosen_column].values))
                return data[chosen_column].values
            except gs.NoValidUrlKeyFound:
                st.error('Enter a valid URL')
            except gs.WorksheetNotFound:
                st.error('Enter a worksheet name found in your provided google sheet')
            except (AttributeError, ValueError):
                pass

        elif data_type == 'Values separated by space':
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


        elif data_type == 'Values separated by comma':
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
    
        st.header('Mean, Variance and Standard Deviation')
        st.metric("Mean", f"{round(mean,2)}")
        st.caption(f"{mean}")
        st.write('''
                _The mean is the central tendency of the normal distribution. It defines the location of the peak for the bell curve._
                ''')
        st.metric("Mean", f"{round(variance,2)}")
        st.caption(f"{variance}")
        st.write('''
                _The variance measures the average degree to which each point differs from the mean. While standard deviation is the square root of the variance, variance is the average of all data points within a group._
                ''')
        st.metric("Mean", f"{round(st_dev,2)}")
        st.caption(f"{st_dev}")
        st.write('''
                _The standard deviation is the measure of how spread out a normally distributed set of data is. It is a statistic that tells you how closely all of the examples are gathered around the mean in a data set._
                ''')

    except (TypeError, NameError) as e:
        pass

with feature3:
    st.header(" Visual Graphs")
    st.write('''
            _The value of the **z-score** tells you how many standard deviations you are away from the mean. If a z-score is equal to 0, it is on the mean. A positive z-score indicates the raw score is higher than the mean._ 
            ''')

    def get_z_deets(X, mean, st_dev):
        
        # Determine the value of z-score
        z = (X - mean) / st_dev
        
        # area of z-score value based on the z-table
        z_area_based_on_z_s_table = abs(0.50 - stats.norm.cdf(round(z,2)))
        
        return round(z,2), round(z_area_based_on_z_s_table,4)

    def get_z_left_right_deets(X_given, z, mean):
        if X_given > mean:
           # get the left side's area of z-score value on normal curve 
            z_left_area = 0.5000 + z
            z_l_area_percent = z_left_area * 100
            # get the rides side's area of z-score value on normal curve
            z_right_area = 0.5000 - z
            z_r_area_percent = z_right_area * 100
            return z_left_area, z_l_area_percent,  z_right_area,  z_r_area_percent
        elif X_given < mean:
            # get the left side's area of z-score value on normal curve 
            z_left_area = 0.5000 - z
            z_l_area_percent = z_left_area * 100
            # get the rides side's area of z-score value on normal curve
            z_right_area = 0.5000 + z
            z_r_area_percent = z_right_area * 100
            return z_left_area, z_l_area_percent,  z_right_area,  z_r_area_percent
        else:
            # get the left side's area of z-score value on normal curve 
            z_left_area = 0.5000 - z
            z_l_area_percent = z_left_area * 100
            # get the rides side's area of z-score value on normal curve
            z_right_area = 0.5000 + z
            z_r_area_percent = z_right_area * 100
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
                    st.metric("X Value",f"{round(X_given,2)}", f"{X_given}", delta_color='normal')

                with zcol2:
                    st.metric("Z Score", f"z = {round(z,2)}", f"{z}", delta_color='normal')
             
                with zcol3:
                    st.metric("Area", f"{round(z_area_based_on_z_s_table,4)}", f"{z_area_based_on_z_s_table}", delta_color='normal')
   

                z_left_area, z_l_area_percent,  z_right_area,  z_r_area_percent = get_z_left_right_deets(X_given, z_area_based_on_z_s_table, mean)
                standard_x_norm = np.arange(-3,3,0.001)

                left_z, right_z = st.columns(2)

                left_subheader, right_subheader = st.columns(2)
                left_subheader.subheader("Left side of the Z-score")
                right_subheader.subheader("Right side of the Z-score")

                left_area, left_percent, right_area, right_percent = st.columns(4)
                left_area.metric("Left Side Area", f"{round(z_left_area,4)}")
                left_percent.metric("Area Percentage", f"{round(z_l_area_percent,2)}%")
                right_area.metric("Right Side Area", f"{round(z_right_area,4)}")
                right_percent.metric("Area Percentage", f"{round(z_r_area_percent,2)}%")

                left_g, right_g = st.columns(2)
                with left_g:
                    z_left_graph = draw_z_score(standard_x_norm, standard_x_norm<z, 0, 1, f'z < {z}')
                    st.pyplot(fig=z_left_graph )

                with right_g:
                    z_right_graph = draw_z_score(standard_x_norm, standard_x_norm>z, 0, 1, f'z > {z}')
                    st.pyplot(fig=z_right_graph )

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

                def DPs_area(dp1_z_score, dp2_z_score, dp1_z_area, dp2_z_area):
                    DPs_list = [dp1_z_area, dp2_z_area] 
                    if dp1_z_score > 0 and dp2_z_score > 0:
                        shaded_area = max(DPs_list) - min(DPs_list)
                    elif dp1_z_score < 0 and dp2_z_score < 0:
                        shaded_area = max(DPs_list) - min(DPs_list)
                    else:
                        shaded_area = dp1_z_area + dp2_z_area
                    return shaded_area

                shaded_area = DPs_area(dp1_z_score, dp2_z_score, dp1_z_area, dp2_z_area)

                z_area_graph, z_area_perccentage = st.columns(2)
                z_area_graph.metric("Area", f'{shaded_area}')
                z_area_perccentage.metric("Percentage", f'{round((shaded_area)*100,2)}%')

                z_between_graph = draw_z_score(standard_x_norm, (z_0 < standard_x_norm) & (standard_x_norm < z_1), 0, 1, title_g)
                st.pyplot(fig=z_between_graph)

            else:
                st.caption('Enter two data points')

    except (TypeError, NameError) as e:
        pass

with feature4:
    try:
        st.header("Ranking Analysis")
        df_ranks = pd.DataFrame(data, columns=['Data'],dtype='float64')
        df_ranks['Default Rank'] = df_ranks['Data'].rank(ascending=False)
        df_ranks['Percentile Rank'] = df_ranks['Data'].rank(pct=True)
        
        def get_left_areas_rank(df_ranks, mean, st_dev):
            norm_s_dist = []
            for val in df_ranks['Data'].values:
                z_scr, based_z_area = get_z_deets(val, mean, st_dev)
                z_left_area = get_z_left_right_deets(val, based_z_area, mean)[0]
                norm_s_dist.append(z_left_area)
            return norm_s_dist

        norm_s_dist_list =  get_left_areas_rank(df_ranks, mean, st_dev)

        df_ranks['Normal Probability Distribution Rank'] = np.array(norm_s_dist_list)

        df_ranks = df_ranks.sort_values(by=['Data'],ascending=False)

        st.dataframe(df_ranks)
    except NameError:
        pass

st.markdown('---')
