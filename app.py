import streamlit as st
import numpy as np
import pandas as pd

from app_classes import Data, NormalDistributionTable, Z_score_value, NormalCurveAreas, NormalCurveGraph, ConvertToOrdinal

st.set_page_config(
   page_title="ANDRUP",
   page_icon=":bar_chart",
   initial_sidebar_state="expanded",
   layout='wide'
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

st.title('ANDRUP')
st.header('Automated Normal Distribution Ranking Using Python ')
st.markdown("""This web application can automatically compute the mean, variance, and standard deviation of data within the given MS excel Sheet, Google Sheets, or Inputted Values. ANDRUP will also determine the z-score, area, and probability percentage of given data points by a user within the dataset supported by a downloadable standard normal distribution curve. ANDRUP also provides ranking analysis based on actual percentile rank and percentile rank determined by the normal distribution model. This web application aims to lessen the time and burden when computing and analyzing with standard normal distribution..
            """)

st.markdown('---')

st.header("Insert Data")
data_type = st.radio("How will you insert your data?",('Excel', 'Google Sheets', 'Values separated by space', 'Values separated by comma'))

try:
    data = list(Data(data_type).acquire_data())
except TypeError:
    pass

st.markdown('---')

with st.container():

    try:

        table_col, deets_col = st.columns(2)
        with table_col:
            df, mean, variance, st_dev = NormalDistributionTable(data).create_distribution_table()
            st.header("Normal Probability Distribution Table")
            st.dataframe(df.style.format({'X': '{:.2f}', 'X*P(X)': '{:.2f}', 'P(X)' : '{:.2f}', 'X-μ' : '{:.2f}', '(X-μ)^2': '{:.2f}', '(X-μ)^2*P(X)' : '{:.2f}'}))
            st.caption("***Note: Values inside the table are rounded off into two decimal places for the display purposes and true values still remain***")

            @st.cache
            def convert_df(nd_table):
                # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return nd_table.to_csv().encode('utf-8')

            csv = convert_df(df)

            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='normal_dist_table.csv',
                mime='text/csv',
            )
            
        with deets_col:
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

        st.markdown('---')

        st.subheader('Single Data Point Analysis')
        X_given = float(st.text_input('Input the data that you want to analyze: ', max_chars=100))

        z, z_area_based_on_z_s_table = Z_score_value(mean, variance, st_dev).get_z_deets(X_given)
        z_left_area, z_l_area_percent,  z_right_area,  z_r_area_percent = NormalCurveAreas(mean, X_given, z_area_based_on_z_s_table).get_z_left_right_deets()
        standard_x_norm = np.arange(-3,3,0.001)
        normal_curve = NormalCurveGraph(standard_x_norm, mean, st_dev)

        if X_given != 0:
            zcol1, zcol2, zcol3= st.columns(3)
            zcol1.metric("X Value",f"{round(X_given,2)}", f"{X_given}", delta_color='normal')
            zcol2.metric("Z Score", f"z = {round(z,2)}", f"{z}", delta_color='normal')
            zcol3.metric("Area", f"{round(z_area_based_on_z_s_table,4)}", f"{z_area_based_on_z_s_table}", delta_color='normal')

            def sda_respect_to_mean(z):
                if z < 0:
                    z0 = z
                    z1 = 0
                    title = f'{round(z,2)} < z < 0'
                
                elif z > 0:
                    z1 = z
                    z0 = 0
                    title = f'{round(z,2)} > z > 0'

                return z0, z1, title

            z_0, z_1, title_g =  sda_respect_to_mean(z)
            z_mean_graph = normal_curve.draw_z_score((z_0 < standard_x_norm) & (standard_x_norm < z_1), 0, 1, title_g)
            st.pyplot(fig=z_mean_graph)

            left_deets_col, right_deets_col = st.columns(2)
            left_deets_col.subheader("Left side of the Z-score")
            right_deets_col.subheader("Right side of the Z-score")

            left_sm_col, left_area_col, right_sm_col, right_area_col = st.columns(4)
            left_sm_col.metric("Left Side Area", f"{round(z_left_area,4)}")
            left_area_col.metric("Probability", f"{round(z_l_area_percent,2)}%")
            right_sm_col.metric("Right Side Area", f"{round(z_right_area,4)}")
            right_area_col.metric("Probability", f"{round(z_r_area_percent,2)}%")

            left_graph_col, right_graph_col = st.columns(2)
            with left_graph_col:
                left_sncurve = normal_curve.draw_z_score(standard_x_norm<z, 0, 1, f'z < {round(z,2)}')
                st.pyplot(fig=left_sncurve)
            with right_graph_col:
                right_sncurve = normal_curve.draw_z_score(standard_x_norm>z, 0, 1, f'z > {round(z,2)}')
                st.pyplot(fig=right_sncurve)


        else:
            st.markdown(":red[Enter a Data Point]")

        st.header('Double Data Point Analysis')
        dp1_col, dp2_col = st.columns(2)
        dp1 = dp1_col.number_input('Enter a data point', key='dp1')
        dp2 = dp2_col.number_input('Enter a data point', key='dp2')

        dp1_z_score, dp1_area = Z_score_value(mean, variance, st_dev).get_z_deets(dp1)
        dp2_z_score, dp2_area = Z_score_value(mean, variance, st_dev).get_z_deets(dp2)


        if dp1 != dp2:

            def double_data_point(dp1_z_score, dp2_z_score):
                if dp1_z_score > dp2_z_score:
                    z_0 = dp2_z_score
                    z_1 = dp1_z_score
                    title_graph = f'{round(dp2_z_score,2)} < z < {round(dp1_z_score,2)}'
                else:
                    z_0 = dp1_z_score
                    z_1 = dp2_z_score
                    title_graph = f'{round(dp1_z_score,2)} < z < {round(dp2_z_score,2)}'

                return z_0, z_1, title_graph 
            
            def dp1dp2_area(dp1_z_score, dp2_z_score, dp1_area, dp2_area):
                if dp1_z_score == dp2_z_score:
                    area = "INVALID"
                elif dp1_z_score < 0 and dp2_z_score > 0:
                    area = max(dp1_area, dp2_area) + min(dp1_area, dp2_area)
                elif dp1_z_score > 0 and dp2_z_score < 0:
                    area = max(dp1_area, dp2_area) + min(dp1_area, dp2_area)
                elif dp1_z_score < 0 and dp2_z_score < 0:
                    area = max(dp1_area, dp2_area) - min(dp1_area, dp2_area)
                elif dp1_z_score > 0 and dp2_z_score > 0:
                    area = max(dp1_area, dp2_area) - min(dp1_area, dp2_area)
                else:
                    area = "INVALID"
                return area

            dp1dp2_area = dp1dp2_area(dp1_z_score, dp2_z_score, dp1_area, dp2_area)

            dp1_z_col, dp2_z_col, dp1dp2_area_col, dp1dp2_proba_col = st.columns(4)
            dp1_z_col.metric(f"Z-score of {dp1}", f'{round(dp1_z_score,2)}')
            dp2_z_col.metric(f"Z-score of {dp2}", f'{round(dp2_z_score,2)}')
            dp1dp2_area_col.metric(f"In Between Area", f'{round(dp1dp2_area,4)}')
            dp1dp2_proba_col.metric(f"Probability", f'{round(round(dp1dp2_area,4) * 100,2)}%')
            zdp1, zdp2, dp1dp2title = double_data_point(dp1_z_score, dp2_z_score)

            dp1dp2_graph = normal_curve.draw_z_score((zdp1 < standard_x_norm) & (standard_x_norm < zdp2), 0, 1, dp1dp2title)
            st.pyplot(fig=dp1dp2_graph)
        else:
            st.markdown(f":red[Your input for the 1st data point must not be equal to your input for the 2nd data point ({dp1}≠{dp2})]")

        st.header("Ranking Analysis")
        df_ranks = pd.DataFrame(data, columns=['Data'],dtype='float64')
        df_ranks['Default Rank'] = df_ranks['Data'].rank(ascending=False)
        df_ranks['Percentile Rank'] = df_ranks['Data'].rank(pct=True)
        
        def get_left_areas_rank(df_ranks, mean, st_dev, variance):
            norm_s_dist = []
            for val in df_ranks['Data'].values:
                z_scr, based_z_area = Z_score_value(mean, variance, st_dev).get_z_deets(val)
                z_left_area = NormalCurveAreas(mean, val, based_z_area).get_z_left_right_deets()[0]
                norm_s_dist.append(z_left_area)
            return norm_s_dist

        norm_s_dist_list = get_left_areas_rank(df_ranks, mean, st_dev, variance)

        df_ranks['Normal Probability Distribution Rank'] = np.array(norm_s_dist_list)

        for val in df_ranks['Default Rank']:
            df_ranks['Default Rank'] = df_ranks['Default Rank'].replace(val, ConvertToOrdinal(val).convert_to_ordinal())

        df_ranks = df_ranks.sort_values(by=['Data'],ascending=False)

        st.dataframe(df_ranks)

    except (ValueError, NameError):
        pass
