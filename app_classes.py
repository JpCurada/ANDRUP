import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import collections
import math as m
import scipy.stats as stats
import matplotlib.pyplot as plt
import gspread as gs
import streamlit as st


plt.style.use('ggplot')

class Data:
    def __init__(self, data_type):
        self.data_type = data_type

    def acquire_data(self):
        if self.data_type == 'Excel':
            try:
                uploaded_excel_file = st.file_uploader("Choose an MS Excel file", accept_multiple_files=False)
                bytes_data = uploaded_excel_file.read()
                st.write("Filename: ", uploaded_excel_file.name)
                df_excel = pd.read_excel(bytes_data)
                st.dataframe(df_excel)
                df_columns = [col for col in df_excel.columns if df_excel[col].dtype == 'float64' or df_excel[col].dtype == 'int64']
                chosen_excel_column = st.selectbox('Choose the column that you want to analyze', tuple(df_columns))
                data = pd.DataFrame(df_excel[chosen_excel_column])
                return data[chosen_excel_column].values
            except (AttributeError, ValueError):
                pass  
            except IndexError:
                st.error('REMOVE values within a column that are grouped with NaN values or REPLACE NaN values')

        elif self.data_type == 'Google Sheets':
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

                return data[chosen_column].values
            except gs.NoValidUrlKeyFound:
                st.error('Enter a valid URL')
            except gs.WorksheetNotFound:
                st.error('Enter a worksheet name found in your provided google sheet')
            except gs.exceptions.APIError:
                st.error('Set the permission access to "Anyone with the link"')
            except (AttributeError, ValueError, ):
                pass

        elif self.data_type == 'Values separated by space':
            st.write("Separate the number by a space(  ) to be readable")
            st.caption('Example: 50 51.20 53 54 55.55 57 58 59 60')
            try:
                space_data = (st.text_input('Input your data:'))
                space_raw_data = [float(x) for x in space_data.split(' ')]
                st.caption(space_raw_data)
                space_rounded_data = [round(val, 2) for val in space_raw_data]
                data = space_rounded_data
                return data
            except ValueError:
                pass

        elif self.data_type == 'Values separated by comma':
            st.write("Separate the number by a comma ( , ) to be readable")
            st.caption('Example: 51.50,52.50,53.50,54.50,54,57,59,61,62.30')
            try:
                comma_data = (st.text_input('Input your data:'))
                comma_raw_data = [float(x) for x in comma_data.split(',')]
                comma_rounded_data = [round(val, 2) for val in comma_raw_data]
                data = comma_rounded_data
                return data
            except ValueError:
                pass

class NormalDistributionTable:

    def __init__(self, data):
        self.data = data

    def create_distribution_table(self):
        # Create df based on the data
        df = pd.DataFrame(self.data)
        df = df.rename(columns={df.columns[0]:"X"})

        # drop duplicate x
        df = df.drop_duplicates(keep='first')

        # get the frequency of each values (x) of the data
        frequency=collections.Counter(self.data).values()
        
        # Insert the column for frequency
        df.insert(1,'Frequency', frequency)
        
        # Sort values by least to greatest, reset index, and set column name to 'X'
        df = df.sort_values(by=['X']).reset_index(drop=True)
        
        # get P(x) or the probability of x and insert
        df.insert(2, 'P(X)', np.array([f / len(self.data) for f in df['Frequency'].values]))
        
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

class Z_score_value:

    def __init__(self, mean, variance, st_dev):
        self.mean = mean
        self.variance = variance
        self.st_dev = st_dev

    def get_z_deets(self, X):

        # Determine the value of z-score
        z = (X - self.mean) / self.st_dev
        
        # area of z-score value based on the z-table
        z_area_based_on_z_s_table = abs(0.50 - stats.norm.cdf(round(z,2)))
        
        return z, z_area_based_on_z_s_table

class NormalCurveAreas:

    def __init__(self, mean, X, z_area_based_on_z_s_table):
        self.mean = mean
        self.X = X
        self.z_area_based_on_z_s_table = z_area_based_on_z_s_table

    def get_z_left_right_deets(self):

        if self.X > self.mean:
           # get the left side's area of z-score value on normal curve 
            z_left_area = 0.5000 + self.z_area_based_on_z_s_table
            z_l_area_percent = z_left_area * 100
            # get the rides side's area of z-score value on normal curve
            z_right_area = 0.5000 - self.z_area_based_on_z_s_table
            z_r_area_percent = z_right_area * 100

        elif self.X < self.mean:
            # get the left side's area of z-score value on normal curve 
            z_left_area = 0.5000 - self.z_area_based_on_z_s_table
            z_l_area_percent = z_left_area * 100
            # get the rides side's area of z-score value on normal curve
            z_right_area = 0.5000 + self.z_area_based_on_z_s_table
            z_r_area_percent = z_right_area * 100

        else:
            # get the left side's area of z-score value on normal curve 
            z_left_area = 0.5000 - self.z_area_based_on_z_s_table
            z_l_area_percent = z_left_area * 100
            # get the rides side's area of z-score value on normal curve
            z_right_area = 0.5000 + self.z_area_based_on_z_s_table
            z_r_area_percent = z_right_area * 100
            
        return z_left_area, z_l_area_percent,  z_right_area,  z_r_area_percent

class NormalCurveGraph:

    def __init__(self, standard_x_norm, mean, st_dev):
        self.standard_x_norm = standard_x_norm
        self.mean =  mean
        self.st_dev = st_dev
        # self.left_sxnorm = standard_x_norm<z
        # self.right_sxnorm = standard_x_norm>z
        # self.between_sxnorm = (z0 < standard_x_norm) & (standard_x_norm < z1)
        # f'z < {z}'
        # f'z > {z}'

    def draw_z_score(self, cond, s_mean, s_st_dev, title):
        plt.figure(figsize=(15,7))
        y = stats.norm.pdf(self.standard_x_norm, s_mean, s_st_dev)
        z = self.standard_x_norm[cond]
        plt.plot(self.standard_x_norm, y, color='black', lw=5)
        plt.fill_between(z, 0, stats.norm.pdf(z, s_mean, s_st_dev))
        
        # change standard x-sticks 
        plot_xticks = [-3,-2,-1,0,1,2,3]
        # set new ticks (mean and st_dev variations)
        plt.xticks(plot_xticks, [round(self.mean-(3*self.st_dev),2), round(self.mean-(2*self.st_dev),2), round(self.mean-self.st_dev,2), round(self.mean,2), round(self.mean+self.st_dev,2), round(self.mean+(2*self.st_dev),2), round(self.mean+(3*self.st_dev),2)])
        
        plt.title(title)
        return plt

class ConvertToOrdinal:
    
    def __init__(self, val):
        self.val = val
        
class ConvertToOrdinal:
    
    def __init__(self, val):
        self.val = val
        
    def convert_to_ordinal(self):
        
        val = str(self.val)
        
        if val.isdigit() == True:
 
            if list(val)[-1] == str(1):
                val = str(val)+"st"
            elif list(val)[-1] == str(2):
                val = val+"nd"
            elif list(val)[-1] == str(3):
                val = (val)+"rd"
            else:
                val = (val)+"th"
                
        elif val.split('.')[-1] == '0':
            val = val.rstrip('0').rstrip('.')
            if list(val)[-1] == str(1):
                val = str(val)+"st"
            elif list(val)[-1] == str(2):
                val = val+"nd"
            elif list(val)[-1] == str(3):
                val = (val)+"rd"
            else:
                val = (val)+"th"
                
        else:
            val = (val)+"th"

        return val




# data = NormalDistributionTable([32,34,40,43,56,43,42,44,48,49,43,47,48,39,38,35,36,37,44])
# df, mean, variance, st_dev  = data.create_distribution_table()
# z, z_area_based_on_z_s_table = Z_score_value(mean, variance, st_dev).get_z_deets(45)
# z_left_area, z_l_area_percent,  z_right_area,  z_r_area_percent = NormalCurveAreas(mean, 45,z_area_based_on_z_s_table).get_z_left_right_deets()

# standard_x_norm = np.arange(-3,3,0.001)
# normal_curve = NormalCurveGraph(standard_x_norm, mean, st_dev)
# left_sncurve = normal_curve.draw_z_score(standard_x_norm<z,0,1,f'z < {z}')
# right_sncurve = normal_curve.draw_z_score(standard_x_norm>z,0,1,f'z > {z}')

# def sda_respect_to_mean(z):
#     if z < 0:
#         z0 = z
#         z1 = 0
#         title = f'0 < z < {z}'
    
#     elif z > 0:
#         z1 = z
#         z0 = 0
#         title = f'0 > z > {z}'

#     return z0, z1, title

# z0, z1, title = sda_respect_to_mean(z)
# between_sncurve = normal_curve.draw_z_score((z0 < standard_x_norm) & (standard_x_norm < z1),0,1,title=title)

# print(df)
# print("")
# print(mean, variance, st_dev)
# print("")
# print(z, z_area_based_on_z_s_table)
# print("")
# print(z_left_area, z_l_area_percent,  z_right_area,  z_r_area_percent)

# print(left_sncurve)