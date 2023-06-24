# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 21:01:29 2023

@author: ASUS VIVOBOOK
"""
import pandas as pd
import re
from datetime import datetime
import numpy as np

df = pd.read_excel('output_all_students_Train_v10.xlsx')



def prepare_data(df):
    df.columns = df.columns.str.strip()
    df=df.dropna(subset=['price'])
    df['price'] = df['price'].apply(lambda x: re.sub(r'\D', '', str(x)))
    df['price'] = pd.to_numeric(df['price'])
    df['Area'] = df['Area'].apply(lambda x: re.sub(r'\D', '', str(x)))
    df['Area'] = pd.to_numeric(df['Area'])
    
    
    
    columns_to_clean = ['Street', 'city_area', 'description']
    def remove_punctuation(text):
        return re.sub(r'[^\w\s]', '', str(text))
    for column in columns_to_clean:
        df[column] = df[column].apply(remove_punctuation)
     
        
    def clean_and_convert_to_num(df, column):       
        df[column] = df[column].astype(str)         
        df[column] = df[column].str.replace('[^\d.]', '', regex=True)  # Remove non-digit characters except '.'           
        df[column] = pd.to_numeric(df[column], errors='coerce')        
        df[column] = df[column].replace('', np.nan)
        df[column] = df[column].astype('float64')

    clean_and_convert_to_num(df , 'Area')
    clean_and_convert_to_num(df , 'room_number')
        
      
   
    
    df['floor'] = df['floor_out_of'].str.extract(r'(\d+)')
    df['floor'] = pd.to_numeric(df['floor']).astype('Int64')
    df['total_floors'] = df['floor_out_of'].str.extract(r'מתוך (\d+)')
    df['total_floors'] = pd.to_numeric(df['total_floors']).astype('Int64')
    
    
    date_pattern = r'\d{2}/\d{2}/\d{4}'
    df['entranceDate '] = df['entranceDate'].apply(lambda x: pd.to_datetime(x).date() if re.match(date_pattern, str(x)) else x)
    
    
    non_date_values = df.loc[~pd.to_datetime(df['entranceDate '], errors='coerce').notna(), 'entranceDate '].unique()
    katagory=[]
    # Print the unique non-date values
    for value in non_date_values:
        katagory.append(value)
    
    # Define the conditions for categorizing entrance date
    def categorize_entrance_date(date):
        if isinstance(date, datetime):
            time_diff = datetime.now().date() - date.date()
            if abs(time_diff.days) < 180:
                return 'less_than_6 months'
            elif abs(time_diff.days) < 365:
                return 'months_6_12'
            elif abs(time_diff.days) > 365:
                return 'above_year'
        elif isinstance(date, str):
            if 'גמיש' in date:
                return 'flexible'
            elif 'לא צויין' in date:
                return 'not_defined'
            elif 'מיידי' in date:
                return 'less_than_6 months'
    

    # Apply the categorization function to create the new column
    df['entranceDate'] = df['entranceDate'].apply(categorize_entrance_date)
    def convert_value(value):
        if value == True or 'יש' in str(value) or value == 'yes' or value == 'כן' or value==1:
          return 1
        elif 'לא נגיש' not in str(value) and 'נגיש' in str(value):
          return 1
        else:
          return 0

    columns_to_convert = ['hasElevator', 'hasParking', 'hasBars', 'hasStorage',
                            'hasAirCondition', 'hasBalcony', 'hasMamad', 'handicapFriendly']
    for column in columns_to_convert:
        df[column] = df[column].apply(convert_value)
        df[column].value_counts()

    

    df['room_number'] = df['room_number'].astype(str)
    
    df['room_number'] = df['room_number'].apply(lambda x: re.findall(r'\d+\.?\d*', x))
    df['room_number'] = df['room_number'].apply(lambda x: ''.join(x) if x else None)
    df['room_number'] = pd.to_numeric(df['room_number'], errors='coerce')
    
    return df

df  = prepare_data(df)
  