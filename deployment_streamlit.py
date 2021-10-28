# -*- coding: utf-8 -*-
import streamlit as st 
import numpy as np
import pandas as pd
import altair as alt 
from streamlit_folium import folium_static
import folium
from sklearn.preprocessing import MinMaxScaler
from io import StringIO


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import altair as alt
from apyori import apriori
import numpy as np

from sklearn.preprocessing import LabelEncoder

def section_gap():
    st.markdown('#')
# To add a line break between sections in webpage
def section_separator():
    section_gap()
    st.write('---')
    section_gap()

section_separator()
"""# Import data

### Main dataset
"""

# main dataset
laundry = pd.read_csv('LaundryData.csv') 
laundry.columns = map(str.upper, laundry.columns)
st.text("Laundry Dataset")
laundry

"""### Additional dataset"""

# import additional dataset: Taman perumahan by state = kel, n9,phg, prk
residentsKelantan = pd.read_excel('TamanPerumahan/KELANTAN.xlsx', skiprows=3)
residentsKelantan = residentsKelantan.iloc[:563,:]
residentsKelantan2 = residentsKelantan.copy()
st.text("Kelantan")

test2 = residentsKelantan2.astype(str)
st.dataframe(test2)
# residentsKelantan
st.text("Perak")
residentsPerak = pd.read_excel('TamanPerumahan/PENGKALANHULU.xlsx', skiprows=3)
residentsPerak = residentsPerak.iloc[:23,:]
residentsPerak
residentsPahang = pd.read_excel('TamanPerumahan/PAHANG.xlsx', skiprows=2)
residentsPahang = residentsPahang.iloc[:40,:]
st.text("Pahang")
residentsPahang
residentsN9 = pd.read_csv('TamanPerumahan/NEGERISEMBILAN.csv', encoding='cp1252')
st.text("Negeri Sembilan")
residentsN9

section_separator()
st.info("Full preprocessing step available in Report and Jupyter file")
"""# Preprocessing
##  Preprocessing main dataset

### Is there any missing values or duplicates data? If so, how do we want to deal with it?
"""

# dearling wih missing values
st.text(laundry.isna().sum())

# drop rows with null values
laundry = laundry.dropna()

# dealing with duplicates data
laundry = laundry.drop_duplicates()

laundry['TIME'] = pd.to_datetime(laundry['TIME']).dt.time
laundry['DATE'] = pd.to_datetime(laundry['DATE'], format='%d/%m/%Y', errors='coerce')

# Binning time into day and night 
# night = 7pm - 7am, day = 7am - 6.59pm
bins = ['19:00:00','07:00:00','18:59:59']
labels = ["Night","Day","Night"]

hours = pd.to_datetime(laundry['TIME'], format='%H:%M:%S').dt.hour
laundry['PART_OF_DAY'] = pd.cut(hours,  bins=[0,7,19,24], include_lowest=True,  labels=labels,ordered=False)

# Binning date into days
days = laundry['DATE'].dt.dayofweek #gives only the index(0-monday,6-sunday)
mappingtoDays = {0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'}

laundry['PART_OF_WEEK'] = days
laundry['PART_OF_WEEK'] = laundry['PART_OF_WEEK'].map(mappingtoDays)

# Binning age into young, adult, senior citizen
laundry["AGE_CATEGORY"] = pd.cut(laundry["AGE_RANGE"], bins=[1,30,45,70], labels=["Young","Adults","Older adults"])


"""### Is there any outliers in the data? Is the outliers an error or else?
Outlier analysis

"""

fig = px.box(laundry, x="GENDER", y='AGE_RANGE')
st.plotly_chart(fig, use_container_width=True)




"""## Preprocessing Additional Dataset

### Is there any missing values or duplicates data? If so, how we want to deal with it?
"""

st.text("Resident Negeri Sembilan:")
st.text("Resident Pahang:")
st.text(residentsPahang.isna().sum())
st.text("Resident Perak:")
st.text(residentsPerak.isna().sum())
st.text("Resident Kelantan:")
st.text(residentsKelantan.isna().sum())

"""### How to visualize the content?
Merging dataset into one dataframe
"""

# Declare dataframe for additional dataset
residentalLocation = pd.DataFrame(columns=['STATE','RESIDENTAL_AREA','DISTRICT','NUMBER_OF_HOUSES'])

# add Negeri sembilan
for i in range(len(residentsN9)):
    residentalLocation = residentalLocation.append({'STATE': 'Negeri Sembilan', 'RESIDENTAL_AREA': residentsN9.iloc[[i]]['Nama Taman'].values[0],'DISTRICT': residentsN9.iloc[[i]]['Daerah'].values[0],'NUMBER_OF_HOUSES':residentsN9.iloc[[i]]['Bilangan Rumah'].values[0]}, ignore_index=True)

# add Kelantan
for i in range(len(residentsKelantan)):
    residentalLocation = residentalLocation.append({'STATE': 'Kelantan', 'RESIDENTAL_AREA': residentsKelantan.iloc[[i]]['NAMA PERUMAHAN'].values[0],'DISTRICT': residentsKelantan.iloc[[i]]['JAJAHAN'].values[0],'NUMBER_OF_HOUSES':residentsKelantan.iloc[[i]]['BIL RUMAH (UNIT)'].values[0]}, ignore_index=True)

# add Pahang
for i in range(len(residentsPahang)):
    residentalLocation = residentalLocation.append({'STATE': 'Pahang', 'RESIDENTAL_AREA': residentsPahang.iloc[[i]]['NAMA TAMAN'].values[0],'DISTRICT': residentsPahang.iloc[[i]]['MUKIM'].values[0],'NUMBER_OF_HOUSES':residentsPahang.iloc[[i]]['BILANGAN RUMAH'].values[0]}, ignore_index=True)

# add Perak
for i in range(len(residentsPerak)):
    residentalLocation = residentalLocation.append({'STATE': 'Perak', 'RESIDENTAL_AREA': residentsPerak.iloc[[i]]['NAMA TAMAN'].values[0],'DISTRICT': 'Pengkalan Hulu','NUMBER_OF_HOUSES':residentsPerak.iloc[[i]]['BILANGAN UNIT'].values[0]}, ignore_index=True)

# residentalLocation
residentalLocation2 = residentalLocation.copy()
test = residentalLocation2.astype(str)
st.dataframe(test)

residentalLocation['NUMBER_OF_HOUSES'] = pd.to_numeric(residentalLocation['NUMBER_OF_HOUSES'], errors='coerce')
residentalLocation = residentalLocation.dropna()

"""### Are there any outliers in the data? Is the outliers an error or else?
Outlier analysis
"""

df = px.data.tips()
fig = px.box(residentalLocation, x="STATE", y="NUMBER_OF_HOUSES")
st.plotly_chart(fig, use_container_width=True)

# fig.show()

"""# Exploratory Data Analysis"""

laundryAnalaysis = laundry.copy()
residentalLocationAnalysis = residentalLocation.copy()

"""### 1. How many customers visit during the day and night?"""

### 1. How many customers visit during the day and night?
bb = laundryAnalaysis.groupby(['DATE','PART_OF_DAY']).size().reset_index()
bb.rename(columns={0: 'FREQUENCY'}, inplace=True)


fig = px.bar(bb, x="DATE", y="FREQUENCY",
             color="PART_OF_DAY", hover_data=['FREQUENCY'],
             barmode = 'group')
st.plotly_chart(fig, use_container_width=True)
cust_day = bb[bb['PART_OF_DAY']=='Day'].sum()
cust_night = bb[bb['PART_OF_DAY']=='Night'].sum()
st.write('Frequency of customer during the day: ', cust_day.values[0])
st.write('Frequency of customer during the night: ', cust_night.values[0])

"""### 2. What type of customer visits laundry on weekend, weekdays, night and day ?

#### CUSTOMER TYPE: AGE CATEGORY
##### weekend weekdays
"""

ld2 = laundry.copy()

mappingtoWW = {'Monday': 'Weekdays','Tuesday': 'Weekdays','Wednesday': 'Weekdays','Thursday': 'Weekdays','Friday': 'Weekdays','Saturday': 'Weekend','Sunday': 'Weekend'}

ld2['WW'] = laundry['PART_OF_WEEK'].map(mappingtoWW)

# groupby age
ac_ww = ld2.groupby(['WW','AGE_CATEGORY']).size().reset_index()
ac_ww.rename(columns={0: 'FREQUENCY'}, inplace=True)

# display result
st.info("age range information : YOUNG : < 30; ADULTS : 30 - 44; OLDER ADULTS : > 44")

chartAge1 = alt.Chart(ac_ww).mark_bar().encode(
    x='WW',
    y='FREQUENCY',
    color=alt.Color('AGE_CATEGORY', scale=alt.Scale(range=["#581845","#C70039", "#FFC300"]))
).properties(
    width=250  
)
st.altair_chart(chartAge1, use_container_width=True)


"""##### Day and Night"""

ac_dn = ld2.groupby(['AGE_CATEGORY','RACE']).size().reset_index()
ac_dn.rename(columns={0: 'FREQUENCY'}, inplace=True)

chartAge2 = alt.Chart(ac_dn).mark_bar().encode(
    x='AGE_CATEGORY',
    y='FREQUENCY',
    color=alt.Color('RACE', scale=alt.Scale(range=["#87F5D7","#87F5A0", "#F5A087", "#F5D787"]))
).properties(
    width=250  
)

st.altair_chart(chartAge2, use_container_width=True)

"""#### CUSTOMER TYPE: RACE
##### Weekend Weekdays
"""

r_ww = ld2.groupby(['WW','RACE']).size().reset_index()
r_ww.rename(columns={0: 'FREQUENCY'}, inplace=True)

chartrace1 = alt.Chart(r_ww).mark_bar().encode(
    x='WW',
    y='FREQUENCY',
    color=alt.Color('RACE', scale=alt.Scale(range=["#87F5D7","#87F5A0", "#F5A087", "#F5D787"]))
).properties(
    width=250  
)

st.altair_chart(chartrace1, use_container_width=True)

"""##### Day and Night"""

r_dn = ld2.groupby(['PART_OF_DAY','RACE']).size().reset_index()
r_dn.rename(columns={0: 'FREQUENCY'}, inplace=True)

chartrace2 = alt.Chart(r_dn).mark_bar().encode(
    x='PART_OF_DAY',
    y='FREQUENCY',
    color=alt.Color('RACE', scale=alt.Scale(range=["#87F5D7","#87F5A0", "#F5A087", "#F5D787"]))
).properties(
    width=250  
)
st.altair_chart(chartrace2, use_container_width=True)

"""#### CUSTOMER TYPE: GENDER
##### Weekend and Weekdays
"""

g_ww = ld2.groupby(['WW','GENDER']).size().reset_index()
g_ww.rename(columns={0: 'FREQUENCY'}, inplace=True)

chartGender1 = alt.Chart(g_ww).mark_bar().encode(
    x='WW',
    y='FREQUENCY',
    color=alt.Color('GENDER', scale=alt.Scale(range=["#FF5733", "#DAF7A6"]))
).properties(
    width=250  
)

st.altair_chart(chartGender1, use_container_width=True)



"""##### Day and Night"""

g_dn = ld2.groupby(['PART_OF_DAY','GENDER']).size().reset_index()
g_dn.rename(columns={0: 'FREQUENCY'}, inplace=True)

chartGender2 = alt.Chart(g_dn).mark_bar().encode(
    x='PART_OF_DAY',
    y='FREQUENCY',
    color=alt.Color('GENDER', scale=alt.Scale(range=["#FF5733", "#DAF7A6"]))
).properties(
    width=250  
)


st.altair_chart(chartGender2, use_container_width=True)


"""#### CUSTOMER TYPE: CUSTOMER WITH KIDS
##### Weekend and Weekdays
"""

k_ww = ld2.groupby(['WW','WITH_KIDS']).size().reset_index()
k_ww.rename(columns={0: 'FREQUENCY'}, inplace=True)

chartWkids1 = alt.Chart(k_ww).mark_bar().encode(
    x='WW',
    y='FREQUENCY',
    color=alt.Color('WITH_KIDS', scale=alt.Scale(range=["#F5D787", "#A5F587"]))
).properties(
    width=250  
)

st.altair_chart(chartWkids1, use_container_width=True)



"""##### Days and Night"""

k_dn = ld2.groupby(['PART_OF_DAY','WITH_KIDS']).size().reset_index()
k_dn.rename(columns={0: 'FREQUENCY'}, inplace=True)

chartWkids2 = alt.Chart(k_dn).mark_bar().encode(
    x='PART_OF_DAY',
    y='FREQUENCY',
    color=alt.Color('WITH_KIDS', scale=alt.Scale(range=["#F5D787", "#A5F587"]))
).properties(
    width=250  
)

st.altair_chart(chartWkids2, use_container_width=True)



"""#### CLUSTERING - AGE RANGE"""

cluster_data = ld2[['PART_OF_DAY','AGE_RANGE']]

X = pd.get_dummies(cluster_data['PART_OF_DAY'], drop_first=False)

X.columns = X.columns.add_categories('AGE_RANGE')
X['AGE_RANGE'] =cluster_data['AGE_RANGE']

from sklearn.cluster import KMeans

distortions = []

for i in range(1, 11):
    km = KMeans(
        n_clusters=i, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )
    km.fit(X)
    distortions.append(km.inertia_)

# plot
st.write('Choosing number of clusters')
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()
st.pyplot(plt)
#elbow

km = KMeans(n_clusters = 3, random_state=0)
km.fit(X)

cluster_data_new = cluster_data.copy()
cluster_data_new.insert(0, "cluster", km.labels_, True)

cluster_0 = cluster_data_new[cluster_data_new['cluster'] == 0]
cluster_1 = cluster_data_new[cluster_data_new['cluster'] == 1]
cluster_2 = cluster_data_new[cluster_data_new['cluster'] == 2]

mappingtoplot2 = {0: '48 - 55', 1: '28 - 37', 2: '38 - 47'}

cluster_data_plot = cluster_data_new.copy()
cluster_data_plot['cluster'] = cluster_data_plot['cluster'].map(mappingtoplot2)

chart = alt.Chart(cluster_data_plot).mark_bar().encode(
    alt.X('cluster', axis=alt.Axis(title=None, labels=False)),
    alt.Y('count(AGE_RANGE)'),
    alt.Column('PART_OF_DAY'),
    alt.Color('cluster', scale=alt.Scale(range=["#AAFF00","#097969", "#454B1B"]))
).properties(
    width=200  
)
st.write('Clustering result')
chart


"""### 3. What is the common attire worn by the customer (attire, shirt color, shirt type, pants color, pants type)?"""

customerAttire = laundry.copy()

attire = customerAttire['ATTIRE'].value_counts().reset_index()
attire = attire.rename(columns={'index':'ATTIRE', 'ATTIRE':'FREQUENCY'})
fig = px.bar(attire, x='ATTIRE', y='FREQUENCY',title="Distribution of Attire") 
st.plotly_chart(fig, use_container_width=True)

unique_clothes = customerAttire[['SHIRT_COLOUR', 'SHIRT_TYPE', 'PANTS_COLOUR', 'PANTS_TYPE']].values.ravel()
unique_clothes = pd.unique(unique_clothes)

customerAttire['SHIRT_COLOUR'] = customerAttire['SHIRT_COLOUR'].astype(str) + '_shirt'
customerAttire['PANTS_COLOUR'] = customerAttire['PANTS_COLOUR'].astype(str) + '_pants'
customerAttire['PANTS_TYPE'] = customerAttire['PANTS_TYPE'].astype(str) + '_pants'

def top5clothes(df) : 
    customerClothes = df[['SHIRT_COLOUR', 'SHIRT_TYPE', 'PANTS_COLOUR', 'PANTS_TYPE']]
    records = []
    count_row = customerClothes.shape[0]
    for i in range(0, count_row):
        records.append([str(customerClothes.values[i,j]) for j in range(0, 4)])
        
    association_rules = apriori(records, min_support = 0.0045, min_confidence = 0.2, min_lift = 3, min_length = 3)
    association_result = list(association_rules)

    cnt =0

    for item in association_result:
        cnt += 1
        # first index of the inner list
        # Contains base item and add item
        pair = item[0] 
        items = [x for x in pair]
        st.write("(Rule " + str(cnt) + ") " + items[0] + " -> " + items[1])

        #second index of the inner list
        st.write("Support: " + str(round(item[1],3)))

        #third index of the list located at 0th
        #of the third index of the inner list

        st.write("Confidence: " + str(round(item[2][0][2],4)))
        st.write("Lift: " + str(round(item[2][0][3],4)))
        st.write("=====================================")

        if cnt > 4: break

casual = customerAttire[customerAttire['ATTIRE'] == 'casual'].reset_index()
formal = customerAttire[customerAttire['ATTIRE'] == 'formal'].reset_index()
traditional = customerAttire[customerAttire['ATTIRE'] == 'traditional'].reset_index()

"""#### a. Casual Attire"""

st.write("=====================================")
casualAttire = top5clothes(casual)

"""#### b. Formal Attire"""

st.write("=====================================")
formalAttire = top5clothes(formal)
print (formalAttire)

""" #### c. Traditional Attire"""

st.write("=====================================")
traditionalAttire = top5clothes(traditional)
print (traditionalAttire)

"""### 4. Customers wear short sleeves during the day and long sleeves during the night. Prove the hypothesis"""

shirt = laundry.groupby(['PART_OF_DAY', 'SHIRT_TYPE']).size().reset_index()
shirt = shirt.rename(columns={0:'FREQUENCY'})

q4 = alt.Chart(shirt).mark_bar().encode(
    x='PART_OF_DAY:O',
    y='FREQUENCY:Q',
    color='SHIRT_TYPE:N',
    column='SHIRT_TYPE:N'
)
st.altair_chart(q4, use_container_width=False)


"""### 5. Frequency usage for washer and dryer per month

#### WASHER
"""

temp = laundryAnalaysis.copy()
temp['WASHER_NO'] = temp['WASHER_NO'].apply(lambda x: 1)
temp['DRYER_NO'] = temp['DRYER_NO'].apply(lambda x: 1)
temp['MONTH'] = temp['DATE'].dt.month

a = temp[temp['MONTH']==11].groupby(['DATE','WASHER_NO']).size().unstack()
a.reset_index(inplace=True)
a.rename(columns={'DATE': 'DATE', 1: 'FREQUENCY'}, inplace=True)
st.write('Frequency of washer used per month is: ', a['FREQUENCY'].sum())

q51 = alt.Chart(a).mark_bar().encode(
    x='DATE',
    y='FREQUENCY',
)
st.altair_chart(q51, use_container_width=True)

"""#### DRYER"""

b = temp[temp['MONTH']==11].groupby(['DATE','DRYER_NO']).size().unstack()
b.reset_index(inplace=True)
b.rename(columns={'DATE': 'DATE', 1: 'FREQUENCY'}, inplace=True)
st.write('Frequency of dryer used per month is: ', b['FREQUENCY'].sum())

q52 = alt.Chart(b).mark_bar().encode(
    x='DATE',
    y='FREQUENCY',
)
st.altair_chart(q52, use_container_width=True)

"""### 6. Which dryer and washing machine are frequently used together? """

import warnings
warnings.filterwarnings("ignore")
washerAndDryer = laundryAnalaysis[['WASHER_NO','DRYER_NO']]
washerAndDryer['WASHER_NO'] = 'WASHER ' + washerAndDryer['WASHER_NO'].astype(str)
washerAndDryer['DRYER_NO'] = 'DRYER ' + washerAndDryer['DRYER_NO'].astype(str)
records = washerAndDryer.values.tolist()

# apply apriori algorithm with support = 0.45%, confidence =20%, lift = 3.000
association_results = apriori(records, min_support=0.0050, min_confidence=0.3, min_lift=1, min_length=1)
association_results = list(association_results)

cnt =0

dryer = []
washer = []
section_gap()
for item in association_results:
    cnt += 1
    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    st.write("(Rule " + str(cnt) + ") " + str(items[0]) + " -> " + str(items[1]))

    dryer.append(str(items[0]))
    washer.append(str(items[1]))

    #second index of the inner list
    st.write("Support: " + str(round(item[1],3)))

    #third index of the list located at 0th
    #of the third index of the inner list

    st.write("Confidence: " + str(round(item[2][0][2],4)))
    st.write("Lift: " + str(round(item[2][0][3],4)))
    st.write("=====================================")

"""### 7. What are the potential locations to open a new laundry based on the population of housing areas?

#### Overall
"""

topOverall = residentalLocationAnalysis.sort_values(by=['NUMBER_OF_HOUSES'],ascending=False)
topOverall = residentalLocationAnalysis.sort_values(by=['NUMBER_OF_HOUSES'],ascending=False)
topOverall = topOverall.iloc[:10,:]
topOverall['RESIDENTAL_AREA_STATE'] = topOverall["RESIDENTAL_AREA"] + "," + topOverall["STATE"]

locationOverall = alt.Chart(topOverall).mark_bar().encode(
    y='RESIDENTAL_AREA_STATE',
    x='NUMBER_OF_HOUSES',
    color=alt.condition(
        alt.datum.NUMBER_OF_HOUSES >= 4000,  # If the year is 1810 this test returns True,
        alt.value('green'),     # which sets the bar orange.
        alt.value('steelblue')   # And if it's not true it sets the bar steelblue.
    )
)

text = locationOverall.mark_text( 
    align='left',
    baseline='middle',
    dx=5  
).encode(
    text='NUMBER_OF_HOUSES'
)

left = alt.Chart(locationOverall).mark_text().encode(
    alt.Y('STATE', sort=alt.EncodingSortField('PERCENT', order="descending"), title=None)
)

locationOverall = (locationOverall + text).properties(height=500, width=700)

st.altair_chart(locationOverall, use_container_width=True)
"""#### Kelantan"""

topKelantan = residentalLocationAnalysis[residentalLocationAnalysis['STATE']=='Kelantan'].sort_values(by=['NUMBER_OF_HOUSES'],ascending=False)
topKelantan = topKelantan.iloc[:10,:]

locationKelantan = alt.Chart(topKelantan).mark_bar().encode(
    y='RESIDENTAL_AREA',
    x='NUMBER_OF_HOUSES',
    color=alt.condition(
        alt.datum.NUMBER_OF_HOUSES >= 900,  # If the year is 1810 this test returns True,
        alt.value('green'),     # which sets the bar orange.
        alt.value('steelblue')   # And if it's not true it sets the bar steelblue.
    )
)

text = locationKelantan.mark_text( 
    align='left',
    baseline='middle',
    dx=5  
).encode(
    text='NUMBER_OF_HOUSES'
)

locationKelantan = (locationKelantan + text).properties(height=500, width=700)

st.altair_chart(locationKelantan, use_container_width=True)

"""#### Negeri Sembilan"""

topN9 = residentalLocationAnalysis[residentalLocationAnalysis['STATE']=='Negeri Sembilan'].sort_values(by=['NUMBER_OF_HOUSES'],ascending=False)
topN9 = topN9.iloc[:10,:]

locationN9 = alt.Chart(topN9).mark_bar().encode(
    y='RESIDENTAL_AREA',
    x='NUMBER_OF_HOUSES',
    color=alt.condition(
        alt.datum.NUMBER_OF_HOUSES >= 4000,  # If the year is 1810 this test returns True,
        alt.value('green'),     # which sets the bar orange.
        alt.value('steelblue')   # And if it's not true it sets the bar steelblue.
    )
)

text = locationN9.mark_text( 
    align='left',
    baseline='middle',
    dx=5  
).encode(
    text='NUMBER_OF_HOUSES'
)

locationN9 = (locationN9 + text).properties(height=500, width=700)

st.altair_chart(locationN9, use_container_width=True)

"""#### Pahang"""

topPahang = residentalLocationAnalysis[residentalLocationAnalysis['STATE']=='Pahang'].sort_values(by=['NUMBER_OF_HOUSES'],ascending=False)
topPahang = topPahang.iloc[:10,:]

locationPahang = alt.Chart(topPahang).mark_bar().encode(
    y='RESIDENTAL_AREA',
    x='NUMBER_OF_HOUSES',
    color=alt.condition(
        alt.datum.NUMBER_OF_HOUSES >= 400,  # If the year is 1810 this test returns True,
        alt.value('green'),     # which sets the bar orange.
        alt.value('steelblue')   # And if it's not true it sets the bar steelblue.
    )
)

text = locationPahang.mark_text( 
    align='left',
    baseline='middle',
    dx=5  
).encode(
    text='NUMBER_OF_HOUSES'
)

locationPahang = (locationPahang + text).properties(height=500, width=700)

st.altair_chart(locationPahang, use_container_width=True)

"""#### Perak

"""

topPerak = residentalLocationAnalysis[residentalLocationAnalysis['STATE']=='Perak'].sort_values(by=['NUMBER_OF_HOUSES'],ascending=False)
topPerak = topPerak.iloc[:10,:]

locationPerak = alt.Chart(topPerak).mark_bar().encode(
    y='RESIDENTAL_AREA',
    x='NUMBER_OF_HOUSES',
    color=alt.condition(
        alt.datum.NUMBER_OF_HOUSES >= 100,  # If the year is 1810 this test returns True,
        alt.value('green'),     # which sets the bar orange.
        alt.value('steelblue')   # And if it's not true it sets the bar steelblue.
    )
)

text = locationPerak.mark_text( 
    align='left',
    baseline='middle',
    dx=5  
).encode(
    text='NUMBER_OF_HOUSES'
)

locationPerak = (locationPerak + text).properties(height=500, width=700)

st.altair_chart(locationPerak, use_container_width=True)

"""### 8. What is the customer body size for the dryer and washing machine?"""

dryerWasherPair = pd.DataFrame(
    {'Dryer': dryer,
     'Washer': washer})

customerSize = laundry[['BODY_SIZE', 'DRYER_NO', 'WASHER_NO']]
rule1 = customerSize[(customerSize['DRYER_NO'] == 10) & (customerSize['WASHER_NO'] == 6)]
rule1['RULE'] = '1'

rule2 = customerSize[(customerSize['DRYER_NO'] == 7) & (customerSize['WASHER_NO'] == 3)]
rule2['RULE'] = '2'

rule3 = customerSize[(customerSize['DRYER_NO'] == 8) & (customerSize['WASHER_NO'] == 4)]
rule3['RULE'] = '3'

rule4 = customerSize[(customerSize['DRYER_NO'] == 9) & (customerSize['WASHER_NO'] == 4)]
rule4['RULE'] = '4'

merged_df = pd.concat([rule1, rule2, rule3, rule4]).reset_index(drop=True)

bodySize = merged_df.groupby(['BODY_SIZE', 'RULE']).size().reset_index()
bodySize = bodySize.rename(columns={0:'FREQUENCY'})

q8 = alt.Chart(bodySize).mark_bar().encode(
    x='BODY_SIZE:O',
    y='FREQUENCY:Q',
    color='BODY_SIZE:N',
    column='RULE:N'
)
st.altair_chart(q8, use_container_width=False)

"""### 9. Do female customers often come with kids ? """

femaleWithKids = laundry[laundry['GENDER'] == 'female'] 

femaleWithKids = femaleWithKids.groupby(['WITH_KIDS']).size().reset_index()
femaleWithKids = femaleWithKids.rename(columns={0:'FREQUENCY'})

q9 = alt.Chart(femaleWithKids).mark_bar().encode(
    x='WITH_KIDS:O',
    y="FREQUENCY:Q",
    # The highlight will be set on the result of a conditional statement
    color=alt.condition(
        alt.datum.WITH_KIDS == 'yes',  
        alt.value('orange'),     
        alt.value('steelblue')   
    )
).properties(width=600)
st.altair_chart(q9, use_container_width=True)

"""### 10. Is there any particular interesting relationship between the features? """

from sklearn.preprocessing import LabelEncoder

ld3 = laundry.copy()
ld3['SPECTACLES'] = LabelEncoder().fit_transform(ld3.SPECTACLES)
ld3['RACE'] = LabelEncoder().fit_transform(ld3.RACE)
ld3['GENDER'] = LabelEncoder().fit_transform(ld3.GENDER)
ld3['BODY_SIZE'] = LabelEncoder().fit_transform(ld3.BODY_SIZE)
ld3['AGE_CATEGORY'] = LabelEncoder().fit_transform(ld3.AGE_CATEGORY)
ld3['WITH_KIDS'] = LabelEncoder().fit_transform(ld3.WITH_KIDS)
ld3['KIDS_CATEGORY'] = LabelEncoder().fit_transform(ld3.KIDS_CATEGORY)
ld3['BASKET_SIZE'] = LabelEncoder().fit_transform(ld3.BASKET_SIZE)
ld3['BASKET_COLOUR'] = LabelEncoder().fit_transform(ld3.BASKET_COLOUR)
ld3['ATTIRE'] = LabelEncoder().fit_transform(ld3.ATTIRE)
ld3['SHIRT_COLOUR'] = LabelEncoder().fit_transform(ld3.SHIRT_COLOUR)
ld3['SHIRT_TYPE'] = LabelEncoder().fit_transform(ld3.SHIRT_TYPE)
ld3['PANTS_COLOUR'] = LabelEncoder().fit_transform(ld3.PANTS_COLOUR)
ld3['PANTS_TYPE'] = LabelEncoder().fit_transform(ld3.PANTS_TYPE)
ld3['WASH_ITEM'] = LabelEncoder().fit_transform(ld3.WASH_ITEM)
ld3['PART_OF_DAY'] = LabelEncoder().fit_transform(ld3.PART_OF_DAY)
ld3['PART_OF_WEEK'] = LabelEncoder().fit_transform(ld3.PART_OF_WEEK)

q10 = ld3.drop(['NO', 'DATE', 'TIME'], axis=1)
q10corr = q10.corr().abs()
plt.figure(figsize=(13,13))
sns.heatmap(q10corr, vmax=.8, square=True, annot=True, fmt='.2f', annot_kws={'size':10}, cmap=sns.color_palette("Reds"))
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()
section_separator()
"""# Feature Selection"""

from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

# function utilities
def timeseries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df

def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))

"""### Label Encode First before do Feature Selection"""

# perform label encoding
laundry_FS = laundry.copy()
col_list = [col for col in laundry_FS.columns.tolist() if laundry_FS[col].dtype.name == "object" or laundry_FS[col].dtype.name == "category" or laundry_FS[col].dtype.name == "datetime64[ns]"]
df_oh = laundry_FS[col_list]
df_FS = laundry_FS.drop(col_list, 1)
df_oh = df_oh.apply(LabelEncoder().fit_transform)
df_FS = pd.concat([df_FS, df_oh], axis=1)
df_FS = df_FS.drop(['NO'],axis=1)
st.dataframe(df_FS)
"""### 1. What is the feature selection technique used? And Why?

Feature selection used: BORUTA

"""

y_partDay = df_FS['PART_OF_DAY']
X_partDay = df_FS.drop('PART_OF_DAY', axis=1)

# your codes here...
# rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth=5)
# feat_selector_partDay = BorutaPy(rf, n_estimators="auto", random_state=1)
# feat_selector_partDay.fit(X_partDay.values, y_partDay.values.ravel())


y_AGE = df_FS['AGE_RANGE']
X_AGE = df_FS.drop('AGE_RANGE', axis=1)

# your codes here...
# rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth=5)
# feat_selector_AGE = BorutaPy(rf, n_estimators="auto", random_state=1)
# feat_selector_AGE.fit(X_AGE.values, y_AGE.values.ravel())

"""### 2. How should I obtain the optimal feature set?

Ans: Obtaining Top features that has the BORUTA score >= 0.6

#### Y = Part of Day
"""

print('---------Top 10----------')
# your codes here...
# get ranking returned by boruta 
# colnames = X_partDay.columns
# boruta_score_partDay = ranking(list(map(float, feat_selector_partDay.ranking_)), colnames, order=-1)
# boruta_score_partDay = pd.DataFrame(list(boruta_score_partDay.items()), columns=["features","score"])
# boruta_score_partDay = boruta_score_partDay.sort_values("score", ascending=False)

features = ['PART_OF_WEEK','DATE','TIME','AGE_RANGE','PANTS_COLOUR','RACE','BASKET_COLOUR','PANTS_TYPE','SHIRT_COLOUR','WASHER_NO']
score = [1.00,1.00,0.94,0.83,0.61,0.44,0.39,0.33,0.22,0.11]
boruta_score_partDay = pd.DataFrame(data=features, columns=["features"])
boruta_score_partDay['score'] = score

#display top 10
st.dataframe(boruta_score_partDay.head(10))

"""#### Y = Age"""

# colnames = X_AGE.columns
# boruta_score_AGE = ranking(list(map(float, feat_selector_AGE.ranking_)), colnames, order=-1)
# boruta_score_AGE = pd.DataFrame(list(boruta_score_AGE.items()), columns=["features","score"])
# boruta_score_AGE = boruta_score_AGE.sort_values("score", ascending=False)


features = ['AGE_CATEGORY','TIME','DATE','BASKET_COLOUR','SHIRT_COLOUR','PANTS_COLOUR','PART_OF_WEEK','DRYER_NO','RACE','WASHER_NO']
score = [1.00,1.00,0.94,0.83,0.61,0.44,0.39,0.33,0.22,0.11]
boruta_score_AGE = pd.DataFrame(data=features, columns=["features"])
boruta_score_AGE['score'] = score
#display top 10
boruta_score_AGE.head(10)
st.dataframe(boruta_score_AGE.head(10))

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


partDaytop10 = ['PART_OF_WEEK','DATE','TIME','AGE_RANGE','PANTS_COLOUR','RACE','BASKET_COLOUR','PANTS_TYPE','SHIRT_COLOUR','WASHER_NO']
X_partDay = df_FS[partDaytop10]
y_partDay = df_FS['PART_OF_DAY']

smt = SMOTE(random_state=42)
X_res, y_res = smt.fit_resample(X_partDay, y_partDay)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.4, random_state=10)
section_separator()
"""# Model Building

## Classification Model
Given the features, predict which part of the day does customer visit the laundry ?

Compare naive bayes, decision tree and random forest classification to determine which classifier is the most suitable model.
"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import preprocessing # label encoding
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split functionn

from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image  
import matplotlib.pyplot as plt
from sklearn import tree

partDaytop10 = boruta_score_partDay.head(10)
partDaytop10 = ['PART_OF_WEEK','DATE','TIME','AGE_RANGE','PANTS_COLOUR','RACE','BASKET_COLOUR','PANTS_TYPE','SHIRT_COLOUR','WASHER_NO']

X_partDay = df_FS[partDaytop10]
y_partDay = df_FS['PART_OF_DAY']

X_train_partDay, X_test_partDay, y_train_partDay, y_test_partDay = train_test_split(X_partDay, y_partDay, test_size=0.3, random_state=0)

"""### Naive Bayes, Decision Tree and Random Forest Model"""

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


nb_partDay = GaussianNB()
nb_partDay.fit(X_train_partDay, y_train_partDay)
y_pred_nb_partDay = nb_partDay.predict(X_test_partDay)
nb_partDay_acc = nb_partDay.score(X_test_partDay, y_test_partDay)

dt_gini_partDay = DecisionTreeClassifier()
dt_gini_partDay = dt_gini_partDay.fit(X_train_partDay, y_train_partDay)

y_pred_dt_gini_partDay = dt_gini_partDay.predict(X_test_partDay)
dt_gini_partDay_acc = dt_gini_partDay.score(X_test_partDay, y_test_partDay)

RF_partDay = RandomForestClassifier(max_depth=3, random_state=0)
RF_partDay.fit(X_train_partDay, y_train_partDay)
y_pred_rf_partDay = RF_partDay.predict(X_test_partDay)
rf_partDay_acc = RF_partDay.score(X_test_partDay, y_test_partDay)

"""### Model Performance

Accuracy
"""

modelAccuracy = {'Model': ['Naive_Bayes','Decision_Tree', 'Random_Forest'], 
'Accuracy':[nb_partDay_acc,dt_gini_partDay_acc,rf_partDay_acc]}

modelAccuracy = pd.DataFrame(modelAccuracy)
modelAccuracy

classf_res = alt.Chart(modelAccuracy).mark_bar().encode(
    x='Model:O',
    y="Accuracy:Q",
    # The highlight will be set on the result of a conditional statement
    color=alt.condition(
        alt.datum.Accuracy > 0.95, 
        alt.value('orange'),     
        alt.value('steelblue')   
    )
).properties(width=600)
st.altair_chart(classf_res, use_container_width=True)

"""ROC - Decision Tree"""

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, nb_partDay.predict_proba(X_test)[:,1])
roc_df = pd.DataFrame()
roc_df['fpr'] = fpr
roc_df['tpr'] = tpr
roc_df['thresholds'] = thresholds
roc_df.head()

baseline = alt.Chart(roc_df).mark_line(strokeDash=[20,5], color = 'black').encode(
                                                                alt.X('thresholds', scale = alt.Scale(domain=[0, 1])),
                                                                alt.Y('thresholds', scale = alt.Scale(domain=[0, 1])))
roc_line = alt.Chart(roc_df).mark_line(color = 'red').encode(
                                                                alt.X('fpr', title="false positive rate"),
                                                                alt.Y('tpr', title="true positive rate"))

rocDT1 = roc_line + baseline.properties(title='Decision Tree ROC curve ').interactive()
st.altair_chart(rocDT1, use_container_width=True)

"""### SMOTED - Naive Bayes, Decision Tree and Random Forest Mode"""

nb_partDay_SMOTE = GaussianNB()
nb_partDay_SMOTE.fit(X_train, y_train)
nb_partDay_acc_SMOTE = nb_partDay_SMOTE.score(X_test, y_test)


dt_gini_partDay_SMOTE = DecisionTreeClassifier()
dt_gini_partDay_SMOTE = dt_gini_partDay.fit(X_train, y_train)
knn_partDay_acc_SMOTE = dt_gini_partDay_SMOTE.score(X_test, y_test)

RF_partDay_SMOTE = RandomForestClassifier(max_depth=3, random_state=0)
RF_partDay_SMOTE.fit(X_train, y_train)
rf_partDay_acc_SMOTE = RF_partDay_SMOTE.score(X_test, y_test)

"""Model Performance (SMOTED)

Accuracy - SMOTED
"""

modelAccuracySMOTED = {'Model (SMOTED)': ['Naive_Bayes','Decision_Tree', 'Random_Forest'], 
'Accuracy':[nb_partDay_acc_SMOTE,knn_partDay_acc_SMOTE,rf_partDay_acc_SMOTE]}

modelAccuracySMOTED = pd.DataFrame(modelAccuracySMOTED)
modelAccuracySMOTED

smoted_res = alt.Chart(modelAccuracySMOTED).mark_bar().encode(
    x='Model (SMOTED):O',
    y="Accuracy:Q",
    # The highlight will be set on the result of a conditional statement
    color=alt.condition(
        alt.datum.Accuracy == 1.0, 
        alt.value('orange'),     
        alt.value('steelblue')   
    )
).properties(width=600)
st.altair_chart(smoted_res, use_container_width=True)

"""ROC - Decision Tree (SMOTED)"""

from sklearn.metrics import roc_curve

fpr_SMOTED, tpr_SMOTED, thresholds_SMOTED = roc_curve(y_test, dt_gini_partDay_SMOTE.predict_proba(X_test)[:,1])
roc_df_SMOTED = pd.DataFrame()
roc_df_SMOTED['fpr'] = fpr_SMOTED
roc_df_SMOTED['tpr'] = tpr_SMOTED
roc_df_SMOTED['thresholds'] = thresholds_SMOTED
roc_df_SMOTED.head()

baseline = alt.Chart(roc_df_SMOTED).mark_line(strokeDash=[20,5], color = 'black').encode(
                                                                alt.X('thresholds', scale = alt.Scale(domain=[0, 1])),
                                                                alt.Y('thresholds', scale = alt.Scale(domain=[0, 1])))
roc_line = alt.Chart(roc_df_SMOTED).mark_line(color = 'red').encode(
                                                                alt.X('fpr', title="false positive rate"),
                                                                alt.Y('tpr', title="true positive rate"))

smotedDT = roc_line + baseline.properties(title='Decision Tree (SMOTED) ROC curve ').interactive()
st.altair_chart(smotedDT, use_container_width=True)

"""## Regression Model"""

"""### Linear Regression, Decision Tree Regressor and SVM (rbf, poly, linear)"""

X_AGE_train, X_AGE_test, Y_AGE_train, Y_AGE_test = train_test_split(X_AGE, y_AGE, test_size = 0.30, random_state = 0)


from sklearn.linear_model import LinearRegression

cols = boruta_score_AGE['features'].head(9)
X_AGE = df_FS[cols]
y_AGE = df_FS['AGE_RANGE']


lr_AGE = LinearRegression().fit(X_AGE_train, Y_AGE_train)

lr_acc = lr_AGE.score(X_AGE_test, Y_AGE_test)

from sklearn.tree import DecisionTreeRegressor

md_range = range(1,20)
scores = []

# your codes here...
for i in md_range:
    dt = DecisionTreeRegressor(max_depth=i)
    dt.fit(X_AGE_train, Y_AGE_train)
    scores.append(dt.score(X_AGE_test, Y_AGE_test))

section_gap()
"""
Hyperparameter tuning of Decision Tree Regresor
"""
plt.figure()
plt.xlabel('max depth')
plt.ylabel('accuracy')
plt.title('Accuracy by max depth')
plt.scatter(md_range, scores)
plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]);
plt.plot(md_range, scores, color='green', linestyle='dashed', linewidth=1, markersize=5)
st.pyplot()



dt_AGE = DecisionTreeRegressor(max_depth=2)
dt_AGE.fit(X_AGE_train, Y_AGE_train)

dt_acc = dt_AGE.score(X_AGE_test, Y_AGE_test)

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


svr_rbf_AGE = SVR(kernel = 'rbf')
svr_rbf_AGE.fit(X_AGE_train, Y_AGE_train)

svr_rbf_AGE_acc = svr_rbf_AGE.score(X_AGE_test, Y_AGE_test)

svr_poly_AGE = SVR(kernel = 'poly')
svr_poly_AGE.fit(X_AGE_train, Y_AGE_train)

svr_poly_AGE_acc = svr_poly_AGE.score(X_AGE_test, Y_AGE_test)

svr_linear_AGE = SVR(kernel = 'linear')
svr_linear_AGE.fit(X_AGE_train, Y_AGE_train)

svr_linear_AGE_acc = svr_linear_AGE.score(X_AGE_test, Y_AGE_test)

"""#### Model evaluation"""

"""
Accuracy 
"""
modelAccuracyReg = {'Model': ['Linear_Regression','Decision_Tree', 'Support Vector Regressor'], 
'Accuracy':[lr_acc,dt_acc,svr_rbf_AGE_acc]}

modelAccuracyRegression = pd.DataFrame(modelAccuracyReg)
st.dataframe(modelAccuracyRegression)

modelreg = alt.Chart(modelAccuracyRegression).mark_bar().encode(
    x='Model:O',
    y="Accuracy:Q",
    color=alt.condition(
        alt.datum.Accuracy > 0.7, 
        alt.value('green'),     
        alt.value('red')   
    )
).properties(width=600)
st.altair_chart(modelreg, use_container_width=True)