# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 17:03:06 2021

@author: K
"""
import pandas as pd 
###우울증 18,19년도 
depres=pd.read_csv('depressive disorder 1819.csv')
depres_ed=depres.drop(['age','measure','sex','metric','cause'],axis=1)
depres_ed1=depres_ed.drop(['upper','lower'],axis=1)
depres_ed2=depres_ed1.rename\
    (columns={'location':'Country','val':'disorders','year':'Year'})

dep1=depres_ed2 #우울증 1819년 

###우울증 17년까지 
depre1=pd.read_csv('number-with-depression-by-country.csv')
depre1_ed=depre1.rename\
    (columns={'Prevalence - Depressive disorders - Sex: Both - Age: All Ages (Number)':'disorders'})
del depre1_ed['Code']
depre1_ed_2=depre1_ed.rename(columns={'Entity':'Country'})

dep2=depre1_ed_2


dep3=pd.concat([dep1,dep2])
dep4=dep3.sort_values(by='Country') # 나라 이름으로 정렬 
dep5=dep4[dep4['Year']>=2005]#2005년 이후 데이터만 추출 
dep5=dep5.sort_values(by=['Country','Year'])
dep5.to_csv('depression.csv',index=False)
depression=pd.read_csv('depression.csv')

'''
depre1_ed_1517.to_csv('depression_15to17.csv',index=False)
depression_15to17=pd.read_csv('depression_15to17.csv')
'''

###### 세계행복지수 whr 
whr=pd.read_csv('world-happiness-report.csv')
whr_ed1=whr.drop(['Positive affect','Negative affect'],axis=1)
whr_ed2=whr_ed1.rename(columns={'Country name':'Country','year':'Year'})
whr_ed2.to_csv('whr.csv',index=False)
whr=pd.read_csv('whr.csv')
whr.Year.count() # 1949 

'''
merge_left = \
    pd.merge(df1,df2, how='left', left_on='stock_name', right_on='name')

'''

#나라와 년도를 기준으로 합치기 
hap=pd.merge(whr,depression,how='left',on=['Country','Year'])
hap.isnull().sum()

####결측값 앞에년도로 채우기 
hap=hap.fillna(method='ffill')
hap.to_csv('report1.csv',index=False)

import pandas as pd
report1=pd.read_csv('report1.csv')
report1

#전세계 학업년수 평균 추가 
scho=pd.read_csv('mean-years-of-schooling-1.csv')
scho.columns
scho=\
    scho.rename(columns={'Entity':'Country','Average Total Years of Schooling for Adult Population (Lee-Lee (2016), Barro-Lee (2018) and UNDP (2018))':'schooling'})
    
del scho['Code']
scho.isnull().sum()
scho.Year.unique()
###중간1이랑 합치기 
hap1=pd.merge(report1,scho,on=['Country','Year'],how='left')

#결측값 채우기 
hap1.fillna(method='ffill',inplace=True)
hap1.isnull().sum()
hap1.to_csv('report2.csv',index=False)
report2=pd.read_csv('report2.csv')

#알콜섭취량 # 알콜 x 사망률 
alc=pd.read_csv('death-rates-from-alcohol-use-disorders (1).csv')
alc.columns
del alc['Code']

alc=\
    alc.rename(columns={'Entity':'Country','Deaths - Alcohol use disorders - Sex: Both - Age: Age-standardized (Rate)':'alcohol disorders'})

###합치기
hap_ed5=pd.merge(report2,alc,on=['Country','Year'],how='left')
hap_ed5
#2018년 이후 

#결측값 채우기 
hap_ed5.fillna(method='ffill',inplace=True)
hap_ed5.to_csv('report3.csv',index=False)

report3=pd.read_csv('report3.csv')
report3.isnull().sum()

##gdp 대비 세금 
tax=pd.read_csv('total-tax-revenues-gdp.csv')

del tax['Code']
tax=tax.rename(columns={'Entity':'Country',\
                    'Total tax revenue (% of GDP) (ICTD (2019))':'tax'})

##합치기 
hap7=pd.merge(report3,tax,on=['Country','Year'],how='left')
hap7.isnull().sum()
#fillna
hap7.fillna(method='ffill',inplace=True)

hap7.to_csv('report4.csv',index=False)

import pandas as pd
report4=pd.read_csv('report4.csv')
report4.info()

#피파 ##평균대입 
import pandas as pd 
import datetime
fifa=pd.read_csv('fifa_ranking-2021-05-27.csv')
fifa.info()
fifa.drop\
    (['id','country_abrv','confederation','rank_change',\
      'previous_points','total_points'],\
     axis=1,inplace=True)
fifa.rename(columns={'country_full':'Country'},inplace=True)
fifa
list1=fifa['rank_date'].tolist()
asd=pd.to_datetime(list1)
df=pd.DataFrame(asd,columns=['asd'])
fifa['Year']=df['asd'].dt.year
fifa.Country.unique()
fifa1=fifa.sort_values(by=['Year','Country'])
fifa1.Country.unique()

fifa_c=fifa1[fifa1['Country']=='China PR']
fifa_c.groupby(['Year']).mean()
fifa_mean=fifa.groupby(['Country','Year']).mean()
fifa_mean=fifa_mean.astype(int) # 피파랭킹 평균 

### report5
hap9=pd.merge(report4, fifa_mean, how='left',on=['Country','Year'])
hap9.fillna(method='ffill',inplace=True)
hap9.to_csv('report5.csv',index=False)
report5=pd.read_csv('report5.csv')
report5.isnull().sum()
report5.rename(columns={'disorders':'depressive disorders','rank':'fifa rank',\
                        'tax':'tax_revenue_gdp'},inplace=True)

import pandas as pd 
import numpy as np     
report5=pd.read_csv('C:/py/pjt/report5.csv')
report5

####자살률 추가 
shadea=pd.read_csv('share-deaths-suicide.csv')
shadea.info()

del shadea['Code']

##컬럼명 바꾸기 
shadea.rename\
    (columns={'Entity':'Country','Deaths - Self-harm - Sex: Both - Age: All Ages (Percent)':'suicide rates'},inplace=True)

shadea

#합치기 
report6=pd.merge(report5,shadea,how='left',on=['Country','Year'])
report6.fillna(method='ffill',inplace=True)

obesity=pd.read_csv('share-of-deaths-obesity.csv')
del obesity['Code']

obesity.rename(columns={'Entity':'Country','Obesity (IHME, 2019)':'Obesity'},inplace=True)
obesity

report7=pd.merge(report6,obesity,how='left',on=['Country','Year'])
report7.fillna(method='ffill',inplace=True)
report7.isnull().sum()
report7.to_csv('report7.csv',index=False)
report7.info()


