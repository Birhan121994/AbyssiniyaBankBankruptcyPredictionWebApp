import numpy as np
import pandas as pd
import webbrowser
import matplotlib.pyplot as plt
import cv2 as cv
# import cv2.cv2

import dtale
from dtale.views import startup
from dtale.app import get_instance
import streamlit as st
import streamlit.components.v1 as com
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from autoviz.AutoViz_Class import AutoViz_Class
from streamlit_option_menu import  option_menu
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score,recall_score,precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import pickle
from pathlib import Path

import streamlit_authenticator as sa
from pywedge import *
import pywedge as py

from dataprep.datasets import load_dataset
from dataprep.eda import create_report



registeredNames = ['Birhan Tamiru','James Miller']

usernames = ['birhan251','james220']

file_path = Path(__file__).parent / "encodede_pin.pkl"

with file_path.open("rb") as file:
    encodded_pin = pickle.load(file)

pin = sa.Authenticate(registeredNames,usernames,encodded_pin ,'BankCryptoCurrencyPrediction','abcdef',cookie_expiry_days=30)

name,Autentication_status,username = pin.login('Login','main')

if Autentication_status == False:
    st.error('Username/Password id incorrect')

if Autentication_status == None:
    st.warning('Please enter your username and password')


if Autentication_status:
    # st.markdown('''
    # <style>
    # .css-9s5bis.edgvbvh3{
    # visibility:hidden;
    # }
    # </style>
    # ''',unsafe_allow_html=True)

    st.markdown('''
    <style>
    
    div.stMarkdown .css-1offfwp .e16nr0p33 p{
    color:#FF4B4B;
    }
    </style>
    
    ''',unsafe_allow_html=True)
    st.markdown('''
    <style>
    .css-18e3th9{
    padding-top:5px;
    }
    .css-6kekos{
    background-color:#FF4B4B;
    color:white;
    border-color:#FF4B4B;
    }
    .css-6kekos:hover{
    background:white;
    color:#FF4B4B;
    border-color:#FF4B4B;
    }
    
    .css-1prua9e{
    background-color:#FF4B4B;
    color:white;
    border-color:#FF4B4B;
    }
    .css-1prua9e:hover{
    background:white;
    color:#FF4B4B;
    border-color:#FF4B4B;
    }
    .css-1ftupb1{
    display:flex;
    flex-direction:row;
    justify-content:center;
    box-shadow: 3px 3px 10px rgba(0, 0,0, 0.15);
    padding-top:20px;
    padding-bottom:20px;
    }
    div.css-1kyxreq.etr89bj2{
    display:flex;
    justify-content:center;
    }
    .css-434r0z{
    column-gap:1rem;
    } 
   div.css-1offfwp.e16nr0p33{
   padding-left:10px;
   padding-right:10px;
   }
   # .css-10trblm{
   # text-align:center;
   # }
   div.css-6qob1r.e1fqkh3o3{
   background-color:#e6f0fa;
   }
    .css-qri22k.egzxvld0{
    visibility:hidden;
    display:none;
    }


    </style>
    ''',unsafe_allow_html=True)

    st.image('https://sp-ao.shortpixel.ai/client/to_auto,q_lossless,ret_img/https://www.bankofabyssinia.com/wp-content/uploads/2020/10/Asset-7@2x.png',use_column_width='auto', width=100)
    selected = option_menu(
        menu_title = "Main Menu",
        options = ['Home','EDA','ML Page','ORS Page'],
        icons = ['house','file-bar-graph-fill','steam','yin-yang'],
        menu_icon = 'cast',
        default_index = 0,
        orientation="horizontal"
    )

    if selected == "Home":
        tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(['1',
                                                 '2',
                                                 '3',
                                                 '4',
                                                 '5',
                                                 '6'])

        with tab1:
            st.image('https://sp-ao.shortpixel.ai/client/to_auto,q_lossless,ret_img,w_1707,h_960/https://www.bankofabyssinia.com/wp-content/uploads/2022/04/attt.jpg',use_column_width='auto')
        with tab2:
            st.image('https://sp-ao.shortpixel.ai/client/to_auto,q_lossless,ret_img,w_1707,h_960/https://www.bankofabyssinia.com/wp-content/uploads/2022/08/299284874_420301920076449_5378619177392451991_n.jpg',use_column_width='auto')
        with tab3:
            st.image('https://sp-ao.shortpixel.ai/client/to_auto,q_lossless,ret_img,w_1707,h_960/https://www.bankofabyssinia.com/wp-content/uploads/2022/04/virtuccc.jpg',use_column_width='auto')
        with tab4:
            st.image('https://sp-ao.shortpixel.ai/client/to_auto,q_lossless,ret_img,w_1920,h_1080/https://www.bankofabyssinia.com/wp-content/uploads/2022/03/bot-with-link.png',use_column_width='auto')
        with tab5:
            st.image('https://sp-ao.shortpixel.ai/client/to_auto,q_lossless,ret_img,w_1920,h_1080/https://www.bankofabyssinia.com/wp-content/uploads/2022/02/c2.jpg',use_column_width='auto')
        with tab6:
            st.image('https://sp-ao.shortpixel.ai/client/to_auto,q_lossless,ret_img,w_1707,h_960/https://www.bankofabyssinia.com/wp-content/uploads/2022/04/virtuccc.jpg',use_column_width='auto')
        st.write('---')
        st.markdown('''### **<span style='color:#FF4B4B'>Presenting products and services that are right for you</span>**''',unsafe_allow_html=True)
        cols1,cols2,cols3,cols4 = st.columns(4,gap="medium")
        with cols1:
            st.image("https://img.icons8.com/color/48/000000/merchant-account.png")
            st.markdown('''#### <span style="text-align:center">**Online Business**</span>''',unsafe_allow_html=True)
            st.markdown('Explore the power of simple and smart banking. Bank online with over 250 services')
            st.markdown('''###### **<span style='color:#FF4B4B;margin-right:20px;'>Apply Using Online</span>**''',unsafe_allow_html=True)

        with cols2:
            st.image("https://img.icons8.com/external-flaticons-flat-flat-icons/55/000000/external-business-plan-home-based-business-flaticons-flat-flat-icons.png")
            st.markdown('''#### <span style="text-align:center">**Business Plan**</span>''',unsafe_allow_html=True)
            st.markdown('From shares to shopping centres ,there’s a huge range of investments to choose')
            st.markdown('''###### **<span style='color:#FF4B4B;margin-right:10px;'>Take the first step</span>**''',unsafe_allow_html=True)

        with cols3:
            st.image("https://img.icons8.com/external-inipagistudio-lineal-color-inipagistudio/55/000000/external-mobile-banking-personal-finance-inipagistudio-lineal-color-inipagistudio.png")
            st.markdown('''#### <span style="text-align:center">**Mobile Bank**</span>''', unsafe_allow_html=True)
            st.markdown('Explore the power of simpler and smarter banking. Bank online with over 250 services')
            st.markdown('''###### **<span style='color:#FF4B4B'>Find out more</span>**''',unsafe_allow_html=True)

        with cols4:
            st.image("https://img.icons8.com/external-kmg-design-outline-color-kmg-design/55/000000/external-deposit-economy-kmg-design-outline-color-kmg-design.png")
            st.markdown('''#### <span style="text-align:center">**Online Deposit**</span>''', unsafe_allow_html=True)
            st.markdown('Explore the power of simpler and smarter banking. Bank online with over 250 services')
            st.markdown('''###### **<span style='color:#FF4B4B'>Learn more</span>**''',unsafe_allow_html=True)
        st._transparent_write('---')
        st.markdown('''### **Business Banking**''')
        t1,t2,t3,t4,t5 = st.tabs(['Investing Banks','Find a Credit Card','Payment Technologies','Card Benefits','Digital Wallets'])

        with t1:
            c1,c2 = st.columns(2,gap="medium")
            with c1:
                st.image('https://alister-bank.cmsmasters.net/wp-content/uploads/2015/11/2.jpg')
            with c2:
                st.subheader('We can help you achieve your goals!')
                st.markdown('The Ethiopian largest banking groups are required to comply with ring-fencing requirements from 1 January 2019. Find out more about our approach and what it means for you')

                def say_hello1():
                    st.write('Developed by Birhan Tamiru')


                st.button('Learn more', help="These app is developed by Birhan Tamiru.", on_click=say_hello1(),key=1)

        with t2:
            c3, c4 = st.columns(2, gap="medium")
            with c3:
                st.image('https://alister-bank.cmsmasters.net/wp-content/uploads/2015/11/home-2-1.jpg')
            with c4:
                st.subheader('Find the card that’s right for you. Explore the benefits.')
                st.markdown('Get the financial freedom you deserve. Credit cards offer exceptional benefits, rewards, services and spending power that can help make your financial and personal dreams come true.')
            st.button('Learn more', help="These app is developed by Birhan Tamiru.", on_click=say_hello1(), key=2)
        with t3:
            c5, c6 = st.columns(2, gap="medium")
            with c5:
                st.image('https://alister-bank.cmsmasters.net/wp-content/uploads/2015/11/home-3-1.jpg')
            with c6:
                st.subheader('Transforming the way you pay. Explore new ways to pay')
                st.markdown('We offer an array of products that make it possible to pay anywhere, on any device. We’re bringing solutions to life to change the way you pay – through our innovative digital wallet service.')
            st.button('Learn more', help="These app is developed by Birhan Tamiru.", on_click=say_hello1(), key=3)

        with t4:
            c7, c8 = st.columns(2, gap="medium")
            with c7:
                st.image('https://alister-bank.cmsmasters.net/wp-content/uploads/2015/11/home-4-1.jpg')
            with c8:
                st.subheader('Debit and Credit Card Protection Tips to Prevent Financial Fraud')
                st.markdown('Escape from the daily routine and relax at a spa, go on a thrill-seeking adventure, or take in a round of golf with pro instruction. Access restaurant reviews and make dining reservations online.')
            st.button('Learn more', help="These app is developed by Birhan Tamiru.", on_click=say_hello1(), key=4)
        with t5:
            c9, c10 = st.columns(2, gap="medium")
            with c9:
                st.image('https://alister-bank.cmsmasters.net/wp-content/uploads/2015/11/home-5-1.jpg')
            with c10:
                st.subheader('Privacy, Innovation and Security in the Digital Payments World')
                st.markdown('Easy – Load credit, debit, reloadable prepaid or small business cards from participating Alister Bank issuers. Any of your cards can be used across hundreds of thousands of supported merchants.')
            st.button('Learn more', help="These app is developed by Birhan Tamiru.", on_click=say_hello1(), key=5)
        st._transparent_write('---')
        st.markdown('''### **Habesha Debit Cards**''')
        with st.container():
            cll1,cll2=st.columns(2,gap="medium")
            with cll1:
                st.image('https://sp-ao.shortpixel.ai/client/to_auto,q_lossless,ret_img/https://www.bankofabyssinia.com/wp-content/uploads/2021/06/Asset-2@4xed-1-300x191.png')
            with cll2:
                st.markdown('''#### **<span style="color:#FF4B4B">Gold Card</span>**''',unsafe_allow_html=True)
                st.markdown('As main business partners of BoA, you are of paramount importance to us. Gold Debit card powers your status in our bank and allows you unmatched purchasing power at a single stop. With superlative cash withdrawing capacity, you are the owner of unimaginable possibilities.')
                st.button('Learn more',help="Developed by Birhan Tamiru",key=6)

        with st.container():
            cll3,cll4=st.columns(2,gap="medium")
            with cll3:
                st.image('https://sp-ao.shortpixel.ai/client/to_auto,q_lossless,ret_img/https://www.bankofabyssinia.com/wp-content/uploads/2021/06/Asset-1@4x-300x187.png')

            with cll4:
                st.markdown('''#### **<span style="color:#FF4B4B">AbyssinAmeen Card</span>**''',unsafe_allow_html=True)
                st.markdown('Ameen interset free banking offers you the Habesha Ameen debit card which gives purchasing power up to ETB 200,000 daily purchases. you are offered exceptional services through this debit card. ')
                st.button('Learn more',help="Developed by Birhan Tamiru",key=7)
        with st.container():
            cll5,cll6=st.columns(2,gap="medium")
            with cll5:
                st.image('https://sp-ao.shortpixel.ai/client/to_auto,q_lossless,ret_img/https://www.bankofabyssinia.com/wp-content/uploads/2021/06/Asset-1@4x-300x187.png')
            with cll6:
                st.markdown('''#### **<span style="color:#FF4B4B">Classic Card</span>**''',unsafe_allow_html=True)
                st.markdown('BoA values its banking customers. That is why we provide you with the Habesha classic debit card which gives a massive purchasing power. With up to ETB 200,000 daily purchases you are offered exceptional services through this debit card.')
                st.button('Learn more',help="Developed by Birhan Tamiru",key=8)


        ta1,ta2,ta3 = st.tabs(['Saving Accounts','Loan','Current Accounts'])
        with ta1:
            st.subheader('Adey Women’s Account')
            st.markdown('''##### **__Empowerment of Women starts here__**''')
            st.markdown('A present from us to women, come and open an Adey account with a higher interest rate. We offer discounts and ease of access to your accounts even when you are shopping in supermarkets.  ')
            st.markdown('BoA understands the role women play in our lives. As such to simplify and facilitate your personal business, endeavors, and efforts, we have designated a special account for you. We offer you increased saving rates to show you our appreciation and valuation of your growth. Saving with Adey account will create empowerment and drive to succeed. ')
            st.button('Learn more',help="Developed by Birhan Tamiru",key=9)

        with ta2:
            st.subheader('Term loan')
            st.markdown('''##### **__The best financial choice__**''')
            st.markdown('This finance is designed to boost working capital or project finance to be repaid within a specific time usually with interest. BoA is engaged in delivering unsurpassed working financing credit to ensure the smooth running of businesses with Short, Medium, and Long-term loans.')
            st.button('Learn more',help="Developed by Birhan Tamiru",key=10)

        with ta3:
            st.subheader('ECX Related Accounts')
            st.markdown('''##### **__Secure your tomorrow, today__**''')
            st.markdown('This account is for business owners who conduct trading activities in the Ethiopian Commodity Exchange (ECX). It is intended to facilitate trading and it is opened by conventional payment operation.')
            st.button('Learn more',help="Developed by Birhan Tamiru",key=11)

        st.write('---')

        st.subheader('PlayLists')
        with st.expander('Abyssinya bank play lists'):
                d1, d2 = st.columns(2)
                with d1:
                   st.video('https://youtu.be/TSfzReLpkoo', format="video/mp4", start_time=0)
                with d2:
                   st.video('https://www.youtube.com/watch?v=eTleriox4EI', format="video/mp4", start_time=0)

        st._transparent_write('___')
        st.subheader('Money Transfer')
        st.markdown('Leading international money transfer agents working with Bank of Abyssinia')
        cld1,cld2,cld3,cld4,cld5= st.columns(5,gap="medium")

        with cld1:
            st.image('https://sp-ao.shortpixel.ai/client/to_auto,q_lossless,ret_img/https://www.bankofabyssinia.com/wp-content/uploads/2020/08/western-union-1.png')
        with cld2:
            st.image('https://sp-ao.shortpixel.ai/client/to_auto,q_lossless,ret_img/https://www.bankofabyssinia.com/wp-content/uploads/2020/08/moneygram-international-1.png')
        with cld3:
            st.image('https://sp-ao.shortpixel.ai/client/to_auto,q_lossless,ret_img/https://www.bankofabyssinia.com/wp-content/uploads/2020/08/ria@2x.png')
        with cld4:
            st.image('https://sp-ao.shortpixel.ai/client/to_auto,q_lossless,ret_img/https://www.bankofabyssinia.com/wp-content/uploads/2022/09/Ethiodash.png')
        with cld5:
            st.image('https://sp-ao.shortpixel.ai/client/to_auto,q_lossless,ret_img/https://www.bankofabyssinia.com/wp-content/uploads/2020/08/world-re@2x.png')

        data1 = ['USD','GBP','EUR','AED','CHF(Switzerland)','NOK','SAR','SEK','CAD']
        data2 = [52.7162, 55.7056, 51.1822, 12.9882, 55.7056, 51.1822, 12.9882,52.7162, 55.7056]
        data3 = [53.7705,56.8197,52.2058,13.2480,53.7705,56.8197,52.2058,13.2480,53.7705]

        da1 = pd.Series(data1)
        da2 = pd.Series(data2)
        da3 = pd.Series(data3)
        frame = {'Currency': da1, 'Buying': da2, 'Selling': da3}
        dataframe = pd.DataFrame(frame)
        clt1,clt2 = st.columns(2,gap="medium")

        with clt1:
            st.subheader('Exchange Rate')
            st.write(dataframe)
        with clt2:
            st.subheader('Donation')
            st.image('https://1.bp.blogspot.com/-FM74gtxXc2s/Xp8nlRrFAmI/AAAAAAAACCY/bD0zbqRObnYQWVMsc8tjMAgvgCq5jSGZwCLcBGAsYHQ/w380/donation-gif.gif')

        st.write('---')
        with st.container():
            lt1, lt2 = st.columns(2, gap="small")
            with lt1:
                st.image('https://sp-ao.shortpixel.ai/client/to_auto,q_lossless,ret_img/https://www.bankofabyssinia.com/wp-content/uploads/2020/10/Asset-7@2x.png')
            with lt2:
                st.markdown('Bank of Abyssinia © 2022 / All Rights Reserved')
        with st.container():
            lt3,lt4,lt5,lt6 = st.columns(4,gap="small")
            with lt3:
                st.image('https://img.icons8.com/ios-glyphs/30/000000/twitter--v1.png')
            with lt4:
                st.image('https://img.icons8.com/material-two-tone/24/000000/facebook-f--v2.png')
            with lt5:
                st.image('https://img.icons8.com/ios-glyphs/30/000000/linkedin.png')
            with lt6:
                st.image('https://img.icons8.com/ios-glyphs/30/000000/instagram-new.png')
            st.markdown('''##### **__Developed By <span style='color:green'>Birhan Tamiru</span> For <span style='color:orange'>BankOfAbyssinia</span>__**''',unsafe_allow_html=True)
        st.write('---')

        st.sidebar.image('https://sp-ao.shortpixel.ai/client/to_auto,q_lossless,ret_img/https://www.bankofabyssinia.com/wp-content/uploads/2020/10/Asset-7@2x.png',use_column_width='auto', width=100)
        st.sidebar.title(f"Welcome {name}")
        # with st.sidebar.header('1. Upload your CSV data'):
        #     uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
        #     st.sidebar.markdown("""
        #   [Example CSV input file](C:/Users/GL/Downloads/data.csv)
        #   """)
        st.sidebar.image('https://th.bing.com/th/id/R.56dada8736e66549e4324d42bb2227aa?rik=U5oROAQVEAbaKw&pid=ImgRaw&r=0')
        st.sidebar.markdown("""
          [Bank of Abyssinya Main Website](https://www.bankofabyssinia.com)
          """)
    elif selected == "EDA":
        st.sidebar.image('https://sp-ao.shortpixel.ai/client/to_auto,q_lossless,ret_img/https://www.bankofabyssinia.com/wp-content/uploads/2020/10/Asset-7@2x.png',use_column_width='auto', width=100)
        clv1,clv2 = st.columns(2,gap="small")
        with clv1:
            st.image('https://image.jimcdn.com/app/cms/image/transf/none/path/s706008bd7e26dccf/image/i2ec15cb83ba7436e/version/1476884865/image.png')
        with clv2:
            st.markdown('''#### **<span style="color:#FF4B4B">Explanatory Data Analysis</span>**''',unsafe_allow_html=True)
            st.markdown('Explanatory Data Analysis (EDA) in statistics is an approach to analyzing data sets to summarize their main characteristics, often with visual methods. A statistical model can be used or not, but primarily EDA is for seeing what the data can tell us beyond the formal modeling or hypothesis testing task.')

        st.write('---')
        # Upload CSV data

        st.sidebar.title(f"Welcome {name}")
        with st.sidebar.header('1. Upload your CSV data'):
            uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
            st.sidebar.subheader('2. Related Links')
          #   st.sidebar.markdown("""
          # [<span style="color:#FF4B4B;text-decoration:none;">Example CSV input file</span>](C:/Users/GL/Downloads/data.csv)
          # """,unsafe_allow_html=True)

            if st.sidebar.button("Distplots",key=90):
                webbrowser.open_new_tab('file:///C:/Users/GL/PycharmProjects/VirtualAssistant/AutoViz_Plots/AutoViz/distplots_nums.html')
            if st.sidebar.button("Pair_Scatters",key=92):
                webbrowser.open_new_tab('file:///C:/Users/GL/PycharmProjects/VirtualAssistant/AutoViz_Plots/AutoViz/pair_scatters.html')
        # Pandas Profiling Report
        if uploaded_file is not None:
            @st.cache
            def load_csv():
                csv = pd.read_csv(uploaded_file)
                return csv


            df = load_csv()
            create_report(df).show_browser()
            av = AutoViz_Class()
            filename = ""
            dft = av.AutoViz(
                filename,
                sep=",",
                depVar="",
                dfte=df,
                header=0,
                verbose=2,
                lowess=False,
                chart_format="html",
                max_rows_analyzed=150000,
                max_cols_analyzed=30,
                save_plot_dir=None
            )

            pr = ProfileReport(df, explorative=True)
            st.markdown('''## <span style="color:#FF4B4B">**DataFrame of Example Dataset**</span>''',
                        unsafe_allow_html=True)

            st.write(df)
            st.write('---')
            st.markdown('''## <span style="color:#FF4B4B">**Data Report**</span>''',unsafe_allow_html=True)

            st_profile_report(pr)
            st.write('---')
            with st.container():
                lt1, lt2 = st.columns(2, gap="small")
                with lt1:
                    st.image(
                        'https://sp-ao.shortpixel.ai/client/to_auto,q_lossless,ret_img/https://www.bankofabyssinia.com/wp-content/uploads/2020/10/Asset-7@2x.png')
                with lt2:
                    st.markdown('Bank of Abyssinia © 2022 / All Rights Reserved')
            with st.container():
                lt3, lt4, lt5, lt6 = st.columns(4, gap="small")
                with lt3:
                    st.image('https://img.icons8.com/ios-glyphs/30/000000/twitter--v1.png')
                with lt4:
                    st.image('https://img.icons8.com/material-two-tone/24/000000/facebook-f--v2.png')
                with lt5:
                    st.image('https://img.icons8.com/ios-glyphs/30/000000/linkedin.png')
                with lt6:
                    st.image('https://img.icons8.com/ios-glyphs/30/000000/instagram-new.png')
                st.markdown(
                    '''##### **__Developed By <span style='color:green'>Birhan Tamiru</span> For <span style='color:orange'>BankOfAbyssinia</span>__**''',
                    unsafe_allow_html=True)
            st.write('---')
        else:
            st.info('Awaiting for CSV file to be uploaded.')
            if st.button('Press to use Example Dataset'):
                # Example data
                @st.cache
                def load_data():
                    a = pd.DataFrame(
                        np.random.rand(100, 10),
                        columns=['Bankrupt?', ' ROA(C) before interest and depreciation before interest',
                                 'ROA(A) before interest and % after tax: Return On Total Assets(A)',
                                 'ROA(B) before interest and depreciation after tax: Return On Total Assets(B)',
                                 'Operating Gross Margin: GrossProfit / Net Sales',
                                 'Realized Sales GrossMargin: RealizedGrossProfit / NetSales',
                                 'Operating ProfitRate: Operating Income / NetSales',
                                 ' Pre-tax net Interest  Rate: Pre - Tax Income / Net Sales',
                                 'After - tax netInterest Rate: Net Income / NetSales',
                                 'Non - industry income and expenditure / revenue: Net Non - operatingIncome Ratio',
                                 # 'Continuous interestrate(aftertax): NetIncome - Exclude Disposal Gain or Loss / NetSales',
                                 # 'Operating Expense Rate: Operating Expenses / NetSales',
                                 # 'Research and development expenserate: (Research and Development Expenses) / NetSales',
                                 # 'Cash flow rate: Cash Flow from Operating / Current Liabilities',
                                 # 'Interest - bearing debt interest rate: Interest - bearing Debt / Equity',
                                 # 'Tax rate(A): Effective Tax Rate',
                                 # 'Net Value Per Share(B): Book Value Per Share(B)',
                                 # 'Net Value Per Share(A): Book Value Per Share(A)',
                                 # 'Net Value Per Share(C): Book Value Per Share(C)',
                                 # 'Persistent EPS in the Last Four Seasons: EPS - Net Income',
                                 # 'Cash Flow Per Share',
                                 # 'Revenue Per Share(Yuan ¥): Sales Per Share',
                                 # 'Operating Profit Per Share(Yuan ¥): Operating IncomePer Share',
                                 # 'Per Share Net profit before tax(Yuan ¥): Pretax Income Per Share',
                                 # 'Realized Sales Gross Profit Growth Rate',
                                 # 'Operating Profit Growth Rate: Operating Income Growth',
                                 # 'After - tax Net Profit Growth Rate: Net Income Growth',
                                 # 'RegularNetProfitGrowth Rate: ContinuingOperatingIncome afterTaxGrowth',
                                 # 'ContinuousNetProfitGrowthRate: Net Income - ExcludingDisposalGain or LossGrowth',
                                 # 'Total Asset GrowthRate: TotalAssetGrowth',
                                 # 'Net Value  Growth Rate: Total Equity Growth',
                                 # 'Total Asset Return Growth Rate Ratio: Return on Total Asset Growth',
                                 # 'Cash Reinvestment %: Cash ReinvestmentRatio',
                                 # 'Current Ratio',
                                 # 'QuickRatio: AcidTest',
                                 # 'InterestExpenseRatio: InterestExpenses / TotalRevenue',
                                 # 'Total  debt / Total  net worth: Total Liability / Equity Ratio',
                                 # 'Debtratio %: Liability / Total Assets',
                                 # 'Net worth / Assets: Equity / TotalAssets',
                                 # 'Long - term fund suitability ratio(A): (Long - term Liability+Equity) / FixedAssets',
                                 # 'Borrowing of Interest - bearing Debt',
                                 # 'Contingent liabilities / Net worth: ContingentLiability / Equity',
                                 # 'Operating profit / Paid - in capital: Operating Income / Capital',
                                 # 'Net profit before tax / Paid - in capital: PretaxIncome / Capital',
                                 # 'Inventory and accountsreceivable / Net value: (Inventory + Accounts Receivables) / Equity',
                                 # 'Total Asset Turnover',
                                 # 'Accounts Receivable Turnover',
                                 # 'Average Collection Days: Days Receivable Outstanding',
                                 # 'InventoryTurnover Rate(times)',
                                 # 'Fixed  Assets  TurnoverFrequency',
                                 # 'Net  Worth Turnover Rate(times): EquityTurnover',
                                 # 'Revenue perperson: Sales Per Employee',
                                 # 'Operating profit per person: Operation Income Per Employee',
                                 # 'Allocation rateper person: Fixed  Assets  PerEmployee',
                                 # 'Working   Capital   toTotalAssets',
                                 # 'Quick Assets / TotalAssets',
                                 # 'Current Assets / Total Assets',
                                 # 'Cash / TotalAssets',
                                 # 'Quick Assets / CurrentLiability',
                                 # 'Cash / Current Liability',
                                 # 'Current Liability to Assets',
                                 # 'Operating Funds to Liability',
                                 # 'Inventory / Working Capital',
                                 # 'Inventory / Current  Liability',
                                 # 'Current Liabilities / Liability',
                                 # 'Working Capital / Equity',
                                 # 'Current Liabilities / Equity',
                                 # 'Long - term Liability to Current Assets',
                                 # 'Retained Earnings to Total Assets',
                                 # 'Total income / Total expense',
                                 # 'Total expense / Assets',
                                 # 'Current Asset Turnover Rate: Current Assets to Sales',
                                 # 'Quick Asset Turnover Rate: Quick Assets to Sales',
                                 # 'Working capitcal Turnover  Rate: Working Capital to Sales',
                                 # 'Cash Turnover Rate: Cash to Sales',
                                 # 'Cash Flow to Sales',
                                 # 'Fixed Assets to Assets',
                                 # 'Current Liability to Liability',
                                 # 'Current Liability to Equity',
                                 # 'Equity to Long - term Liability',
                                 # 'Cash Flow to Total Assets',
                                 # 'Cash Flow  to Liability',
                                 # 'CFO to Assets',
                                 # 'Cash Flow to Equity',
                                 # 'Current Liability to Current Assets',
                                 # 'Liability - Assets Flag: 1 if Total Liability  exceeds Total Assets, 0 otherwise',
                                 # 'Net Income to Total Assets',
                                 # 'Total assets  to GNP price',
                                 # 'No - credit Interval',
                                 # 'Gross Profit to Sales',
                                 # 'Net Income to Stockholder Equity',
                                 # 'Liability to Equity',
                                 # 'Degree of  Financial Leverage(DFL)',
                                 # 'Interest CoverageRatio (Interest to EBIT)',
                                 # 'Net  Income Flag: 1 if Net Income is Negative for the last two years, 0 otherwise'
                                 # 'Equity to Liability'
                                 ]
                    )
                    return a


                df = load_data()
                create_report(df).show_browser()
                filename = ""
                av = AutoViz_Class()
                dft = av.AutoViz(
                    filename,
                    sep=",",
                    depVar="",
                    dfte=df,
                    header=0,
                    verbose=2,
                    lowess=False,
                    chart_format="html",
                    max_rows_analyzed=150000,
                    max_cols_analyzed=30,
                    save_plot_dir=None
                )

                pr = ProfileReport(df, explorative=True)
                st.markdown('''## <span style="color:#FF4B4B">**DataFrame of Example Dataset**</span>''',unsafe_allow_html=True)
                st.write(df)
                st.write('---')
                st.markdown('''## <span style="color:#FF4B4B">**Data Report**</span>''',unsafe_allow_html = True)
                st_profile_report(pr)
                st.write('---')
                with st.container():
                    lt1, lt2 = st.columns(2, gap="small")
                    with lt1:
                        st.image(
                            'https://sp-ao.shortpixel.ai/client/to_auto,q_lossless,ret_img/https://www.bankofabyssinia.com/wp-content/uploads/2020/10/Asset-7@2x.png')
                    with lt2:
                        st.markdown('Bank of Abyssinia © 2022 / All Rights Reserved')
                with st.container():
                    lt3, lt4, lt5, lt6 = st.columns(4, gap="small")
                    with lt3:
                        st.image('https://img.icons8.com/ios-glyphs/30/000000/twitter--v1.png')
                    with lt4:
                        st.image('https://img.icons8.com/material-two-tone/24/000000/facebook-f--v2.png')
                    with lt5:
                        st.image('https://img.icons8.com/ios-glyphs/30/000000/linkedin.png')
                    with lt6:
                        st.image('https://img.icons8.com/ios-glyphs/30/000000/instagram-new.png')
                    st.markdown(
                        '''##### **__Developed By <span style='color:green'>Birhan Tamiru</span> For <span style='color:orange'>BankOfAbyssinia</span>__**''',
                        unsafe_allow_html=True)
                st.write('---')
    elif selected == "ML Page":
        st.sidebar.image('https://sp-ao.shortpixel.ai/client/to_auto,q_lossless,ret_img/https://www.bankofabyssinia.com/wp-content/uploads/2020/10/Asset-7@2x.png',use_column_width='auto', width=100)
        st.sidebar.title(f"Welcome {name}")
        # with st.sidebar.header('1. Upload your CSV data'):
        #     uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
        #     st.sidebar.markdown("""
        #   [Example CSV input file](C:/Users/GL/Downloads/data.csv)
        #   """)
        data = pd.read_csv('data.csv')
        X = data.drop('Bankrupt?',axis=1)
        Y = data['Bankrupt?']
        if st.sidebar.button('Basic Dataset info',key = 15):
            st.write('---')
            st.write(data.describe())
            # data['Bankrupt?'].value_counts().plot(kind="bar", color=['lightblue', 'orange']);
            # plt.title(
            #     "Bar graph showing frequency distrubution for data with Bankruptcy and without Bankruptcy");
            # plt.xlabel("0=Data without Bankruptcy , 1= Data with Bankruptcy");
            # plt.ylabel('Frequency distrubution values');
            # plt.legend(['Data without Bankruptcy']);
            # plt.xticks(rotation=0);
            st.write('---')
            st.write(data.head(5))
            st.write('---')
            st.write(data.tail(5))
        option  = st.sidebar.selectbox('Choose trained model',
                                       ('RandomForestClassifier','LogisticRegression','SVC','KNC','LinearRegression'))

        if option == 'RandomForestClassifier':
            model = RandomForestClassifier()
            x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=1234)
            model.fit(x_train,y_train)
            if st.sidebar.button("Trained Model Status"):

                y_preds = model.predict(x_test)
                list1 = {"Accuracy Score (%)":model.score(x_test,y_test)*100,
                                    "Recall Score":recall_score(y_preds,y_test),
                                    "F1 Score":f1_score(y_preds,y_test),
                                    "Prescion Score":precision_score(y_preds,y_test),
                                    "Final Score":model.score(x_test,y_test)*100
                                    }
                df3 = pd.DataFrame(list1,index=[0])
                st.write('---')
                st.markdown('''#### <span style="color:#FF4B4B">**Predicction Result Report**</span>''',
                            unsafe_allow_html=True)
                st.write(df3)
                st.write('---')
                c1,c2 = st.columns(2)
                with c1:
                    st.markdown('''#### <span style="color:#FF4B4B">**Predicted Value**</span>''', unsafe_allow_html=True)
                    st.write(np.array(y_preds))
                with c2:
                    st.markdown('''#### <span style="color:#FF4B4B">**Actual True Value**</span>''', unsafe_allow_html=True)
                    st.write(np.array(y_test))

                if st.button('Save Model',key="model1"):
                    file = 'trainedmodel.pkl'
                    with open(file,'wb') as f1:
                        pickle.dump(model,f1)
                        st.success("Model Saved Sucessffuly.")
            st.write('---')
            st.markdown('''#### Load Dataset to Predict''')
            input_file = st.file_uploader('Upload any CSV input file',type=["csv"])
            if input_file is not None:

                data2 = pd.read_csv(input_file)
                data3 = data2.drop(['Name','Account','Job','Email','Gender'],axis=1)
                data4 = data2[['Name','Account','Job','Email','Gender']]
                file_path1 = Path(__file__).parent / "trainedmodel.pkl"
                with file_path1.open("rb") as f1:
                    testmodel = pickle.load(f1)

                result = testmodel.predict(data3)
                predicted_data = np.array(result)
                dataframe = pd.DataFrame(data=predicted_data,columns=['PredictedData'])
                newdata = pd.concat([data2,dataframe],axis=1,join="outer")
                st.write(data2)
                st.markdown('''##### <span style="color:#FF4B4B">**Your Data With Predicted Value**</span>''', unsafe_allow_html=True)
                st.write(newdata)
                # st.write((testmodel.feature_importances_)*100)

                name = st.text_input('Enter Customer Name:')
                gender = st.radio('Choose customer gender:',('Male','Female'))
                job = st.selectbox('Choose Customer job status:',('Doctor','Engineer','Teacher','Software Developer'))
                email = st.text_input('Enter Customer Email Address:')
                account = st.number_input('Enter Customer Account Number:')
                bankrupt_value  = ''


                if st.button('show detail info',key='showinfo'):
                    if newdata[newdata['Name'] == name] is not None and newdata[newdata['Account']==account] is not None:
                        dff4 = newdata[newdata['Account'] == account]
                        dff5 = int(dff4['PredictedData'].to_numpy())
                        if dff5 == 1:
                            bankrupt_value = 'No Bankrupt(ለኪሳራ አይዳርግም)'
                        elif dff5 == 0:
                            bankrupt_value = 'Bankrupt(ለኪሳራ ይዳርገዋል)'
                        st.write(dff4)
                        st.write('---')
                        fig = plt.figure(figsize=(20, 10))
                        sorted_idx = testmodel.feature_importances_.argsort()[-10:]
                        plt.barh(data3.columns[sorted_idx], testmodel.feature_importances_[sorted_idx])
                        z1,z2 = st.columns(2,gap="small")
                        with z1:

                            st.markdown(
                                '''##### <span style="color:#FF4B4B;text-align:left;">**Customer Financial Status Information**</span>''',
                                unsafe_allow_html=True)
                            info = pd.DataFrame({'Name': name,
                                                 'Gender': gender,
                                                 'Job-Occupation': job,
                                                 'Email-Address': email,
                                                 'Account Number': account,
                                                 'Bankruptcy status': bankrupt_value}, index=['Customer information'])

                            st.write(info.T)
                        with z2:
                            series1 = pd.Series(data3.columns[sorted_idx])
                            series2 = pd.Series((testmodel.feature_importances_[sorted_idx]) * 100)
                            frame1 = pd.DataFrame({'የኪሳራ ሁኔታ ውጤቱን የወሰኑ 10 ጠቃሚ ገጽታዎች': series1,
                                                   'ከላይ በተመለከትናቸው 10 ገጽታዎች የተሸፈነ የበመቶ ዋጋ': series2})
                            st.write(frame1.sort_values(by=['ከላይ በተመለከትናቸው 10 ገጽታዎች የተሸፈነ የበመቶ ዋጋ'], axis=0,
                                                        ascending=False))

                        # plt.xlabel("")
                        st.write('የኪሳራ ሁኔታ ውጤቱን የወሰኑ 10 ጠቃሚ ገጽታዎች')
                        st.pyplot(fig)

                        if dff5 == 1:
                            com.html("""
                             <style>
                             div.box1{
                             display:flex;
                             }
                             div.box{
                             display:flex;
                             flex-direction:column;
                             box-shadow:3px 3px 15px rgba(0,0,0,0.15);
                             padding-right:20px;
                             padding-left:20px;
                             border-radius:10px;
                             padding-top:10px;
                             padding-bottom:10px;
                             background-color:#dbfeda;
                             }
                             .reco{
                             text-align:center;
                             margin-bottom:5px;
                             color:#414441;
                             }
                             .recopara{
                             text-align:justify;
                             color:#414441;
                             }

                             </style>
                             <div class="box1">
                             <div class="box">
                             <h3 class="reco">ምክረ ሀሳብ</h3>
                             <p class="recopara">
                            በጠረጴዛው ና ከላይ በታየው ግራፍ ላይ ባሉት 10 ዋና ዋና ገጽታዎች መሰረት የደንበኛው የኢኮኖሚ ሁኔታ የአቢሲኒያ ባንክን ኢኮኖሚ እንደማይጎዳ ይገልፃል። 
                            ስለሆነም ደንበኛው አሁን ካለው የኢኮኖሚ(ሃብት) ሁኔታ ጋር እንዲቀጥል አሳስባለሁ። 
                            በተለይም ደንበኛው ከላይ በተዘረዘሩት 10 ገጽታዎች(አስፈላጊ ገጽታዎች) ላይ ኩባንያው ከኪሳራ ነፃ እንዲሆን ጥሩ የኢኮኖሚ ሁኔታ እንዲኖረው አበረታታለሁ።
                             </p>
                             </div>

                             </div>
                             """, width=850, height=500)
                        elif dff5 == 0:
                            com.html("""
                             <style>
                             div.box2{
                             display:flex;
                             }
                             div.box3{
                             display:flex;
                             flex-direction:column;
                             box-shadow:3px 3px 15px rgba(0,0,0,0.15);
                             padding-right:20px;
                             padding-left:20px;
                             border-radius:10px;
                             padding-top:10px;
                             padding-bottom:10px;
                             background-color:#f9b7b7;
                             }
                             .reco1{
                             text-align:center;
                             margin-bottom:5px;
                             color:#414441;
                             }
                             .recopara1{
                             text-align:justify;
                             color:#414441;
                             }

                             </style>
                             <div class="box2">
                             <div class="box3">
                             <h3 class="reco1">ምክረ ሀሳብ</h3>
                             <p class="recopara1">
                            በጠረጴዛው ና ከላይ በታየው ግራፍ ላይ ባሉት 10 ዋና ዋና ገጽታዎች መሰረት የደንበኛው የኢኮኖሚ ሁኔታ የአቢሲኒያ ባንክን ኢኮኖሚ እንደሚጎዳ ይገልፃል። 
                            ስለሆነም ደንበኛው አሁን ካለው የኢኮኖሚ(ሃብት) ሁኔታ ጋር እንዲቀጥል አልመክርም። 
                            በተለይም ደንበኛው ከላይ በተዘረዘሩት 10 ገጽታዎች(አስፈላጊ ገጽታዎች) ላይ ኩባንያው ከኪሳራ ነፃ እንዳሆይን ያደርገዋል።
                            የደንበኛችሁ የኢኮኖሚ ሁኔታ የእርስዎ ኩባንያ ላይ ኪሳራ ስለሚያደርስ,ለደንበኛዎ ማስጠንቀቂያ በመስጠት ወደ መልካም የኢኮኖሚ ሁኔታ እንዲመለስ እንዲያደርጉ ስል እመክራለሁ። 
                             </p>
                             </div>

                             </div>
                             """, width=850, height=500)




                    else:
                        st.error('Error')



                if st.sidebar.button('Filter Predicted Data',key = 80):
                    st.markdown('''##### <span style="color:#FF4B4B">**Reliable Persons With Relaible Data For Your Company**</span>''',
                                unsafe_allow_html=True)
                    newdata3 = newdata[newdata['PredictedData'] == 1]
                    st.write(newdata3)
                    st.write('---')
                    st.markdown('''##### <span style="color:#FF4B4B">**Non Realiable Persons with Non Reliable Data For Your Company**</span>''',
                                unsafe_allow_html=True)

                    newdata2 = newdata[newdata['PredictedData'] == 0]
                    st.write(newdata2)


                choice = st.number_input("Choose the number to predict specific data",min_value=0,step=1)
                if choice:
                    if int(dataframe.iloc[int(choice)]) == 1:
                        st.success("These Data is free from Bankruptcy")
                        st.balloons()
                    elif int(dataframe.iloc[int(choice)]) == 0:
                        st.error("These Data makes your company Bankrupt")
                    choice2 = st.number_input('Choose Percentage Value',min_value=0,max_value=125,step=25)
                    if choice2 == 25:
                        if int(np.percentile(dataframe,[int(choice2)])) == 0:
                            st.success("Company Bankruptcy status Good")
                        if int(np.percentile(dataframe,[int(choice2)])) == 1:
                            st.error("Company Bankruptcy status Very Bad")

                    elif choice2 == 50:
                        if int(np.percentile(dataframe,[int(choice2)])) == 0 or int(np.percentile(dataframe,[int(choice2)])) == 1:
                            st.warning("Company Bankruptcy status Intermediate")
                    elif choice2 >= 75:
                        if int(np.percentile(dataframe,[int(choice2)])) == 1:
                            st.success("Company Bankruptcy status very good")
                        if int(np.percentile(dataframe,[int(choice2)])) == 0:
                            st.error("Company Bankruptcy status very good")
                    # elif choice2 == 100:
                    #     if int(np.percentile(dataframe,[int(choice2)])) == 1:
                    #         st.error("Company Bankruptcy status Very Bad")

                    if st.button('Show Predicttion status', key="predict1"):
                        v1,v2 = st.columns(2)
                        with v1:
                            st.markdown('''#### <span style="color:#FF4B4B">**Predicted Value**</span>''',
                                        unsafe_allow_html=True)

                            st.write(dataframe)
                        with v2:
                            st.markdown('''#### <span style="color:#FF4B4B">**Statistical Data For Predicted Value**</span>''',
                                        unsafe_allow_html=True)

                            st.write(dataframe.describe())



                    value = [str(i) for i in result]
                    response = int("".join(value))
                    if response == 0:
                        st.error(f"Hello,Dear {name} , The data you have provided shows that your company will be bankrupt ,"
                                 f"I will recommend you to analyze the data carefully.")
                    elif response == 1:
                        st.balloons()
                        st.success("Your company is in a good condition.")
            elif option == 'LogisticRegression':
                model_log = LogisticRegression()
                x_train1,x_test1,y_train1,y_test1 = train_test_split(X,Y,test_size=0.2,random_state=5678)
                model_log.fit(x_train1,y_train1)
                st.write(f'Score Result: {model_log.score(x_test1,y_test1)*100}%')
                y_preds = model_log.predict(x_test1)
                st.write(f'Accuracy score Result : {accuracy_score(y_test1,y_preds)*100}%')
                c1,c2 = st.columns(2)
                with c1:
                    st.markdown('''#### <span style="color:#FF4B4B">**Predicted Data**</span>''', unsafe_allow_html=True)
                    st.write(np.array(y_preds))
                with c2:
                    st.markdown('''#### <span style="color:#FF4B4B">**Actual True Data**</span>''', unsafe_allow_html=True)
                    st.write(np.array(y_test1))

            elif option == 'SVC':
                model_svc = SVC()
                x_train2,x_test2,y_train2,y_test2 = train_test_split(X,Y,test_size=0.2,random_state=9863)
                model_svc.fit(x_train2,y_train2)
                st.write(f'Score Result: {model_svc.score(x_test2,y_test2)*100}%')
                y_preds = model_svc.predict(x_test2)
                st.write(f'Accuracy score Result : {accuracy_score(y_test2,y_preds)*100}%')
                c1,c2 = st.columns(2)
                with c1:
                    st.markdown('''#### <span style="color:#FF4B4B">**Predicted Data**</span>''', unsafe_allow_html=True)
                    st.write(np.array(y_preds))
                with c2:
                    st.markdown('''#### <span style="color:#FF4B4B">**Actual True Data**</span>''', unsafe_allow_html=True)
                    st.write(np.array(y_test2))
            elif option == 'KNC':
                model_knc = KNeighborsClassifier()
                x_train3,x_test3,y_train3,y_test3 = train_test_split(X,Y,test_size=0.2,random_state=9863)
                model_knc.fit(x_train3,y_train3)
                st.write(f'Score Result: {model_knc.score(x_test3,y_test3)*100}%')
                y_preds = model_knc.predict(x_test3)
                st.write(f'Accuracy score Result : {accuracy_score(y_test3,y_preds)*100}%')
                c1,c2 = st.columns(2)
                with c1:
                    st.markdown('''#### <span style="color:#FF4B4B">**Predicted Data**</span>''', unsafe_allow_html=True)
                    st.write(np.array(y_preds))
                with c2:
                    st.markdown('''#### <span style="color:#FF4B4B">**Actual True Data**</span>''', unsafe_allow_html=True)
                    st.write(np.array(y_test3))
            elif option == "LinearRegression":
                model_lg = LinearRegression()
                x_train3,x_test3,y_train3,y_test3 = train_test_split(X,Y,test_size=0.2,random_state=7865)
                model_lg.fit(x_train3,y_train3)
                st.write(f'Score Result: {model_lg.score(x_test3,y_test3)*100}%')
                y_preds = model_lg.predict(x_test3)
                c1,c2 = st.columns(2)
                with c1:
                    st.markdown('''#### <span style="color:#FF4B4B">**Predicted Data**</span>''', unsafe_allow_html=True)
                    st.write(np.array(y_preds))
                with c2:
                    st.markdown('''#### <span style="color:#FF4B4B">**Actual True Data**</span>''', unsafe_allow_html=True)
                    st.write(np.array(y_test3))
            else:
                st.info("Waiting For Data to be uploaded.")
    elif selected == 'ORS Page':
        st.sidebar.image('https://sp-ao.shortpixel.ai/client/to_auto,q_lossless,ret_img/https://www.bankofabyssinia.com/wp-content/uploads/2020/10/Asset-7@2x.png',use_column_width='auto', width=100)
        st.sidebar.title(f"Welcome {name}")
        com.html("""
        <style>
        .container1{
        display:flex;
        flex-direction:column;
        height:700px;
        background-color:#f2f3f5;
        margin-left:0px;
        
        }
        body{
        width:100%;
        }
        .TextHeading{
        display:flex;
        flex-direction:column;
        align-items:center;
        text-align:center;
        height:200px;
        width:100%;
        
        }
        .accountHead{
            font-size:40px;
            font-family:aerial;
            margin-top:50px;
            margin-bottom:2px;
            font-weight:500;
        }
        .p1{
        font-size:20px;
        font-weight:500;
        color:#6f7070;
        margin-top:0px;
        font-family:calibri;
        }
        .row1{
            width:100%;
            height:200px;
            display:flex;
            flex-direction:row;
        }
        .row2{
            width:100%;
            height:250px;
            display:flex;
            flex-direction:row;
        }
        .row1part1,.row2part1{
        width:33%;
        height:200px;
        display:flex;
        flex-direction:column;
        justify-content:center;
        text-align:center;
        border-right:1px solid #366aa3;
        border-bottom:1px solid #366aa3;
        }
        .row1part2,.row2part2{
        width:33%;
        height:200px;
        display:flex;
        flex-direction:column;
        justify-content:center;
        text-align:center;
        border-right:1px solid #366aa3;
        border-bottom:1px solid #366aa3;
        }
        .row1part3,.row2part3{
        width:33%;
        height:200px;
        display:flex;
        flex-direction:column;
        justify-content:center;
        text-align:center;

        border-right:1px solid #366aa3;
        border-bottom:1px solid #366aa3;
        }
        .img1{
        width:100px;
        height:80px;
        margin-left:90px;
        }
        .img2{
        width:100px;
        height:80px;
        margin-left:90px;
        margin-top:170px;
        margin-bottom:5px;
        }
         .head1,.head2,.head3{
        font-family:calibri;
        font-size:22px;
        font-weight:800;
        margin-bottom:0px;
        margin-top:5px;
        }
        .head4,.head5,.head6{
        font-family:calibri;
        font-size:22px;
        font-weight:800;
        margin-bottom:0px;
        margin-top:5px;
        }
        .row1part1 p,.row1part2 p,.row1part3 p{
        font-family:calibri;
        font-size:17px;
        font-weight:500;
        color:#6f7070;
        margin-top:7px;
        padding-right:2px;
        padding-left:2px;
        }
     .row2part1 p,.row2part2 p,.row2part3 p{
        font-family:calibri;
        font-size:17px;
        font-weight:500;
        color:#6f7070;
        margin-top:7px;
        padding-right:2px;
        padding-left:2px;
        padding-bottom:150px;
        }
        </style>
        <body>
        <div class="container1">
        <div class="TextHeading">
           <h2 class="accountHead">Our Savings Account Benefits<h2>
           <p class="p1">We help businesses and customers achieve more</p>
        </div>
        <div class="row1">
            <div class="row1part1">
               <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAOEAAADhCAMAAAAJbSJIAAABNVBMVEX////v7+/+/v6DgHv/vTEaMj79/PT8wkJ9eXPLy8eNjId9eXbNysmRj4r5+PiAf3oAABPc4OHX1tOShnP9+evPMjSTm5/AwL2YlZIAGCwAAACao6YAITCFj5S5v8EAGy3p7e0sQ00AIDAAJzVJWWGosLMjOkTBxsr+vzvPLzESLTqqp6UAFCUAABoAESlUY2r9uh/KICMAABD74aj65LP6xlf9uBHfn5/PTE3RY2HLGh3WcXLMQUBqdn3t5OTr19j42Yz69N33znb67s75zGf9znT6w0v73Zz85LH6wDD66sPuzMr68fDOVFfXfn3ntrXEnpypXl3ZjY2aa2uzU1GYhIPlwsR7gofirK2gZWHdjo7cm5vGwK+kmYiMgGzb08TWUjfcbDTnhzHzrEMOLD5ATVdOXmTmmYVOAAAQ1UlEQVR4nOWdC0PbRhKAJWHhhFIRuYjURbZlGz/k8zMEMPgVAg0lJHClLebCXe8V5///hJtdPbySJdva9UPiJk0RstbezzM7MzvaVTgupCII6L+gbVCTwM3WJEZvA7aJDh4dYKQIafgQYXQA8R+qhtEQgcpGIyTPne/ZA9LEwfUIpavAhAvuynKETg8R0R4WnD4FbxMtQpqkKzqIVO4iMj6Go7W3CIUJIVojKrhESBdU8tyTSiE6OQmtRAWPcgaKG0UCUaDxhub4iwQgR+UPo1QloawDRSgK0vn7CEUJ2p5GBjAKymDqXwTwpoz3OfoeAQWy9JEqDq5aGBPfCAAyjMLwmyfH5CbMlmGnFOgZI5F0MQ2iSEzsWH1MyAmZZ9fLBhS4w/j1u02WNwi5lzg/asZix++f7exaOIxhae7SVH4jkIicXzZNwgsaJdLcBlmpCIcmn0lI8Q7h9oO2AhHhx+dX8IcROAYsX799dqssSAUCYewwzNqgkcNrkq9cjr1bd48WKQK3e3RMqK9cjpeb75+VCt82m6QC46DC2M/r7tRsmT/4Hh7HSBONgxpjscsFdmQ5Mq3Q4Ty/GScVaPyNxY6W1K/FyfylnF23gWL5sOT+sUuA4Htkj8I4IFqHDLOLlUiQtPk83rSjRNwy2fLuMrvHLsGS/M3DctNCtHQYCzdhwPxQAMaYW86X1jt2oakXXzRdgM1lErJXiwO33vzgViFMn5DIr02R6Dvk1UeGqTLVjW/unVuFaPqEZOuFId9t03bI5xPpVUil/48EoDXJf4tf2dowZKGETHxUbXdJDV59aBLjcDmETBZKMwiPCA0egkY/NJvoJ5IlEDK4GdpK7HuieIEmTcLmxVvLky6ekMWN0gEKpJeZyNUWTsgUJ+gaX5FeZiKRWSyhwLpekKYxOTk0YyApfoS0y0BWX/35hdTgx8mP9yakc9lrqYu8JQHfelzgSUi1jGA9tVRSg8eHXlf46JBiOd1a7nsezgT0JKRzF+u4a3NFOplfvK/xJKT5sDUAOuLg9UefqzwIafu62nEIc973ZDI6GSZMmSSMyhL4zQ9Eqhb3r1i4CdfBR/V5dvEJDLR5OaWs5iKMigK5j8R9mObVtCsdhFHBI2/1xppl3yGIxUkY7ju8Y3l3bdcMm5e7082cIAz/KgRTdo9w9R7zxbzDPCEkYdiXIRgiwBC06r7Ny9kVQwdhBPjwbNAEbMZ/cfdY2n65DfLytWyfihbhJrf58xjw3WQQfGmWDl98b7O4rDTschE3AMux46MLD5W8fGHyeBJGQFC9At1eijU/eE0FI0+4e2kAogHok8REmVAQPpabZQwYP/RN0qJMuImmSvFyGevP12OEhjCw2xawi4E/H6avVgtKuNRVFj5v7nEaLr5q4kz75+kpaHh0GHQGgxf9xo6uzme2CgthICsV0Iq15vHlx3mWVoSEMJjxnx8dH8evduerrYSDMFAdaPPquPzuwmgWFcIghVXh4vL9RZCFP4shZPOwgTR4PtfgI2Q2odAfDG5ABv1Pfh2kes6g1SRw64AfNYXwBcwtbk5O24rewqJ0P9/e9T07S/NMH2HcdqkzGF/CnZ2dv94Cm64QAr+3T9yqpH6QotWcpf+zxYdwZ+fFr11Ep+vw/24bpKu0dITbOuvcePc1iAjmk5mWPwP1JtzZ+LWL8FrK6f3doN/nQfqDm8fbNqZunRKMlDcWjRUWK6gDeRLu/Gbwnd71eYknROJv7rst9NKtZausi7mWLl6EO7/jIXc74D3lrg2MrS5WIyPfKqokHoR/IAW2bvuSNyBoEjOe3TNXxJcAyMum2GcmCW+QN2k/+PLxvMbzT8hUO6yjaPGEwvdWZc1GnCB8wAr0xzPVeNPVu8qpTw6wPpG2dsxgnrBOAeHWxtaY8AbZ3+MsQNBj/xQhLnaBJrt4E25sbW1ZhIMz0OA0CyWk0+piQw2T+BJaOuwjJ4oAJV7TNN6TdDttHd22lNaXNeJ4iCch1qBBKJ3qin4HfKXhQVEsHqQSE4xSoVodWmc7unJ2t06gCfEi3NjYsj3NPSjlBHo+zOUroihW8vk9N6FcqVRGNncbVN4Pk6HOIBy0IAJIvNTLAJ6aU4Eyk3JqUeqpYm7fPjcAwtN1ErllglBAhHa0aEOODbEunQT1FQulwggQqyUHYLoqVrJjaOkRtB4mO3UTCg7CO+jtA/S6AWB5GIGSDAeVA4cSixVR3SZPwMhthygqughRSkFE/C7YKBppaAT2JGySeVGsy4QKU6qoDh3IYNmtx3VzjcVBaOSUFuHOj6BCHeXapRyocA8TFlRRzKXHOIkKmK9MAvLSLZj2kn1NgPmZW4doHFqE3382VIix1ALuftqGNQTsN1dwBRCkxIfl0eF+zp+/koRm1mxb6W8KHoW8tJe3FbeNCG2rlAoZsdKYiJAdHdzpUrXoWwfi/2KKHRpswo2EldfbOvwdXIYx1oAwU7IJjSHJmwM0L7sB+YeWcrZUX+Nf6HoNnd9BE4kfvQg5J+FGW9GfphIiv5Pb80jkussNGFOmaK+tzs9BuPMHGOnAJjSttKQSVlqqimJ2kg/5Gr2zRMApfmYKoTEOScK/gUs0UAo5h6dRrQwmC+AlD0IenPASvem0ctx0Qs5J+DtO2LCukHvByRp2OhlDndJejnA6pGgDSNcHU3rIxDfVjwYihFhxYvY4C1HvAFMdoLRNw2chFIrFSTeDpN9WWjfeXWAtNc2oAwUi7BqxwtQWCnuSlEbD0HA0yM2o7lBoDcRT3TetYdxWOeMLCkK4oShnVvFQE0FfaiOFcraKER6kdMadoRKE4Gp8J8IsO55mfjkBCHde6Eqrb3U5UcFsKO/OpyULOuNto0D4Rddv/fpIr8E5EIMQ/gGEn+w+y42cmgdJfts2bHQINmrOFCU5sf0y4SimPvmFC3obFeYq+AclJPWS2Bv2hntWcCiBjWYNvlKjqGYy+exQlghCpeO5JIThYaZzaT8g4ZnT9JBYv4BPTWJzlVIZVN7ANpyWZhCy1fvnaf76ux1DSEJDviMIjRPkOHSPs/2c6VKlQhLVN4oijNFK0tKwdO81Dhm3Vc600M3d3d2///mDIf/YtcQ88cOf/7TOWBdZSZuHyMi3YjejFZEG05pWQGrMWoSTvpR5W+VMDe7Gr6+J3Qf4+Np5yvmzrNjxcEKFKOM2Ujc8I0ZRUdofJ68QD5XWyUQHlzwrPnc8QmgOuf6X0vKp5SM3MzLDPlIh1qact5MBVFN0zoEFbvk3Bs8nnpcwi/Dfin7rHdHBMqsJ43Bk5XO8hNxN0WiASopEXjqfo2cmDKzD/yjW3MJlo6j4ZIZCuWhPF6UREFaN0zAFVog7NKvZg7J7FA8o/1WMOpRbEnlwKUbyjU3TImzYSTlypWRVeEW7DoXNgPKp7T0QUfHJinxyzkmYlK1hSDqa0O6z+dLSTz3uw+TGDmWCsIoJb1qQtJPvFNZNGjctDzOVwc1U7Iwb18Ab5jiE4zo+utWV9ro7P5dIbfCmbhUO8+R9GDQ3/kr40q/olb4eqqL3NDkBl+hSYinpvA/jjIfYYCW0ZiFENy6miaQr+r1zJILOMmTxyShSScb4NMpVffDB9+vsdpAxD0p05KaoEOUsPiEzFcWSJJVQXlrUjFG4ThUG82pSVzHK3mPHgikI5jS6d5opFvEPFEQeWpM56Sol4GMbHs6U1tNYZx7FJ6mg5kVzgog8UL8NX8o6V5wEjUwdnVhqsg1upuFyrryU6OUzqprL9HB9Ay1V8CkkrkKEwNPsT2CnXWsoJupq0qv4pJUKhZKZr4GNrtHN0OSHA/Ab7YGpxZd7CQ9AQp9PZ+tdp0A1y747I7Q4Y2kUREJ9nffwKScxJ6CX7s08y77ARPWu18L2uTvIJPRlEmR6+uNMxP4pACp9pn+YhL4pnsBQJ/iPaAVwx6/wZtrvA+DpXTYTZVxczNAc916ftgJz0EFLGDsSmwbnqdv7NmWbovU/IzW2H33WQQ++6PAVtE4Yq4V0j0PkFrPCXzhBOyta7fuBy59qEn/XQa+dffa/JzrXJ9jbEah22yxgit3vnOHtJO0vd33JEv7msaOgDRct/ZFNgeZAot5ts5AiwqCjt4x9QK32aef2tvNZOWsZ+0m6J4yp6JiMAnCB+2z6T5+tbU+6rpsHLaXzILHddOEY+DjmHRCuNxucdLrGxrUWUqb++f6BIcabbzruYDhqVtKnm7vHk6enk8eHwSeyQ7TjfQUF8cXINJfmS7CSgv+ihPIpg5Hho3X0oS2IewpdKFtCR7glvC1t5hsO5zmPROY5fJTCOHcJvzx//T1zBUbHW9D1M0ohm2YSsppVFosSmllWlPiok67oANL0NUpulFoZEeF7/knJc89JIhOxaeWZ43ERMlGGpCsagHSP14pUxKZNSaJCSNdRShNdx5dCqYrIOJlI1IGY756Gm4/5+VjT22tpeFUu4eOSzEnjZ19pxoFkHmnoJqAg2yehZaEEP+0W+CahML4WXVLQOPuNZN6+1tkhpnnk7O02jZ/SnFStFVBPMln4tVarVmuvfkokatVXID/1uO031TdvXtWKaY7bf1XDJ/c5qVd7VasVE1zKarGP3i5dw9dm8VcmjOpZiZONC2pvRK5XM5onHF1g3PI0q+1BssBpVbGigSbVIldKpXr5bCqVktKZ7D6SBJdOjgqFQk9VNS6Va+CTGrdXb8haKlMUEqnUMF+BFvh5YPvqQaGw38iLPPxSyuVV+P7gtWyll0oVuAO1h5s77gkv+yl83wzCfAMI80V0ppQ8QD/Sxg98BC9y0iiT5lKZPfNkr4oeVzqqa+ilqmp92H4GXcupdaSmXjKV6eGrG8k0+nGQSXv1kRLQ/GpmtTYIs9l6gZO/WoSCSWi8h0GoFeslLpXcM+1imMumNRhYqAVfVy2t7OcQEp+ro4GsipyYkV2Ebo3Rq3DOgI0Ja6OSWtFchBkxi0SGIzDbYbbeELiUWkTnRgKnjarV+mhPchOq6Fqxijj36ykuVU2RhBXUutgj+ViczFyEBwahkKr25IqTsNhrNBo9jUvnxGJVFQtAkVJH+CS8LqV7xWomq7kI8/ha5KC5UVLm5ExRIgjzB6h5iugk2+bteRqbOpQ4UU2Jbis1BFmpNsqjfiErNSQho3+wpJjcdxHCOJSzeXTVy6rY6/VEPPQ8x6HAvOdprsY2YalaqfgTcnIVLhwTSsUagkrVEbeWdBBychKR9PIVVVUrlQOCMOkgXE0ucmARckPVJKyahCPNiNOGp9nL5XnwpUPNiO695FCS5GwVXAfHq05C5IYkvpZHTkarIKdj67CAmmuGC1xRfesAgj3/EwRmjs+qBmENE5ZqORydRxDFv8HvUra2D44jaaYBcrH+NVut9VBXtVzeIizUcGSp1NJpfASqhGbctxom/Jap4+Zps5i6kmyyBK6EG+J8RO6hzAayFdwbaX8PS5rjjROJXoLTjHN7EO34/WEvhVIXAWK6NTrhPUrGu8p8L2GekdHvPH4Ls7m29nr47A8XiCPB76VpzZdloUt4V8q55//BKotF92Rpsu4BtWQJ+9yaWUJfPWCUZ35DCklUNEhpahHyMFT5b1TUh2W9S/xXIPSLECIidA4x/FVmW6h7uio+1ogUflUwPz4q5Hwc4zw5/BpktNLQ0y3gNnToGRlv2ITeRGfHJN9Xo7HKQmCw0UjstmEs+IcdkLWcGv6kK/QdtGVdG+hXJlTdXGHBfwFCu4E+MkI7nLwb/Q/X6PFODUSFiQAAAABJRU5ErkJggg==" class="img1"/>
               <h4 class="head1"> Earn Interest up to 7%</h4>
               <p>Holds these matters principles selection right some rejects.</p>
            </div>
            <div class="row1part2">
<img src="https://orell.com/images/inner/SMS-Alert-System.png" class="img1"/>
               <h4 class="head2">Free SMS Alerts</h4>
               <p>Business will frequently occur that pleasure have to be repudiated.</p>
               

            </div>
            <div class="row1part3">
<img src="https://static.vecteezy.com/system/resources/previews/000/330/091/original/vector-discount-geometric-banner.jpg" class="img1"/>
               <h4 class="head3">Discounts on Locker</h4>
               <p>The wise man therefore always holds these principle of selection.</p>
            

            </div>
        </div>
                <div class="row2">
            <div class="row2part1">
<img src="https://www.bssbelgium.com/wp-content/uploads/2014/05/MechanischeBeveiligingBlue.png" class="img2"/>
               <h4 class="head4">Secure Banking</h4>
               <p>Holds these matters principles selection right some rejects.</p>
            </div>
            <div class="row2part2">
<img src="https://th.bing.com/th/id/R.5a03da04febc2cc754e43ce19f4c1c12?rik=dHP2rQv5ef0QPg&riu=http%3a%2f%2fwww.pngplay.com%2fwp-content%2fuploads%2f2%2fBank-PNG-HD-Quality.png&ehk=uUhmzbFPOJxi0XacF8kTw8F5IXsu8x0mzQclBeX%2fQk8%3d&risl=&pid=ImgRaw&r=0" class="img2"/>
               <h4 class="head5">Top Banks</h4>
               <p>Business will frequently occur that pleasure have to be repudiated.</p>
               

            </div>
            <div class="row2part3">
<img src="https://th.bing.com/th/id/R.493bae3576994d3f4400d96be81bdeb6?rik=mO6Cqd68aIID4w&riu=http%3a%2f%2fwww.pngall.com%2fwp-content%2fuploads%2f5%2fOnline-Payment-PNG-Image-HD.png&ehk=OVj3rwhuLGMbev3RZd3xvYWtFRi2sA2UYfXgB5yMGXc%3d&risl=&pid=ImgRaw&r=0" class="img2"/>
               <h4 class="head6">Flexible Pay</h4>
               <p>The wise man therefore always holds these principle of selection.</p>
            

            </div>
        </div>
        <div>
        </body>
        
        """,width=860,height = 600)
        st.write('---');
        st.markdown('''<span style="text-align:center;font-size:30px;margin-left:270px;font-weight:700;">Exciting Offers For You</span>''',unsafe_allow_html=True)
        st.markdown('''<span style="margin-left:240px;font-size:20px;">Enjoy exclusive deals & offers with our bank.</span>''',unsafe_allow_html=True);
        com.html('''
        <style>
        .box{
        width:100%;
        height:500px;
        display:flex;
        flex-direction:row;
        justify-content:space-between;
        }
        .b1{
        display:flex;
        flex-direction:column;
        width:32%;
        
        }
        .b2{
        display:flex;
        width:32%;
        box-shadow:3px 3px 10px rgba(0,0,0,0.25);
        }
        .b3{
        display:flex;
        flex-direction:column;
        width:32%;
        }
        .bb1{
        width:100%;
        height:47%;
        box-shadow:2px 2px 10px rgba(0,0,0,0.25);
        margin-bottom:30px;
        }
        .bb2{
        width:100%;
        height:47%;
        box-shadow:2px 2px 10px rgba(0,0,0,0.25);
        }
        .bc1{
        width:100%;
        height:47%;
        box-shadow:2px 2px 10px rgba(0,0,0,0.25);
        margin-bottom:30px;
        }
        .bc2{
        width:100%;
        height:47%;
        box-shadow:2px 2px 10px rgba(0,0,0,0.25);
        }
        .p1{
        display:flex;
        flex-direction:row;
        margin-top:10px;
        font-size:17px;
        font-family:calibri;
        margin-right:5px;
        }
        .logo1{
        width:130px;
        height:40px;
        margin-left:10px;
        margin-top:10px;
        }
        .p2 p{
        color:green;
        font-family:calibri;
        font-size:17px;
        margin-left:15px;
        }
        .p3 p{
        font-family:calibri;
        font-size:20px;
        font-weight:600;
        margin-left:15px;
        }
        .span1{
        display:flex;
        flex-direction:row;
        }
        
        .span2{
        display:flex;
        flex-direction:row;
        margin-left:25px;
        margin-top:10px;
        }
        .span1 p{
        margin-top:7px;
        font-family:calibri;
        font-weight:600;
        font-size:18px;
        }
        .span2 p{
        margin-top:1px;
        font-family:calibri;
        font-weight:600;
        font-size:18px;
        margin-left:10px;
        }
        .p4{
        display:flex;
        flex-direction:row;
        }
        .b2{
        display:flex;
        flex-direction:column;
        align-items:center;
        
        }
        .centerIMG{
        margin-top:40px;
        }
        .sub1{
        font-size:22px;
        font-weight:600;
        font-family:calibri;
        }
        .sub2{
        font-family:calibri;
        font-size:19px;
        margin-top:5px;
        
        }
        .email{
        height:30px;
        width:80%;
        margin-left:15px;
        font-size:18px;
        font-family:calibri;
        margin-right:20px;
        
        }
        
        .subscribe{
        height:50px;
        width:100px;
        margin-top:20px;
        text-align:center;
        font-size:20px;
        font-family:Calibri;
        font-weight:600;
        background-color:green;
        color:white;
        border:1px solid green;
        }
        </style>
        <div class="box">
            <div class="b1">
                <div class="bb1">
                    <div class="p1">
                    <img src="https://st.ourhtmldemo.com/new/finbank-demo/assets/images/resources/offer-logo-1.png" class="logo1"/>
                    <p>Till: 25th Jun’22</p>
                    </div>
                    <div class="p2">
                    <p>Medical &nbsp&nbsp ______</p>
                    </div>
                    <div class="p3">
                    <p>Get 10% Cashback on Xfinity Restuarant.</p>
                    </div>
                    <div class="p4">
                    <span class="span1"><svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" fill="green" class="bi bi-arrow-right-short" viewBox="0 0 16 16">
                      <path fill-rule="evenodd" d="M4 8a.5.5 0 0 1 .5-.5h5.793L8.146 5.354a.5.5 0 1 1 .708-.708l3 3a.5.5 0 0 1 0 .708l-3 3a.5.5 0 0 1-.708-.708L10.293 8.5H4.5A.5.5 0 0 1 4 8z"/>
                    </svg><p>Know More</p></span>

                    <span class="span2"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-share" viewBox="0 0 16 16">
                      <path d="M13.5 1a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3zM11 2.5a2.5 2.5 0 1 1 .603 1.628l-6.718 3.12a2.499 2.499 0 0 1 0 1.504l6.718 3.12a2.5 2.5 0 1 1-.488.876l-6.718-3.12a2.5 2.5 0 1 1 0-3.256l6.718-3.12A2.5 2.5 0 0 1 11 2.5zm-8.5 4a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3zm11 5.5a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3z"/>
                    </svg><p>Share</p></span>
                    </div>
                </div>
                <div class="bb2">
                
                                  <div class="p1">
                    <img src="https://st.ourhtmldemo.com/new/finbank-demo/assets/images/resources/offer-logo-2.png" class="logo1"/>
                    <p>Till: 25th Jun’22</p>
                    </div>
                    <div class="p2">
                    <p>Medical &nbsp&nbsp ______</p>
                    </div>
                    <div class="p3">
                    <p>Get 10% Cashback on Xfinity Restuarant.</p>
                    </div>
                    <div class="p4">
                    <span class="span1"><svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" fill="green" class="bi bi-arrow-right-short" viewBox="0 0 16 16">
                      <path fill-rule="evenodd" d="M4 8a.5.5 0 0 1 .5-.5h5.793L8.146 5.354a.5.5 0 1 1 .708-.708l3 3a.5.5 0 0 1 0 .708l-3 3a.5.5 0 0 1-.708-.708L10.293 8.5H4.5A.5.5 0 0 1 4 8z"/>
                    </svg><p>Know More</p></span>

                    <span class="span2"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-share" viewBox="0 0 16 16">
                      <path d="M13.5 1a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3zM11 2.5a2.5 2.5 0 1 1 .603 1.628l-6.718 3.12a2.499 2.499 0 0 1 0 1.504l6.718 3.12a2.5 2.5 0 1 1-.488.876l-6.718-3.12a2.5 2.5 0 1 1 0-3.256l6.718-3.12A2.5 2.5 0 0 1 11 2.5zm-8.5 4a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3zm11 5.5a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3z"/>
                    </svg><p>Share</p></span>
                    </div>
                </div>
            </div>
            <div class="b2">
            <img src="https://st.ourhtmldemo.com/new/finbank-demo/assets/images/icon/subscribe-1.png" class="centerIMG"/>
            <p class="sub1">Subscribe Us</p>
            <p class="sub2">Subscribe us & Stay updated.</p>
            <input type="email" placeholder="Email Address" class="email"/>
            <input type = "Submit" value="Subscribe" class="subscribe"/>
            </div>
            <div class="b3">
                <div class="bc1">
                                  <div class="p1">
                    <img src="https://st.ourhtmldemo.com/new/finbank-demo/assets/images/resources/offer-logo-3.png" class="logo1"/>
                    <p>Till: 25th Jun’22</p>
                    </div>
                    <div class="p2">
                    <p>Medical &nbsp&nbsp ______</p>
                    </div>
                    <div class="p3">
                    <p>Get 10% Cashback on Xfinity Restuarant.</p>
                    </div>
                    <div class="p4">
                    <span class="span1"><svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" fill="green" class="bi bi-arrow-right-short" viewBox="0 0 16 16">
                      <path fill-rule="evenodd" d="M4 8a.5.5 0 0 1 .5-.5h5.793L8.146 5.354a.5.5 0 1 1 .708-.708l3 3a.5.5 0 0 1 0 .708l-3 3a.5.5 0 0 1-.708-.708L10.293 8.5H4.5A.5.5 0 0 1 4 8z"/>
                    </svg><p>Know More</p></span>

                    <span class="span2"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-share" viewBox="0 0 16 16">
                      <path d="M13.5 1a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3zM11 2.5a2.5 2.5 0 1 1 .603 1.628l-6.718 3.12a2.499 2.499 0 0 1 0 1.504l6.718 3.12a2.5 2.5 0 1 1-.488.876l-6.718-3.12a2.5 2.5 0 1 1 0-3.256l6.718-3.12A2.5 2.5 0 0 1 11 2.5zm-8.5 4a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3zm11 5.5a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3z"/>
                    </svg><p>Share</p></span>
                    </div>
                </div>
                <div class="bc2">
                                  <div class="p1">
                    <img src="https://st.ourhtmldemo.com/new/finbank-demo/assets/images/resources/offer-logo-1.png" class="logo1"/>
                    <p>Till: 25th Jun’22</p>
                    </div>
                    <div class="p2">
                    <p>Medical &nbsp&nbsp ______</p>
                    </div>
                    <div class="p3">
                    <p>Get 10% Cashback on Xfinity Restuarant.</p>
                    </div>
                    <div class="p4">
                    <span class="span1"><svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" fill="green" class="bi bi-arrow-right-short" viewBox="0 0 16 16">
                      <path fill-rule="evenodd" d="M4 8a.5.5 0 0 1 .5-.5h5.793L8.146 5.354a.5.5 0 1 1 .708-.708l3 3a.5.5 0 0 1 0 .708l-3 3a.5.5 0 0 1-.708-.708L10.293 8.5H4.5A.5.5 0 0 1 4 8z"/>
                    </svg><p>Know More</p></span>

                    <span class="span2"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-share" viewBox="0 0 16 16">
                      <path d="M13.5 1a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3zM11 2.5a2.5 2.5 0 1 1 .603 1.628l-6.718 3.12a2.499 2.499 0 0 1 0 1.504l6.718 3.12a2.5 2.5 0 1 1-.488.876l-6.718-3.12a2.5 2.5 0 1 1 0-3.256l6.718-3.12A2.5 2.5 0 0 1 11 2.5zm-8.5 4a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3zm11 5.5a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3z"/>
                    </svg><p>Share</p></span>
                    </div>
                </div>
            </div>
        </div>
        ''',width=860,height=520)
        if st.sidebar.button('Face detector camera',key="camera1"):
            detectedFace = cv.CascadeClassifier("Resources/haarcascades/haarcascade_frontalface_default.xml");
            capture = cv.VideoCapture(0);

            while True:
                img_counter = 0;
                istrue, frame = capture.read();
                grayFrame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY);
                faces = detectedFace.detectMultiScale(grayFrame, 1.1, 4);
                for (x, y, w, h) in faces:
                    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2);
                    # cv2.cv2.putText(frame, "Birhan Face", (250, 150), cv2.cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2);

                cv.imshow("Birhan Camera Detecting Face", frame);

                if cv.waitKey(1) & 0XFF == ord('q'):
                    break;
                if cv.waitKey(1) & 0XFF == ord('t'):
                    cv.cv2.imwrite('C:/Users/GL/PycharmProjects/VirtualAssistant/OpencvPhotos',frame);
                    print("Image Taken");

            capture.release();


        if st.sidebar.button("Take a picture of your id",key="IDpicture"):
            id_photo = st.camera_input('Take a picture of your id card:')
            if id_photo:
                st.image(id_photo);

        if st.sidebar.button("Application form site",key="application"):
            webbrowser.open_new_tab('http://localhost/php/AbyssiniyaBankOnlineRegistrationPage.html');
        if st.sidebar.button("BankofAbyssinia admin dashboard",key="app"):
            webbrowser.open_new_tab('http://localhost/php/index.php');








