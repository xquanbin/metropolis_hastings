# -*- coding: utf-8 -*-
# Author : xiequanbin
# Date:  18-06-04
# Email:  xquanbin072095@gmail.com

import os
import sys
import itertools
import numpy as np
import pandas as pd


# read several .xls files under the given path
def read_several_xls(path, header, skiprows, names):
    df = pd.DataFrame()
    dirs = os.listdir(path)
    for i in dirs:
        if os.path.splitext(i)[1] == ".xls":
            sub_df = pd.read_excel(path + '/' + i, header=header, skiprows=skiprows, names=names)
            df = pd.concat([df, sub_df])

    return df


# remove the stock which has been suspended longer than half of trading days or
# suspended continuously longer than 60 days.
def filter_stocks(df):
    columns = df.columns
    drop_columns = []
    for stk in columns:
        series = df[stk]
        miss_day = list(np.isnan(series).astype(int))
        miss_day_count = np.sum(miss_day)
        max_ctn_miss_day_count = np.max([len(list(v)) for k, v in itertools.groupby(miss_day) if k == 1])
        drop_or_not = 1 if (miss_day_count * 2 >= len(miss_day)) or (max_ctn_miss_day_count > 60) else 0

        if drop_or_not:
            drop_columns.append(stk)

    filtered_df = df.drop(drop_columns, axis=1)

    return filtered_df


if __name__ == "__main__":

    reload(sys)
    sys.setdefaultencoding('utf8')

    INPUT_PATH = "./input"
    MIDDLE_PATH = "./intermediate"
    hs300_path = INPUT_PATH + '/hs300'
    thr_factors_path = INPUT_PATH + '/factors/thr_factors'
    five_factors_path = INPUT_PATH + '/factors/five_factors'
    start_date = "2008-01-01"

    # daily trading data of stocks in hs300
    hs300 = read_several_xls(hs300_path, header=0, skiprows=2,
                             names=['symbol', 'trading_date', 'cls_price', 'mkt_value', 'daily_return'])
    hs300['daily_return'] = hs300['daily_return'].apply(lambda x: np.log(1 + x))

    # industry class for each stock in the hs300
    ind_class = pd.read_excel(INPUT_PATH + "/industry_class.xlsx", header=0, skiprows=0,
                              names=['symbol', 'security_name', "short_name", "class_name", 'list_date'])
    ind_class['class_name'] = ind_class['class_name'].apply(lambda x: x[-2: ])
    ind_class = ind_class[ind_class.list_date <= start_date]

    # match each stock to its industry class
    matched_hs300 = pd.merge(ind_class, hs300, how="left", on='symbol')

    # standard trading date in the Chinese A-share market
    stk_index =pd.read_excel(INPUT_PATH + "/index_trading_date(1990-2017).xls", header=0, skiprows=2,
                             names=['mkt_type', 'trading_date', "daily_return", 'company_num'])
    A_shares = stk_index[stk_index.mkt_type == 5]
    std_trading_date = A_shares.drop(['mkt_type', "daily_return", 'company_num'], axis=1)
    std_trading_date = std_trading_date[std_trading_date.trading_date >= start_date].sort_values(by='trading_date')
    np.savetxt(MIDDLE_PATH + '/trading_date.txt', np.array(std_trading_date), fmt='%s')

    # merge all trading data
    data = pd.merge(std_trading_date, matched_hs300, how='left', on='trading_date').set_index(['trading_date', 'symbol'])
    ind_class_name = {u'能源': 'energy', u'材料': 'material', u'工业': 'industrial', u'可选': 'consDisc',
                      u'消费': 'consStap', u'医药': 'healthCare', u'银行': 'bank', u'金融': 'financial',
                      u'信息': 'technology', u'电信': 'telService', u'公用': 'utilities', 'all': 'all'}

    if not os.path.exists(MIDDLE_PATH):
        os.mkdir(MIDDLE_PATH)
    save_file = ['mkt_value', 'cls_price', 'daily_return', 'stock_info']
    for u in save_file:
        SAVE_PATH = MIDDLE_PATH + '/' + u
        if not os.path.exists(SAVE_PATH):
            os.mkdir(SAVE_PATH)

    # get cls price, daily return, mkt value of stocks in each industry of hs300 under the standard trading date
    for v in ind_class_name.keys():

        MKT_VALUE_SAVE_PATH = MIDDLE_PATH + '/' + 'mkt_value'
        if v == 'all':
            sub_mkt_value = data[['mkt_value']]
        else:
            sub_mkt_value = data[data.class_name == v][['mkt_value']]
        filtered_sub_mkt_value = filter_stocks(sub_mkt_value.unstack())
        filtered_sub_mkt_value.columns = filtered_sub_mkt_value.columns.droplevel(0)

        for u in save_file[1:]:

            SAVE_PATH = MIDDLE_PATH + '/' + u
            if u != 'stock_info':
                if v == 'all':
                    sub_data = data[[u]]
                else:
                    sub_data = data[data.class_name == v][[u]]

                filtered_sub_data = filter_stocks(sub_data.unstack())
                filtered_sub_data.columns = filtered_sub_data.columns.droplevel(0)
                # replace nan by value-weighted mean
                temp_sum = filtered_sub_data.T * filtered_sub_mkt_value.T
                nan_add_T = temp_sum.sum() * 1. / filtered_sub_mkt_value.T.sum() * np.isnan(filtered_sub_data.T)
                filtered_sub_data = nan_add_T.T + filtered_sub_data.fillna(0)
                # transfer df to array and save it
                filtered_sub_data_array = np.array(filtered_sub_data)
                np.savetxt(SAVE_PATH + '/' + ind_class_name[v] + '.txt', filtered_sub_data_array)
            else:
                sub_stock_info = ind_class[ind_class.symbol.isin(filtered_sub_data.columns)]
                sub_stock_info.index = range(0, len(sub_stock_info))
                sub_stock_info['symbol'] = sub_stock_info.symbol.apply(lambda x: '0' * (6 - len(str(x))) + str(x))
                sub_stock_info_array = np.array(sub_stock_info)
                np.savetxt(SAVE_PATH + '/' + ind_class_name[v] + '_info.txt', sub_stock_info_array, fmt='%s', encoding='utf-8')

        # replace nan in mkt_value by equal-weighted mean
        filtered_sub_mkt_value = filtered_sub_mkt_value.apply(lambda df: df.fillna(df.mean()))
        filtered_sub_mkt_value_array = np.array(filtered_sub_mkt_value)
        np.savetxt(MKT_VALUE_SAVE_PATH + '/' + ind_class_name[v] + '.txt', filtered_sub_mkt_value_array)

    # three factors and five factors
    original_thr_factors = read_several_xls(thr_factors_path, header=0, skiprows=2,
                                            names=['mkt_type', 'trading_date', 'risk_premium', 'SMB', 'HML'])
    original_five_factors = read_several_xls(five_factors_path, header=0, skiprows=2,
                                             names=['mkt_type', 'trading_date', 'risk_premium','SMB', 'HML',
                                                    'RMW', 'CMA'])
    thr_factors = original_thr_factors[original_thr_factors.mkt_type == 'P9709'].drop('mkt_type', axis=1)
    five_factors = original_five_factors[original_five_factors.mkt_type == 'P9709'].drop('mkt_type', axis=1)
    thr_factors = thr_factors.sort_values(by='trading_date').set_index('trading_date')
    five_factors = five_factors.sort_values(by='trading_date').set_index('trading_date')
    thr_factors_array = np.array(thr_factors)
    five_factors_array = np.array(five_factors)

    FACTORS_SAVE_PATH = MIDDLE_PATH + '/' + 'factors'
    if not os.path.exists(FACTORS_SAVE_PATH):
        os.mkdir(FACTORS_SAVE_PATH)

    np.savetxt(FACTORS_SAVE_PATH + '/' + 'thr_factors.txt', thr_factors_array)
    np.savetxt(FACTORS_SAVE_PATH + '/' + 'five_factors.txt', five_factors_array)



