### PREV ###
prev_money_cols = ['AMT_ANNUITY', 'AMT_APPLICATION',
                   'AMT_CREDIT', 'AMT_DOWN_PAYMENT', 'AMT_GOODS_PRICE']
prev_rate_cols = ['RATE_DOWN_PAYMENT',
                  'RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED']
prev_other_cols = ['HOUR_APPR_PROCESS_START', 'SELLERPLACE_AREA']
prev_cnt_cols = ['CNT_PAYMENT']
prev_day_cols = ['DAYS_DECISION', 'DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE',
                 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION']

# các cột dùng để tính toán với các bảng khác
prev_use_cols = ["SK_ID_PREV", "SK_ID_CURR", "AMT_ANNUITY",
                 "AMT_APPLICATION", "AMT_CREDIT", "DAYS_DECISION", "CNT_PAYMENT"]
