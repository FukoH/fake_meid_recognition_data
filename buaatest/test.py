import pandas as pd
import random
import csv
# df = pd.read_csv('./userdminfo_d_20161201.TXT',sep='|',names=['period','IMEI','number','imsi','province','city','manu','type','soft','timestamp'])
df = pd.read_table(u'userdminfo_d_20161201.TXT',
                   header=None,sep='|',
                   index_col =9,
                   names=['date','meid','mdn','imsi','branch','city','manufactor','phone_model','software_version','id','pid','uuid','none'],
                    parse_dates =True).drop(['date','id','pid','uuid','none'],axis = 1).astype(str)
# with open('log_20161201.txt',mode='w',encoding= 'utf-8') as f:
for index, row in df.iterrows():
    l = ['[' + df.index[0].strftime('%Y%m%d%H%M%S') + '.' + '{:06}'.format(random.randint(0, 999999)) + ']',
        'SmgpRecv:[DELIVER]', df.index[0].strftime('%Y%m%d%H%M%S'), row['mdn']]

    l2 = ['<a1><b1>', row['phone_model'], '</b1><b2>', row['meid'], '</b2><b3>', row['imsi'], '</b3><b4>',
        row['software_version'], '</a1><a2><b5>', row['branch'], '</b5><b6>', row['city'], '</b6></a2><a3>',
        row['manufactor'], '</a3>']
    df.loc[index,'result'] = ','.join(l)+''.join(l2)
    # f.writelines(','.join(l)+''.join(l2))
df = df['result']
df.to_csv('log_20161201_full.txt',encoding='utf-8',index=False)
