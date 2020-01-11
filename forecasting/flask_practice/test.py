import datetime
import pandas as pd

#print("Cron job has run at %s" %datetime.datetime.now())

list1 = [2, 3, 4]
list2 = [4, 7, 10]
indices = [0, 1, 2]
test = pd.DataFrame({'list1': list1, 'list2': list2})
test.to_csv('../usr/bin/python/test.csv', index=True)