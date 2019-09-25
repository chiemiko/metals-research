import pandas as pd
import csv

x = 13
print("It's working if we get 13*2 = ", x*2)


d = {'col1': [1, 2], 'col2': [3, 4]}
df = pd.DataFrame(data=d)

df.to_csv("Wassup.csv")


# open a (new) file to write
outF = open("Wassup.txt", "w")
