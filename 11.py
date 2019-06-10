import numpy as np
import xlrd

from sklearn.linear_model import LinearRegression

y1 = []
y2 = []
X = []

filename = r"C:\Users\Administrator\Desktop\12.xlsx"
data = xlrd.open_workbook(filename)
table = data.sheets()[4]
for i in range(table.nrows):
    f = table.row_values(i)
    y1.append(f[0]*100)
    y2.append(f[1]*100)
    X.append(f[2:])

X = np.asarray(X, dtype=np.float64)
y1 = np.asarray(y1, dtype=np.float64)
y2 = np.asarray(y2, dtype=np.float64)

reg = LinearRegression(n_jobs=True)
reg.fit(X, y1)

print("一面\t" + "\t".join([str(r) for r in reg.coef_]))
print("一面截距\t" + str(reg.intercept_))


reg2 = LinearRegression(n_jobs=True)
reg2.fit(X, y2)

print("二面\t" + " ".join([str(r) for r in reg2.coef_]))
print("二面截距\t" + str(reg2.intercept_))
