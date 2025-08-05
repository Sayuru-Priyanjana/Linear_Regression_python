import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('linear_dataset.csv')

# plt.scatter(df['x'], df['y'])
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

def Gradient_Desent(m_now, b_now, points, L):
    n = len(points)
    m_gradient =0
    b_gradient =0
    for i in range(n):
        y = points.iloc[i]['y']
        x = points.iloc[i]['x']

        m_gradient += -(2/n) * x * (y-(m_now*x+b_now))
        b_gradient += -(2/n)  * (y-(m_now*x+b_now))

    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return m, b

m=0
b=0
L=0.001
epochs = 100
for i in range(epochs):
    m,b = Gradient_Desent(m, b, df, L)
    print(m, b)
    i+=1

plt.scatter(df.x, df.y , color='black')
plt.plot(list(range(0,50)), [m*x + b for x in df.x], color='red')
plt.show()



