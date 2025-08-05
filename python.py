import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('linear_dataset.csv')


def Gradient_Descent(m_now, b_now, points, L):
    n = len(points)
    m_gradient = 0
    b_gradient = 0
    for i in range(n):
        x = points.iloc[i]['x']
        y = points.iloc[i]['y']
        m_gradient += -(2 / n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2 / n) * (y - (m_now * x + b_now))
    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return m, b


m = 0
b = 0
L = 0.01
epochs = 3000


for i in range(epochs):

    m, b = Gradient_Descent(m, b, df, L)

    if i % 50 == 0:
        print(f"Epoch {i}: m = {m:.4f}, b = {b:.4f}")

print(f"Final parameters: m = {m:.4f}, b = {b:.4f}")

plt.scatter(df.x, df.y, color='black')
plt.plot(df.x, [m * x + b for x in df.x], color='red')  
plt.xlabel('x')
plt.ylabel('y')
plt.show()