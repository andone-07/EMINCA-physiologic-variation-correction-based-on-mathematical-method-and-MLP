import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from kneed import KneeLocator

style.use('seaborn-whitegrid')

x = np.arange(1, 3, 0.01)*np.pi
y = np.cos(x)

# 计算各种参数组合下的拐点
kneedle_cov_inc = KneeLocator(x,
                      y,
                      curve='convex',
                      direction='increasing',
                      online=True)

kneedle_cov_dec = KneeLocator(x,
                      y,
                      curve='convex',
                      direction='decreasing',
                      online=True)

kneedle_con_inc = KneeLocator(x,
                      y,
                      curve='concave',
                      direction='increasing',
                      online=True)

kneedle_con_dec = KneeLocator(x,
                      y,
                      curve='concave',
                      direction='decreasing',
                      online=True)


fig, axe = plt.subplots(2, 2, figsize=[12, 12])

axe[0, 0].plot(x, y, 'k--')
axe[0, 0].annotate(text='Knee Point', xy=(kneedle_cov_inc.knee+0.2, kneedle_cov_inc.knee_y), fontsize=10)
axe[0, 0].scatter(x=kneedle_cov_inc.knee, y=kneedle_cov_inc.knee_y, c='b', s=200, marker='^', alpha=1)
axe[0, 0].set_title('convex+increasing')
axe[0, 0].fill_between(np.arange(1, 1.5, 0.01)*np.pi, np.cos(np.arange(1, 1.5, 0.01)*np.pi), 1, alpha=0.5, color='red')
axe[0, 0].set_ylim(-1, 1)

axe[0, 1].plot(x, y, 'k--')
axe[0, 1].annotate(text='Knee Point', xy=(kneedle_cov_dec.knee+0.2, kneedle_cov_dec.knee_y), fontsize=10)
axe[0, 1].scatter(x=kneedle_cov_dec.knee, y=kneedle_cov_dec.knee_y, c='b', s=200, marker='^', alpha=1)
axe[0, 1].fill_between(np.arange(2.5, 3, 0.01)*np.pi, np.cos(np.arange(2.5, 3, 0.01)*np.pi), 1, alpha=0.5, color='red')
axe[0, 1].set_title('convex+decreasing')
axe[0, 1].set_ylim(-1, 1)

axe[1, 0].plot(x, y, 'k--')
axe[1, 0].annotate(text='Knee Point', xy=(kneedle_con_inc.knee+0.2, kneedle_con_inc.knee_y), fontsize=10)
axe[1, 0].scatter(x=kneedle_con_inc.knee, y=kneedle_con_inc.knee_y, c='b', s=200, marker='^', alpha=1)
axe[1, 0].fill_between(np.arange(1.5, 2, 0.01)*np.pi, np.cos(np.arange(1.5, 2, 0.01)*np.pi), 1, alpha=0.5, color='red')
axe[1, 0].set_title('concave+increasing')
axe[1, 0].set_ylim(-1, 1)

axe[1, 1].plot(x, y, 'k--')
axe[1, 1].annotate(text='Knee Point', xy=(kneedle_con_dec.knee+0.2, kneedle_con_dec.knee_y), fontsize=10)
axe[1, 1].scatter(x=kneedle_con_dec.knee, y=kneedle_con_dec.knee_y, c='b', s=200, marker='^', alpha=1)
axe[1, 1].fill_between(np.arange(2, 2.5, 0.01)*np.pi, np.cos(np.arange(2, 2.5, 0.01)*np.pi), 1, alpha=0.5, color='red')
axe[1, 1].set_title('concave+decreasing')
axe[1, 1].set_ylim(-1, 1)

# 导出图像
plt.savefig('图2.png', dpi=300)

x = np.arange(0, 6, 0.01)*np.pi
y = np.cos(x)

# 计算convex+increasing参数组合下的拐点
kneedle = KneeLocator(x,
                      y,
                      curve='convex',
                      direction='increasing',
                      online=True)

fig, axe = plt.subplots(figsize=[8, 4])

axe.plot(x, y, 'k--')
axe.annotate(text='Knee Point', xy=(kneedle.knee+0.2, kneedle.knee_y), fontsize=10)
axe.set_title('convex+increasing')
axe.fill_between(np.arange(1, 1.5, 0.01)*np.pi, np.cos(np.arange(1, 1.5, 0.01)*np.pi), 1, alpha=0.5, color='red')
axe.fill_between(np.arange(3, 3.5, 0.01)*np.pi, np.cos(np.arange(3, 3.5, 0.01)*np.pi), 1, alpha=0.5, color='red')
axe.fill_between(np.arange(5, 5.5, 0.01)*np.pi, np.cos(np.arange(5, 5.5, 0.01)*np.pi), 1, alpha=0.5, color='red')
axe.scatter(x=list(kneedle.all_knees), y=np.cos(list(kneedle.all_knees)), c='b', s=200, marker='^', alpha=1)
axe.set_ylim(-1, 1)

# 导出图像
plt.savefig('图3.png', dpi=300)