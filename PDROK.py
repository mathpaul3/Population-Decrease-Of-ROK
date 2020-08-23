import numpy as np
import matplotlib.pyplot as plt


x = np.array([1970,1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019])
B = np.array([1006645, 1024773, 952780, 965521, 922823, 874030, 796331, 825339, 750728, 862669, 862835, 867409, 848312, 769155, 674793, 655489, 636019, 623831, 633092, 639431, 649738, 709275, 730678, 715826, 721185, 715020, 691226, 675394, 641594, 620668, 640089, 559934, 496911, 495036, 476958, 438707, 451759, 496822, 465892, 444849, 470171, 471265, 484550, 436455, 435435, 438420, 406243, 357771, 326822, 303100])
D = np.array([258589, 237528, 210071, 267460, 248807, 270657, 266857, 249254, 252298, 239986, 277284, 237481, 245767, 254563, 236445, 240418, 239256, 243504, 235779, 236818, 241616, 242270, 236162, 234257, 242439, 242838, 241149, 244693, 245825, 247734, 248740, 243813, 247524, 246463, 246220, 245874, 244162, 246482, 246113, 246942, 255405, 257396, 267221, 266257, 267692, 275895, 280827, 285534, 298820, 295100])

for_num = 10 ** 5
w = -180000
b = 39200000
learning_rate_b = 0.00000001

wd = 2500
bd = -4492000
learning_rate_d = 0.0000000007

print(w, "\n", b)
print(wd, "\n", bd)

# Death Function
for i in range(for_num):
    D_predict = wd * x + bd
    error = np.square(D_predict - D).mean()
    
    wd = wd - learning_rate_d * ((D_predict - D) * x).mean()
    bd = bd - learning_rate_d * (D_predict - D).mean()
    
    if i % 10 == 0:
        print("error = {}, w = {}, b = {}".format(error, wd, bd))

print("\nresult : h(x) = ({0:.3f})x + ({1:.3f})".format(wd, bd))

plt.scatter(x, D, color = 'r')
plt.plot(x, wd * x + bd, color = 'b')
plt.show()
print("\nresult : h(x) = ({0:.3f})x + ({1:.3f})".format(wd, bd))

# # Birth Function
# for i in range(for_num):
#     B_predict = w * x + b
#     error = np.square(B_predict - B).mean()
    
#     w = w - learning_rate_b * ((B_predict - B) * x).mean()
#     b = b - learning_rate_b * (B_predict - B).mean()
    
#     if i % 10 == 0:
#         print("error = {}, w = {}, b = {}".format(error, w, b))

# print("\nresult : h(x) = ({0:.3f})x + ({1:.3f})".format(w, b))

# plt.scatter(x, B, color = 'r')
# plt.plot(x, w * x + b, color = 'b')
# plt.show()
# print("\nresult : h(x) = ({0:.3f})x + ({1:.3f})".format(w, b))
