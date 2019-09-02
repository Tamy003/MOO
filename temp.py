import pandas as pd
import matplotlib as mpl
import numpy as np
import matplotlib.pylab as plt
from pandas import DataFrame
mpl.rcParams['font.sans-serif']=[u'SimHei']
mpl.rcParams['axes.unicode_minus']=False
df=pd.read_excel(r'D:\大三下\数据挖掘\第二次平时作业和数据（聚类分析）\security.xlsx',header=0)
da=df.dropna(how='any',axis=1)
#数据集X
X=da.values
X=X[:520,2:]
#无量纲化
from sklearn.preprocessing import MinMaxScaler
X= np.array(X)
X= MinMaxScaler().fit_transform(X)
#2000年的样本数据
da2000=da[da['年份'].isin([2000])]
da2000=da2000.values
da2000=da2000[:130,2:]
#2001年的样本数据
da2001=da[da['年份'].isin([2001])]
da2001=da2001.values
da2001=da2001[:130,2:]
#2002年的样本数据
da2002=da[da['年份'].isin([2002])]
da2002=da2002.values
da2002=da2002[:130,2:]
#2003年的样本数据
da2003=da[da['年份'].isin([2003])]
da2003=da2003.values
da2003=da2003[:130,2:]




#样本系统聚类
from scipy import spatial
from scipy.cluster.hierarchy import dendrogram,linkage
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
#2000年
dist0=spatial.distance.pdist(da2000,metric='euclidean') #pdist计算矩阵间向量的距离
z0=linkage(dist0,method='complete')#基于最大距离的系统聚类方法
dendrogram(z0,get_leaves=False,show_leaf_counts=False)#系统聚类的系谱图
scores0=[]
for n in range(2,8):
    clustering=AgglomerativeClustering(n_clusters=n).fit(da2000)
    a0=clustering.labels_
    scores0.append(silhouette_score(da2000,a0))
print(scores0)
plt.plot(scores0)
plt.show()
#2001年
dist1=spatial.distance.pdist(da2001,metric='euclidean') #pdist计算矩阵间向量的距离
z1=linkage(dist1,method='complete')#基于最大距离的系统聚类方法
dendrogram(z1,get_leaves=False,show_leaf_counts=False)#系统聚类的系谱图
scores1=[]
for n in range(2,8):
    clustering=AgglomerativeClustering(n_clusters=n).fit(da2001)
    a1=clustering.labels_
    scores1.append(silhouette_score(da2001,a1))
print(scores1)
plt.plot(scores1)
plt.show()
#2002年
dist2=spatial.distance.pdist(da2002,metric='euclidean') #pdist计算矩阵间向量的距离
z2=linkage(dist2,method='complete')#基于最大距离的系统聚类方法
dendrogram(z2,get_leaves=False,show_leaf_counts=False)#系统聚类的系谱图
scores2=[]
for n in range(2,8):
    clustering=AgglomerativeClustering(n_clusters=n).fit(da2002)
    a2=clustering.labels_
    scores2.append(silhouette_score(da2002,a2))
print(scores2)
plt.plot(scores2)
plt.show()
#2003年
dist3=spatial.distance.pdist(da2003,metric='euclidean') #pdist计算矩阵间向量的距离
z3=linkage(dist3,method='complete')#基于最大距离的系统聚类方法
dendrogram(z3,get_leaves=False,show_leaf_counts=False)#系统聚类的系谱图
scores3=[]
for n in range(2,8):
    clustering=AgglomerativeClustering(n_clusters=n).fit(da2003)
    a3=clustering.labels_
    scores3.append(silhouette_score(da2003,a3))
print(scores3)
plt.plot(scores3)
plt.show()

#变量系统聚类
Y=X.T
#pdist计算矩阵间向量的距离
dist4=spatial.distance.pdist(Y,metric='euclidean')
#基于最大距离的系统聚类方法
z4=linkage(dist4,method='complete')
#系统聚类的系谱图
dendrogram(z4,get_leaves=False,show_leaf_counts=False)
scores4=[]
for n in range(2,8):
    clustering=AgglomerativeClustering(n_clusters=n).fit(Y)
    a4=clustering.labels_
    scores4.append(silhouette_score(Y,a4))
print(scores4)
plt.plot(scores4)
plt.show()

#样本的K-均值聚类
#2000
from sklearn.cluster import KMeans
k_range=range(2,8)
k_scores=[]
for k in k_range:
    clf=KMeans(n_clusters=k)
    clf.fit(da2000)
    scores=clf.inertia_
    k_scores.append(scores) #不同的k值下，拟合模型得到的对应的interia值
plt.plot(k_range,k_scores)
plt.xlabel('k_clusters for KMeans')
plt.ylabel('inertia')
plt.show()

clf=KMeans(n_clusters=3)
clf.fit(da2000)
result=clf.predict(da2000)
da0=da[da['年份'].isin([2000])]
da0.insert(loc=33,column='tag',value=result)
da0.to_excel(r'D:\大三下\数据挖掘\第二次平时作业和数据（聚类分析）\output0.xlsx',index=False,encoding="utf- 8")
#2001
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
k_range=range(2,8)
k_scores=[]
for k in k_range:
    clf=KMeans(n_clusters=k)
    clf.fit(da2001)
    scores=clf.inertia_
    k_scores.append(scores) #不同的k值下，拟合模型得到的对应的interia值
plt.plot(k_range,k_scores)
plt.xlabel('k_clusters for KMeans')
plt.ylabel('inertia')
plt.show()

clf=KMeans(n_clusters=4)
clf.fit(da2001)
result=clf.predict(da2001)
da1=da[da['年份'].isin([2001])]
da1.insert(loc=33,column='tag',value=result)
da1.to_excel(r'D:\大三下\数据挖掘\第二次平时作业和数据（聚类分析）\output1.xlsx',index=False,encoding="utf- 8")
#2002
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
k_range=range(2,8)
k_scores=[]
for k in k_range:
    clf=KMeans(n_clusters=k)
    clf.fit(da2002)
    scores=clf.inertia_
    k_scores.append(scores) #不同的k值下，拟合模型得到的对应的interia值
plt.plot(k_range,k_scores)
plt.xlabel('k_clusters for KMeans')
plt.ylabel('inertia')
plt.show()

clf=KMeans(n_clusters=4)
clf.fit(da2002)
result=clf.predict(da2002)
da2=da[da['年份'].isin([2002])]
da2.insert(loc=33,column='tag',value=result)
da2.to_excel(r'D:\大三下\数据挖掘\第二次平时作业和数据（聚类分析）\output2.xlsx',index=False,encoding="utf- 8")
#2003
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
k_range=range(2,8)
k_scores=[]
for k in k_range:
    clf=KMeans(n_clusters=k)
    clf.fit(da2003)
    scores=clf.inertia_
    k_scores.append(scores) #不同的k值下，拟合模型得到的对应的interia值
plt.plot(k_range,k_scores)
plt.xlabel('k_clusters for KMeans')
plt.ylabel('inertia')
plt.show()

clf=KMeans(n_clusters=3)
clf.fit(da2003)
result=clf.predict(da2003)
da3=da[da['年份'].isin([2003])]
da3.insert(loc=33,column='tag',value=result)
da3.to_excel(r'D:\大三下\数据挖掘\第二次平时作业和数据（聚类分析）\output3.xlsx',index=False,encoding="utf- 8")

#变量的K-均值聚类
k_range=range(2,8)
k_scores=[]
for k in k_range:
    clf=KMeans(n_clusters=k)
    clf.fit(Y)
    scores=clf.inertia_
    k_scores.append(scores) #不同的k值下，拟合模型得到的对应的interia值
plt.plot(k_range,k_scores)
plt.xlabel('k_clusters for KMeans')
plt.ylabel('inertia')
plt.show()

clf=KMeans(n_clusters=3)
clf.fit(Y)
result=clf.predict(Y)
Y0=da.T
columnsname=Y0.index.values.tolist()
del columnsname[0:2]
frame = DataFrame(Y,columnsname)
frame.insert(loc=520,column='tag',value=result)
frame=frame.T
frame.to_excel(r'D:\大三下\数据挖掘\第二次平时作业和数据（聚类分析）\output4.xlsx',encoding="utf- 8")


#样本K均值聚类，每类选择有代表性样本50%作为R1
#2000
ff=pd.read_excel(r'D:\大三下\数据挖掘\第二次平时作业和数据（聚类分析）\output0.xlsx',header=0)
f0=ff[ff['tag'].isin([0])]
f1=ff[ff['tag'].isin([1])]
f2=ff[ff['tag'].isin([2])]
ff0=f0.values
ff1=f1.values
ff2=f2.values
ff0=ff0[:,2:33]
ff1=ff1[:,2:33]
ff2=ff2[:,2:33]
#1st
from collections import Counter
clf=KMeans(n_clusters=1) #进行模型拟合
clf.fit(ff0)
result=clf.predict(ff0)
b=clf.cluster_centers_
print(b) #获得每一簇的中心点坐标
print(Counter(result))
ff0_distance=clf.transform(ff0)
ff0_distance=ff0_distance.min(1)
f0.insert(loc=34,column='dis',value=ff0_distance)
f0=f0.sort_values(by='dis',axis=0,ascending=True) #升序排序
f0=f0.values
f00=f0[:50,:] #一半数据
#2nd
clf=KMeans(n_clusters=1) #进行模型拟合
clf.fit(ff1)
result=clf.predict(ff1)
b=clf.cluster_centers_
print(b) #获得每一簇的中心点坐标
print(Counter(result))
ff1_distance=clf.transform(ff1)
ff1_distance=ff1_distance.min(1)
f1.insert(loc=34,column='dis',value=ff1_distance)
f1=f1.sort_values(by='dis',axis=0,ascending=True) #升序排序
f1=f1.values
f10=f1[:5,:] #一半数据
#3rd
clf=KMeans(n_clusters=1) #进行模型拟合
clf.fit(ff2)
result=clf.predict(ff2)
b=clf.cluster_centers_
print(b) #获得每一簇的中心点坐标
print(Counter(result))
ff2_distance=clf.transform(ff2)
ff2_distance=ff2_distance.min(1)
f2.insert(loc=34,column='dis',value=ff2_distance)
f2=f2.sort_values(by='dis',axis=0,ascending=True) #升序排序
f2=f2.values
f20=f2[:11,:] #一半数据
#合并数据
data0=np.vstack((f00,f10,f20))

#2001
ff=pd.read_excel(r'D:\大三下\数据挖掘\第二次平时作业和数据（聚类分析）\output1.xlsx',header=0)
f0=ff[ff['tag'].isin([0])]
f1=ff[ff['tag'].isin([1])]
f2=ff[ff['tag'].isin([2])]
f3=ff[ff['tag'].isin([3])]
ff0=f0.values
ff1=f1.values
ff2=f2.values
ff3=f3.values
ff0=ff0[:,2:33]
ff1=ff1[:,2:33]
ff2=ff2[:,2:33]
ff3=ff3[:,2:33]
#1st
from collections import Counter
clf=KMeans(n_clusters=1) #进行模型拟合
clf.fit(ff0)
result=clf.predict(ff0)
b=clf.cluster_centers_
print(b) #获得每一簇的中心点坐标
print(Counter(result))
ff0_distance=clf.transform(ff0)
ff0_distance=ff0_distance.min(1)
f0.insert(loc=34,column='dis',value=ff0_distance)
f0=f0.sort_values(by='dis',axis=0,ascending=True) #升序排序
f0=f0.values
f00=f0[:10,:] #一半数据
#2nd
clf=KMeans(n_clusters=1) #进行模型拟合
clf.fit(ff1)
result=clf.predict(ff1)
b=clf.cluster_centers_
print(b) #获得每一簇的中心点坐标
print(Counter(result))
ff1_distance=clf.transform(ff1)
ff1_distance=ff1_distance.min(1)
f1.insert(loc=34,column='dis',value=ff1_distance)
f1=f1.sort_values(by='dis',axis=0,ascending=True) #升序排序
f1=f1.values
f10=f1[:23,:] #一半数据
#3rd
clf=KMeans(n_clusters=1) #进行模型拟合
clf.fit(ff2)
result=clf.predict(ff2)
b=clf.cluster_centers_
print(b) #获得每一簇的中心点坐标
print(Counter(result))
ff2_distance=clf.transform(ff2)
ff2_distance=ff2_distance.min(1)
f2.insert(loc=34,column='dis',value=ff2_distance)
f2=f2.sort_values(by='dis',axis=0,ascending=True) #升序排序
f2=f2.values
f20=f2[:1,:] #一半数据
#4st
clf=KMeans(n_clusters=1) #进行模型拟合
clf.fit(ff3)
result=clf.predict(ff3)
b=clf.cluster_centers_
print(b) #获得每一簇的中心点坐标
print(Counter(result))
ff3_distance=clf.transform(ff3)
ff3_distance=ff3_distance.min(1)
f3.insert(loc=34,column='dis',value=ff3_distance)
f3=f3.sort_values(by='dis',axis=0,ascending=True) #升序排序
f3=f3.values
f30=f3[:32,:] #一半数据
#合并数据
data1=np.vstack((f00,f10,f20,f30))

#2002
ff=pd.read_excel(r'D:\大三下\数据挖掘\第二次平时作业和数据（聚类分析）\output2.xlsx',header=0)
f0=ff[ff['tag'].isin([0])]
f1=ff[ff['tag'].isin([1])]
f2=ff[ff['tag'].isin([2])]
f3=ff[ff['tag'].isin([3])]
ff0=f0.values
ff1=f1.values
ff2=f2.values
ff3=f3.values
ff0=ff0[:,2:33]
ff1=ff1[:,2:33]
ff2=ff2[:,2:33]
ff3=ff3[:,2:33]
#1st
from collections import Counter
clf=KMeans(n_clusters=1) #进行模型拟合
clf.fit(ff0)
result=clf.predict(ff0)
b=clf.cluster_centers_
print(b) #获得每一簇的中心点坐标
print(Counter(result))
ff0_distance=clf.transform(ff0)
ff0_distance=ff0_distance.min(1)
f0.insert(loc=34,column='dis',value=ff0_distance)
f0=f0.sort_values(by='dis',axis=0,ascending=True) #升序排序
f0=f0.values
f00=f0[:7,:] #一半数据
#2nd
clf=KMeans(n_clusters=1) #进行模型拟合
clf.fit(ff1)
result=clf.predict(ff1)
b=clf.cluster_centers_
print(b) #获得每一簇的中心点坐标
print(Counter(result))
ff1_distance=clf.transform(ff1)
ff1_distance=ff1_distance.min(1)
f1.insert(loc=34,column='dis',value=ff1_distance)
f1=f1.sort_values(by='dis',axis=0,ascending=True) #升序排序
f1=f1.values
f10=f1[:41,:] #一半数据
#3rd
clf=KMeans(n_clusters=1) #进行模型拟合
clf.fit(ff2)
result=clf.predict(ff2)
b=clf.cluster_centers_
print(b) #获得每一簇的中心点坐标
print(Counter(result))
ff2_distance=clf.transform(ff2)
ff2_distance=ff2_distance.min(1)
f2.insert(loc=34,column='dis',value=ff2_distance)
f2=f2.sort_values(by='dis',axis=0,ascending=True) #升序排序
f2=f2.values
f20=f2[:1,:] #一半数据
#4st
clf=KMeans(n_clusters=1) #进行模型拟合
clf.fit(ff3)
result=clf.predict(ff3)
b=clf.cluster_centers_
print(b) #获得每一簇的中心点坐标
print(Counter(result))
ff3_distance=clf.transform(ff3)
ff3_distance=ff3_distance.min(1)
f3.insert(loc=34,column='dis',value=ff3_distance)
f3=f3.sort_values(by='dis',axis=0,ascending=True) #升序排序
f3=f3.values
f30=f3[:17,:] #一半数据
#合并数据
data2=np.vstack((f00,f10,f20,f30))

#2003
ff=pd.read_excel(r'D:\大三下\数据挖掘\第二次平时作业和数据（聚类分析）\output3.xlsx',header=0)
f0=ff[ff['tag'].isin([0])]
f1=ff[ff['tag'].isin([1])]
f2=ff[ff['tag'].isin([2])]
ff0=f0.values
ff1=f1.values
ff2=f2.values
ff0=ff0[:,2:33]
ff1=ff1[:,2:33]
ff2=ff2[:,2:33]
#1st
from collections import Counter
clf=KMeans(n_clusters=1) #进行模型拟合
clf.fit(ff0)
result=clf.predict(ff0)
b=clf.cluster_centers_
print(b) #获得每一簇的中心点坐标
print(Counter(result))
ff0_distance=clf.transform(ff0)
ff0_distance=ff0_distance.min(1)
f0.insert(loc=34,column='dis',value=ff0_distance)
f0=f0.sort_values(by='dis',axis=0,ascending=True) #升序排序
f0=f0.values
f00=f0[:59,:] #一半数据
#2nd
clf=KMeans(n_clusters=1) #进行模型拟合
clf.fit(ff1)
result=clf.predict(ff1)
b=clf.cluster_centers_
print(b) #获得每一簇的中心点坐标
print(Counter(result))
ff1_distance=clf.transform(ff1)
ff1_distance=ff1_distance.min(1)
f1.insert(loc=34,column='dis',value=ff1_distance)
f1=f1.sort_values(by='dis',axis=0,ascending=True) #升序排序
f1=f1.values
f10=f1[:1,:] #一半数据
#3rd
clf=KMeans(n_clusters=1) #进行模型拟合
clf.fit(ff2)
result=clf.predict(ff2)
b=clf.cluster_centers_
print(b) #获得每一簇的中心点坐标
print(Counter(result))
ff2_distance=clf.transform(ff2)
ff2_distance=ff2_distance.min(1)
f2.insert(loc=34,column='dis',value=ff2_distance)
f2=f2.sort_values(by='dis',axis=0,ascending=True) #升序排序
f2=f2.values
f20=f2[:6,:] #一半数据
#合并数据
data3=np.vstack((f00,f10,f20))
R1=np.vstack((data0,data1,data2,data3))
R1=R1[:,:33]
R1=R1.T
columnsname=Y0.index.values.tolist()
R1=DataFrame(R1,columnsname)
R1=R1.T
R1.to_excel(r'D:\大三下\数据挖掘\第二次平时作业和数据（聚类分析）\R1.xlsx',encoding="utf- 8")












