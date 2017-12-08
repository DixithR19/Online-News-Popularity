
# coding: utf-8

# In[132]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import operator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR,LinearSVC,SVC
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
get_ipython().magic(u'matplotlib notebook')


# In[133]:


data=pd.read_csv('OnlineNewsPopularity.csv')
data.head()


# ## Check for null values

# In[134]:


data.isnull().values.any()


# ## Pre processing

# In[135]:


samples = data.drop([' shares'],axis=1)
shares = data[[' shares']]
np.array(shares)


# ## Train Test Split

# In[136]:


X_train ,X_test, Y_train, Y_test = train_test_split(samples, shares, test_size = 0.3)
#X_train_urls = X_train[['url']]
#X_train_td = X_train[[' timedelta']]
X_train = X_train.drop(['url'],axis=1)
#X_test_urls = X_test[['url']]
#X_test_td = X_test[[' timedelta']]
X_test = X_test.drop(['url'],axis=1)
X_train_norm = X_train.copy()
X_test_norm = X_test.copy()


# ## Normalization

# In[137]:



scaler = MinMaxScaler(feature_range=(-1, 1))
for key in X_train:
     if len(X_train[key].unique())>2:
         X_train[key] = scaler.fit_transform(X_train_norm[key])#(X_train[key]-np.mean(X_train[key]))/np.std(X_train[key])
         X_test[key] = scaler.fit_transform(X_test_norm[key])#(X_test_norm[key]-np.mean(X_train_norm[key]))/np.std(X_train_norm[key])

X_train.head()


# In[138]:


X_test.head()


# ## Dimnesionality Reduction

# In[139]:


dimensions = np.zeros((2,56))
for i in range(3,59):
    lda = LinearDiscriminantAnalysis(n_components=i)
    X_train1 = lda.fit(X_train,np.ravel(Y_train)).transform(X_train)
    X_test1=lda.transform(X_test)
    model = LinearRegression(fit_intercept=True,normalize=False,n_jobs=4)
    model.fit(X_train1,Y_train)
    pred_label = model.predict(X_test1)
    dimensions[0][i-3] = i
    dimensions[1][i-3] = r2_score(Y_test,pred_label)
print dimensions[1]


# In[140]:


plt.plot(dimensions[0],dimensions[1])
plt.xlabel('No of Dimensions')
plt.ylabel('R2_score')
plt.title('Dimensionality vs R2_Score')
plt.show()


# In[141]:



kf = KFold(n_splits=5)
sum1=0
for train, val in kf.split(X_train):
        train_data = np.array(X_train1)[train]
        val_data = np.array(X_train1)[val]
        train_label = np.array(Y_train)[train]
        val_label = np.array(Y_train)[val]
        for i in range(2,4):
            model = RandomForestRegressor()
            model.fit(train_data,np.ravel(train_label))
            Y_pred = model.predict(val_data)
            print str(i)+' :'+str(r2_score(np.ravel(val_label),Y_pred))


# In[142]:


# model = LinearRegression(fit_intercept=True,normalize=True,n_jobs=4)


# In[143]:


model.fit(X_train1,Y_train)
pred_label = model.predict(X_test1)
r2_score(Y_test,pred_label)


# In[144]:


lda = LinearDiscriminantAnalysis(n_components=15)
xtrain = lda.fit(X_train,np.ravel(Y_train)).transform(X_train)
xtest = lda.transform(X_test)


# ## Regression Analysis

# ### Linear Regression

# In[145]:


kf = KFold(n_splits=10)
count = 0
R2score = np.zeros((10,))
for train, val in kf.split(xtrain):
    train_data = np.array(xtrain)[train]
    val_data = np.array(xtrain)[val]
    train_label = np.array(Y_train)[train]
    val_label = np.array(Y_train)[val]
    model = LinearRegression(fit_intercept=True,normalize=True,n_jobs=4)
    model.fit(train_data,train_label)
    Y_pred = model.predict(val_data)
    R2score[count] = r2_score(val_label,Y_pred)
    if count == 0:
        X_train_final = train_data
        Y_train_final = train_label
    elif count > 0 and R2score[count] > max(R2score[0:count]):
        X_train_final = train_data
        Y_train_final = train_label
    count+=1

print 'Average Cross-validation R2-Score: '+str(np.mean(R2score))
model.fit(X_train_final,Y_train_final)
Y_pred = model.predict(xtest)
print 'R2-Score of the test set: '+str(r2_score(Y_test,Y_pred))


# ### Kernel Ridge Regression

# In[146]:


#alpha = np.logspace(-3,0,100)
model1 = KernelRidge(kernel='rbf')
model1.fit(xtrain,np.ravel(Y_train))
pred = model1.predict(xtest)
r2_score(np.ravel(Y_test),pred)
for i in range(2,10):
    model=SVR(kernel='poly',degree=i)
    model.fit(xtrain,np.ravel(Y_train))
    pred = model.predict(xtest)
    print r2_score(np.ravel(Y_test),pred)


# ## Classification

# In[147]:


train_labels=[]
for i in range(len(Y_train)):
    if np.ravel(Y_train)[i]>1400:
        train_labels.append(1)
    else:
        train_labels.append(0)
        
test_labels=[]
for i in range(len(Y_test)):
    if np.ravel(Y_test)[i]>1400:
        test_labels.append(1)
    else:
        test_labels.append(0)
        
kf = KFold(n_splits=5)


# ### Logistic Regression

# In[148]:


model = LogisticRegression(penalty='l1')
for train, val in kf.split(xtrain):
    train_data = np.array(xtrain)[train]
    val_data = np.array(xtrain)[val]
    train_label = np.array(train_labels)[train]
    val_label = np.array(train_labels)[val]
    model.fit(train_data,np.ravel(train_label))
    print model.score(val_data,np.ravel(val_label))
LogisticRegression()


# ### Linear SVM

# In[149]:


C = np.logspace(-3,1,50)
gamma = np.logspace(-3,1,50)
acc=np.zeros((5,))
avg_acc=np.zeros((len(C),))
for i in range(len(C)):
    print i
    p=0
    model = LinearSVC(C=C[i])
    for train, val in kf.split(xtrain):
        train_data = np.array(xtrain)[train]
        val_data = np.array(xtrain)[val]
        train_label = np.array(train_labels)[train]
        val_label = np.array(train_labels)[val]
        model.fit(train_data,np.ravel(train_label))
        acc[p]=model.score(val_data,np.ravel(val_label))
        p+=1
    avg_acc[i]=np.mean(acc)


# In[150]:


plt.plot(C,avg_acc)
plt.xlabel('C value')
plt.ylabel('Avg Accuracy')
plt.title('Optimization of hyperparameter')
plt.show()
avg_acc


# In[151]:


n=np.argmax(avg_acc)
C_opt=C[n]
## Testing accuracy
model = LinearSVC(C=4.5)
model.fit(xtrain,train_labels)
model.score(xtest,test_labels)


# ### SVM

# In[152]:


C = np.logspace(-3,0,20)
gamma = np.logspace(-3,0,20)
acc=np.zeros((5,))
avg_acc=np.zeros((len(C),len(gamma)))
for i in range(len(C)):
    print i
    for j in range(len(gamma)):
        print j
        p=0
        model = SVC(C=C[i],gamma=gamma[j])
        for train, val in kf.split(xtrain):
            train_data = np.array(xtrain)[train]
            val_data = np.array(xtrain)[val]
            train_label = np.array(train_labels)[train]
            val_label = np.array(train_labels)[val]
            model.fit(train_data,np.ravel(train_label))
            acc[p]=model.score(val_data,np.ravel(val_label))
            p+=1
        avg_acc[i][j]=np.mean(acc)
    


# In[153]:


Cx=[]
gammax=[]
for i in range(len(C)):
    for j in range(len(gamma)):
        Cx.append(C[i])
        gammax.append(gamma[j])
        
accx=avg_acc.flatten()
gridx,gridy=np.mgrid[min(Cx):max(Cx):50j,min(gammax):max(gammax):50j]
gridz=griddata((Cx,gammax),accx,(gridx,gridy))
fig=plt.figure()
ax=fig.gca(projection='3d')
plt.xlabel('C')
plt.ylabel('Gamma')

plt.title('Accuracy plot C vs Gamma')
ax.plot_surface(gridx, gridy, gridz,cmap=plt.cm.Spectral)
plt.show()


# In[154]:


## Testing accuracy
m,n = np.unravel_index(avg_acc.argmax(), avg_acc.shape)
C_opt=C[m]
print C_opt
gamma_opt=gamma[n]
print gamma_opt
model = SVC(C=C_opt,gamma=gamma_opt)
model.fit(xtrain,train_labels)
model.score(xtest,test_labels)


# ### Random Forest Classifier

# In[155]:


acc=np.zeros((5,))
avg_acc=[]
start=15
stop=151
for i in range(start,stop):
    model=RandomForestClassifier(n_estimators=i)
    print 'Trees:'+str(i)
    p=0
    for train, val in kf.split(xtrain):
        train_data = np.array(xtrain)[train]
        val_data = np.array(xtrain)[val]
        train_label = np.array(train_labels)[train]
        val_label = np.array(train_labels)[val]
        model.fit(train_data,np.ravel(train_label))
        acc[p]=model.score(val_data,np.ravel(val_label))
        p+=1
    avg_acc.append(np.mean(acc))
    


# In[156]:


a=np.linspace(start,stop-1,stop-start)
fig=plt.figure()
plt.plot(a,avg_acc)
plt.xlabel('No of trees')
plt.ylabel('Accuracy')
plt.title('Optimizing the no of trees in a Random Forest CLassifier')
plt.show()


# In[157]:


## Testing Accuracy
#pos=np.argmax(avg_acc)
#n_trees=start+pos
model=RandomForestClassifier(n_estimators=80,max_depth=2)
model.fit(xtrain,train_labels)
model.score(xtest,test_labels)


# ### AdaBoost Classifier

# In[158]:


acc=np.zeros((5,))
avg_acc=[]
start=10
stop=101
for i in range(start,stop):
    model=AdaBoostClassifier(n_estimators=i)
    print 'Trees:'+str(i)
    p=0
    for train, val in kf.split(xtrain):
        train_data = np.array(xtrain)[train]
        val_data = np.array(xtrain)[val]
        train_label = np.array(train_labels)[train]
        val_label = np.array(train_labels)[val]
        model.fit(train_data,np.ravel(train_label))
        acc[p]=model.score(val_data,np.ravel(val_label))
        p+=1
    avg_acc.append(np.mean(acc))


# In[159]:


a=np.linspace(start,stop-1,stop-start)
fig=plt.figure()
plt.plot(a,avg_acc)
plt.xlabel('No of trees')
plt.ylabel('Accuracy')
plt.title('Optimizing the no of estimators in AdaBoost CLassifier')
plt.show()


# In[160]:


## Testing Accuracy
#pos=np.argmax(avg_acc)
n_trees=10
learning_rate=np.logspace(-3,0,100)
# model.fit(xtrain,train_labels)
# model.score(xtest,test_labels)
acc=np.zeros((5,))
avg_acc=[]
start=0.001
stop=1
for i in learning_rate:
    model=AdaBoostClassifier(n_estimators=n_trees,learning_rate=i)
    print 'Trees:'+str(i)
    p=0
    for train, val in kf.split(xtrain):
        train_data = np.array(xtrain)[train]
        val_data = np.array(xtrain)[val]
        train_label = np.array(train_labels)[train]
        val_label = np.array(train_labels)[val]
        model.fit(train_data,np.ravel(train_label))
        acc[p]=model.score(val_data,np.ravel(val_label))
        p+=1
    avg_acc.append(np.mean(acc))


# In[161]:


fig=plt.figure()
plt.plot(learning_rate,avg_acc)
plt.xlabel('Learning_rate')
plt.ylabel('Accuracy')
plt.title('Optimization of Learning rate')
plt.show()


# In[162]:


#Test accuracy
n_estimators=n_trees
learning_rate=learning_rate[np.argmax(avg_acc)]
model=AdaBoostClassifier(n_estimators=n_estimators,learning_rate=learning_rate)
model.fit(xtrain,train_labels)
model.score(xtest,test_labels)


# ### Decision Tree Classifier

# In[163]:


from sklearn.tree import DecisionTreeClassifier
acc=np.zeros((5,))
avg_acc=[]
for i in range(1,101):
    model=DecisionTreeClassifier(max_depth=i)
    print 'Trees:'+str(i)
    p=0
    for train, val in kf.split(xtrain):
        train_data = np.array(xtrain)[train]
        val_data = np.array(xtrain)[val]
        train_label = np.array(train_labels)[train]
        val_label = np.array(train_labels)[val]
        model.fit(train_data,np.ravel(train_label))
        acc[p]=model.score(val_data,np.ravel(val_label))
        p+=1
    avg_acc.append(np.mean(acc))


# In[164]:


fig=plt.figure()
max_depth=np.linspace(1,100,100)
plt.plot(max_depth,avg_acc)
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Optimizing the max depth of a decision tree')
plt.show()


# In[165]:


from sklearn.tree import export_graphviz,DecisionTreeClassifier
model=DecisionTreeClassifier(max_depth=2)
model.fit(xtrain,train_labels)
export_graphviz(model,out_file='tree.dot')
model.score(xtest,test_labels)


# ## Clustering

# In[193]:


labelsTrain=np.ravel(Y_train)
labelsTest=np.ravel(Y_test)
aaa=np.sort(labelsTrain)
values=[]
i=0
n=2
while i < len(labelsTrain):
    i+=2775*10/n
    values.append(aaa[i-1])
cluster_train_labels=np.zeros((len(train_labels),))
cluster_test_labels=np.zeros((len(test_labels),))
for i in range(len(values)):
    for j in range(len(train_labels)):
        if i==0 and labelsTrain[j]<=values[i]:
            cluster_train_labels[j]=0
        elif labelsTrain[j]>values[i-1] and labelsTrain[j]<=values[i]:
            cluster_train_labels[j]=i
    
    for k in range(len(test_labels)):
        if i==0 and labelsTest[k]<=values[i]:
            cluster_test_labels[k]=0
        elif labelsTest[k]>values[i-1] and labelsTest[k]<=values[i]:
            cluster_test_labels[k]=i
print values

from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=n).fit(xtrain)
aa=kmeans.labels_
print (aa==cluster_train_labels).sum()/float(len(aa))


# In[194]:


from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(max_depth=2)
model.fit(xtrain,cluster_train_labels)
model.score(xtest,cluster_test_labels)


# In[182]:


import matplotlib
fig=plt.figure()
colors=['red','green','blue','yellow','cyan']
plt.scatter(xtrain[:,0],xtrain[:,5],c=cluster_train_labels,cmap=matplotlib.colors.ListedColormap(colors))
cb = plt.colorbar()
loc = np.arange(0,max(cluster_train_labels),max(cluster_train_labels)/float(len(colors)))
cb.set_ticks(loc)
cb.set_ticklabels(colors)


# In[ ]:





# In[ ]:





# In[ ]:




