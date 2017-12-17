#Import libraries
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, Normalize
import seaborn as sns
import numpy as np
import pandas as pd
import os
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, cross_validation
from sklearn import svm, linear_model, tree
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor

#----Read dataset for white wine and red wine, upsample minority class and concatenate----#

#Get path for winequality-white.csv and winequality-red.csv
#Input your file path where the data is located
filepath = 'C:/Users/mwija/OneDrive/Documents/GitHub/Data_Science_Project/data/wine.csv'
pwd = os.getcwd()
os.chdir(os.path.dirname(filepath))
df = pd.read_csv(os.path.basename(filepath))
#set to current directory
os.chdir(pwd)

#Separate majority and minority classes
df_majority = df[df.color==0]
df_minority = df[df.color==1]

#Upsample minority class:red wine
df_minority_upsampled = resample(df_minority,
                                 replace=True, #sample with replacement
                                 n_samples=len(df_majority), #to match majority class
                                 random_state=123) #reproducible results

#Combine Majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

#Display new class counts
df_upsampled.color.value_counts()

#---------------------------------Correlation Matrix using seaborn----------------------------#
f, ax = plt.subplots(figsize=(10,8))
corr = df_upsampled.corr()
sns.heatmap(corr, mask=np.zeros_like(corr,dtype=np.bool),cmap=sns.diverging_palette(220,10,as_cmap=True),square=True,ax=ax)
#used to show correlation matrix plot
plt.show()

#--------------------------------Get train, validate, and test data for regression------------#
X_train, X_test, y_train, y_test = train_test_split(df_upsampled.iloc[:,0:11],df_upsampled.iloc[:,11], test_size=0.2, random_state=1)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
y_train = pd.DataFrame(y_train, columns=["quality"])

#concatenate X_train, y_train
xy_train = pd.concat([X_train, y_train],axis=1)

#----------------------Multiple Linear Regression----------------------------------------------#
lm1 = smf.ols(formula='quality ~ fixed_acidity + volatile_acidity + citric_acid + residual_sugar + chlorides + free_sulfur_dioxide + total_sulfur_dioxide + density + pH + sulphates + alcohol', data=xy_train).fit()
#Used to print summary of regression model
print(lm1.summary())



#Forward Selection helper function
def forward_selected(data, response):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model

#Get best OLS obtained from forward selection
model = forward_selected(xy_train,'quality')
print(model.summary())



#-----------------------------------SGD Regression-----------------------------#
X_scale_train = (X_train - X_train.mean())/np.std(X_train)
clf_sgd = linear_model.SGDRegressor()
clf_sgd.fit(X_scale_train, y_train)
rsquare = clf_sgd.score(X_scale_train, y_train)
X_test_scale = (X_test - X_train.mean())/np.std(X_train)
y_predicted = clf_sgd.predict(X_test_scale)
print(rsquare)



#------------------------------------Decision tree with PCA----------------------#
pca = PCA(n_components=1)
pca.fit(df_upsampled.iloc[:,0:11])
y_all = df_upsampled.iloc[:,11].values
originalX = pca.transform(df_upsampled.iloc[:,0:11])


X_det_train = pca.transform(X_train)
X_det_test = pca.transform(X_test)

#fit regression model
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X_det_train, y_train)
regr_2.fit(X_det_train,y_train)
#Predict
y_1 = regr_1.predict(X_det_test)
y_2 = regr_2.predict(X_det_test)

print(len(originalX),len(y_all))

#plot results
plt.figure()
plt.scatter(originalX,y_all, s=20, edgecolor="black",
            c="darkorange", label="data")
plt.plot(X_det_test, y_1, color="cornflowerblue",
         label="max_depth=2", linewidth=2)
plt.plot(X_det_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()





#------------------------------------Classification Algorithm-------------------------------------------------#

#------------------SVM with rbf kernel: No PCA-----------------#
X_scale_data = df_upsampled.iloc[:,0:12]
y_scale_data = df_upsampled.iloc[:,12]
min_max_scaler = preprocessing.MinMaxScaler()
X_scales = min_max_scaler.fit_transform(X_scale_data)
X_train, X_test, y_train, y_test = train_test_split(X_scales,y_scale_data, test_size=0.2, random_state=1)
y_test = y_test.values #convert to numpy array


C_range = np.logspace(-2,10,13)
gamma_range = np.logspace(-9,3,13)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(X_train,y_train)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

#fit rbf kernel svm with best parameters C = 0.1, gamma = 1
clf = svm.SVC(C=0.1, kernel='rbf',gamma=1)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

pred_score = 0;
for i in range(len(y_test)):
    if y_test[i] == y_pred[i]:
        pred_score += 1

#print result of prediction accuracy: turned out to be 100% clearly the model is overfitting and possibly very complex model
print(pred_score/len(y_test))



#--------------------------------SVM with PCA RBF Kernel----------------------------------------------------------#
#Pre-process dataset
#Get features and label data
X_clf = df_upsampled.iloc[:,0:12]
y_clf = df_upsampled.iloc[:,12]

#scale dataset to a range of (0,1)
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X_clf)
#convert label to numpy array
label = np.asarray(y_clf)

#initialize pca to do dimension reduction to 2 dimension
pca = PCA(n_components=2)
#initialize 10 fold cross validation and shuffle as True
kf = KFold(n_splits=10, shuffle=True)
#count holder to visualize last SVM iteration
count = 0;
prediction_accuracy = [];

#perform for loop for these varying parameters
#c_grid = [2**-5, 2**-3, 2**-1, 2, 2**3, 2**5, 2**7, 2**9, 2**11, 2**13, 2**15]
#gamma_grid = [2**-15, 2**-13, 2**-11, 2**-9, 2**-7, 2**-5, 2**-3, 2**-1, 2, 2**3, 2**5]

print('--------------------------- Result for SVM --------------- \n')
for train_indices, test_indices in kf.split(X_scale):
    #make training and testing datasets for each fold
    features_train = [X_scale[ii] for ii in train_indices]
    features_test = [X_scale[ii] for ii in test_indices]
    label_train = [label[ii] for ii in train_indices]
    label_test = [label[ii] for ii in test_indices]
    count += 1
    #do PCA on training features
    pca.fit(features_train)
    #transform training features with PCA
    X_t_train = pca.transform(features_train)
    #transform testing features with PCA
    X_t_test = pca.transform(features_test)
    #fit svm with c or penalty = 1 for larger margin
    clf = svm.SVC(C=1, gamma=0.001,kernel='rbf')
    clf.fit(X_t_train, label_train)
    #Get prediction accuracy for testing dataset
    score = clf.score(X_t_test, label_test)
    prediction_accuracy.append(score)
    print('Fold '+str(count)+' prediction accuracy is '+str(score))

    #visualize last fold as example
    if count == 10:
        label_test_pred = clf.predict(X_t_test)
        all_x = np.concatenate((X_t_train, X_t_test), axis=0)
        all_y = np.concatenate((label_train, label_test_pred), axis=0)
        key = {0: ('red','GoodBad=0'), 1: ('green','GoodBad=1')}
        plt.scatter(all_x[:,0],all_x[:,1], c=[key[index][0] for index in all_y], s=30)
        patches = [mpatches.Patch(color=color, label=label) for color,label in key.values()]
        plt.legend(handles=patches, labels=[label for _, label in key.values()], bbox_to_anchor=(.65,1))

        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy,xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T

        Z = clf.decision_function(xy).reshape(XX.shape)

        ax.contour(XX,YY,Z, colors='k', levels=[-1,0,1], alpha=0.5, linestyles=['--', '-', '--'])
        ax.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1], s=100, linewidth=1, facecolors='none')
        plt.title('SVM Classification c=1, gamma=0.001, kernel=linear')
        plt.show()

prediction_accuracy_avg = sum(prediction_accuracy)/len(prediction_accuracy)
print('Average prediction accuracy of 10-fold cross validation using SVM is '+str(prediction_accuracy_avg)+'\n')







#--------------------------------SVM with PCA Linear Kernel-------------------------------------------------------#

#Pre-process dataset
#Get features and label data
X_clf = df_upsampled.iloc[:,0:12]
y_clf = df_upsampled.iloc[:,12]

#scale dataset to a range of (0,1)
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X_clf)
#convert label to numpy array
label = np.asarray(y_clf)

#initialize pca to do dimension reduction to 2 dimension
pca = PCA(n_components=2)
#initialize 10 fold cross validation and shuffle as True
kf = KFold(n_splits=10, shuffle=True)
#count holder to visualize last SVM iteration
count = 0;
prediction_accuracy = [];

#perform for loop for these varying parameters
#c_grid = [2**-5, 2**-3, 2**-1, 2, 2**3, 2**5, 2**7, 2**9, 2**11, 2**13, 2**15]
#gamma_grid = [2**-15, 2**-13, 2**-11, 2**-9, 2**-7, 2**-5, 2**-3, 2**-1, 2, 2**3, 2**5]


print('--------------------------- Result for SVM --------------- \n')
for train_indices, test_indices in kf.split(X_scale):
    #make training and testing datasets for each fold
    features_train = [X_scale[ii] for ii in train_indices]
    features_test = [X_scale[ii] for ii in test_indices]
    label_train = [label[ii] for ii in train_indices]
    label_test = [label[ii] for ii in test_indices]
    count += 1
    #do PCA on training features
    pca.fit(features_train)
    #transform training features with PCA
    X_t_train = pca.transform(features_train)
    #transform testing features with PCA
    X_t_test = pca.transform(features_test)
    #fit svm with c or penalty = 1 for larger margin
    clf = svm.SVC(C=1, gamma=0.001,kernel='linear')
    clf.fit(X_t_train, label_train)
    #Get prediction accuracy for testing dataset
    score = clf.score(X_t_test, label_test)
    prediction_accuracy.append(score)
    print('Fold '+str(count)+' prediction accuracy is '+str(score))

    #visualize last fold as example
    if count == 10:
        label_test_pred = clf.predict(X_t_test)
        all_x = np.concatenate((X_t_train, X_t_test), axis=0)
        all_y = np.concatenate((label_train, label_test_pred), axis=0)
        key = {0: ('red','GoodBad=0'), 1: ('green','GoodBad=1')}
        plt.scatter(all_x[:,0],all_x[:,1], c=[key[index][0] for index in all_y], s=30)
        patches = [mpatches.Patch(color=color, label=label) for color,label in key.values()]
        plt.legend(handles=patches, labels=[label for _, label in key.values()], bbox_to_anchor=(.65,1))

        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy,xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T

        Z = clf.decision_function(xy).reshape(XX.shape)

        ax.contour(XX,YY,Z, colors='k', levels=[-1,0,1], alpha=0.5, linestyles=['--', '-', '--'])
        ax.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1], s=100, linewidth=1, facecolors='none')
        plt.title('SVM Classification c=1, gamma=0.001, kernel=linear')
        plt.show()

prediction_accuracy_avg = sum(prediction_accuracy)/len(prediction_accuracy)
print('Average prediction accuracy of 10-fold cross validation using SVM is '+str(prediction_accuracy_avg)+'\n')



#------------------------------------------ KNN with varies K-------------------------------------------#
kf = KFold(n_splits=5, shuffle=True)
general_notion_optimal = round(len(df_upsampled)**0.5)
varies_K = [1,3,5,7,10,15,20,25,30, 40,general_notion_optimal]
error_prediction_accuracy_avg = [];
print('--------------------------- Result for KNN --------------- \n')
for i in range(len(varies_K)):
    error_prediction_accuracy = [];
    count = 0
    for train_indices, test_indices in kf.split(X_scale):
        count += 1
        #make training and testing datasets for each fold
        features_train = [X_scale[ii] for ii in train_indices]
        features_test = [X_scale[ii] for ii in test_indices]
        label_train = [label[ii] for ii in train_indices]
        label_test = [label[ii] for ii in test_indices]
        #do PCA on training features
        pca.fit(features_train)
        #transform training features with PCA
        X_t_train = pca.transform(features_train)
        #transform testing features with PCA
        X_t_test = pca.transform(features_test)
        #fit svm with c or penalty = 1 for larger margin
        knn = KNeighborsClassifier(n_neighbors=varies_K[i], weights='distance')
        knn.fit(X_t_train, label_train)
        label_pred = knn.predict(X_t_test)
        #Get prediction accuracy for testing dataset
        error_prediction_knn = 1 - accuracy_score(label_test, label_pred)
        error_prediction_accuracy.append(error_prediction_knn)
        if count == 5:
            avg_error_cv = sum(error_prediction_accuracy)/len(error_prediction_accuracy)
            error_prediction_accuracy_avg.append(avg_error_cv)
        if (varies_K[i] == 40 and count == 5) or (varies_K[i] == general_notion_optimal  and count == 5) :
            all_x = np.concatenate((X_t_train, X_t_test), axis=0)
            all_y = np.concatenate((label_train, label_pred), axis=0)

            h = .02 #step size in the mesh
            markers = ('x', 'o')
            colors = ('red', 'blue')
            cmap = ListedColormap(colors[:len(np.unique(label_test))])
            x_min, x_max = X_t_train[:,0].min() - 1, X_t_train[:,0].max() + 1
            y_min, y_max = X_t_train[:,1].min() - 1, X_t_train[:,1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
            Z = knn.predict(np.array([xx.ravel(), yy.ravel()]).T)
            Z = Z.reshape(xx.shape)
            plt.contourf(xx,yy, Z, alpha=0.4, cmap=cmap)
            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())

            for idx, cl in enumerate(np.unique(label_train)):
                plt.scatter(x=all_x[all_y==cl,0], y=all_x[all_y==cl,1],
                           alpha=0.8, c=cmap(idx),
                           marker=markers[idx], label=cl)
            plt.legend(loc='upper right')
            plt.title('KNN Classification with K='+str(varies_K[i]))
            plt.show()

print('K=99: Prediction accuracy with 10-fold CV is '+str(1- error_prediction_accuracy_avg[10]) + '\n')

for i in range(len(varies_K)):
    print('K='+str(varies_K[i])+': Error prediction accuracy with 10-fold CV is '+str(error_prediction_accuracy_avg[i]))


plt.plot(varies_K, error_prediction_accuracy_avg,'-o')
plt.title('Error Prediction Rate with Variety of Ks')
plt.ylabel('Error Prediction Rate')
plt.xlabel('Number of Neighbors')
plt.show()



#------------------------------------Decision Tree Classification-------------------------------------------------#
clf_tree = tree.DecisionTreeClassifier()
X_train, X_test, y_train, y_test = train_test_split(df_upsampled.iloc[:,0:11],df_upsampled.iloc[:,12], test_size=0.2, random_state=1)
cross_val_score(clf_tree, X_train, y_train, cv=10)
clf_tree = clf_tree.fit(X_train, y_train)
y_test_pred = clf_tree.predict(X_test)
accuracy_count = 0
y_test = y_test.values
X_title = list(df_upsampled.iloc[:,0:11])


for i in range(len(y_test)):
    if y_test[i] == y_test_pred[i]:
        accuracy_count += 1
tree_accuracy = accuracy_count/len(y_test)
print(tree_accuracy)
