
# coding: utf-8

# # Estudo de caso: Verificação de emails da Enron para identificação de fraudes
# 
# ## Introdução
# 
# Este estudo de caso tem como objetivo identificar os empregados que cometeram fraude na Enron, analisando um conjunto de dados de funcionarios e folha de dados com a ajuda de Machine Learning.
# 
# <b>1.Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]</b>
# 
# Machine Learning será muito util para a realização deste projeto, pois nos possibilita classificar os funcionários. Neste caso, o conjunto de dados classifica os funcionário em 2 tipos:
# <ul>
#    <li>0: Funcionários que são inocentes</li>
#    <li>1: Funcionários que são considerados culpados</li>
# </ul>
# 
# Muitos dados estão faltando neste DataSet, para conseguir trabalhar com os nulos (NaN), substituimos por 0 ou pela média dependendo do tipo de dado. 
# 
# Existem 146 registros com 22 caracteristicas para uma chave e encontramos um outlier quando verificamos uma relação de 'Salário' e 'Bonus'
# 
# <b>2. What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “intelligently select features”, “properly scale features”]</b>
# 
# Para selecionar minhas features, eu selecionei 7 features entre os melhores features de todos os algoritmos de seleção de features que eu usei (SelecktKBest, DecisionTree_FeatureImportance, RandomForest) + poi + novas fetures criadas, para saber quais as features mais importantes para o meu classificador. Com relação a escala, nenhum dos meus classificados precisaram de ajustes.
# 
# Eu criei 2 novas features para o projeto, fraction_from_poi'(from_poi_to_this_person/to_messages)and 'fraction_to_poi'(from_this_person_to_poi/from_messages), além disso usei o SelectKBest e o recurso de featureImportance do DecisionTree para verificar os features mais importantes para realizar o treinamento dos meus dados.
# 
# <table>
#     <tr>
#         <th>Classificador</th>
#         <th>Acurracy</th>
#         <th>Precision</th>
#         <th>Recall</th>
#     </tr>
#     <tr>
#         <td>DecisionTree</td>
#         <td>0.795</td>
#         <td>0.227</td>
#         <td>0.223</td>
#     </tr>
#     <tr>
#         <td>DecisionTree + new features</td>
#         <td>0.831</td>
#         <td>0.36</td>
#         <td>0.34</td>
#     </tr>
# </table>
# 
# <b>3. What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?</b>
# 
# Eu tentei compara o GausianNB com a DecisionTree, e apesar da seleção de features do GausianNB parecer melhor para aprimorar o algoritimo, o DecisionTree apresenta uma avaliação melhor no final
# 
# <table>
#     <tr>
#         <th>Classificador</th>
#         <th>Acurracy</th>
#         <th>Precision</th>
#         <th>Recall</th>
#     </tr>
#     <tr>
#         <td>GaussianNB</td>
#         <td>0.739</td>
#         <td>0.223</td>
#         <td>0.385</td>
#     </tr>
#     <tr>
#         <td>DecisionTree</td>
#         <td>0.795</td>
#         <td>0.227</td>
#         <td>0.223</td>
#     </tr>
# </table>
# 
# <b>What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric items: “discuss parameter tuning”, “tune the algorithm”]</b>
# 
# O objetivo de 'tunar' o algoritimo é dar algumas premissas para que ele classifique melhor os dados de acordo com o que você precisa, se isso nâo acontecer direito, pode ocorrer de aparecer mais falsos positivos, além de uando realizar a validação, o algoritimo não vai ter uma boa avaliação e tambêm pode prejudicar a performance
# 
# Para fazer isso, eu usei o GridSearchCV para avaliar os meus dados e selecionar os melhores parametros para o meu classificador 
# o resultado foi modificar o meu classificador DecisionTree colocando:
# 
# DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=8,
#             max_features=None, max_leaf_nodes=None,
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf=1, min_samples_split=2,
#             min_weight_fraction_leaf=0.0, presort=False, random_state=None,
#             splitter='best')
# 
# e no meu metodo de obter daods de test eu coloquei : random_state = 42
# 
# <b>5. What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis? [relevant rubric items: “discuss validation”, “validation strategy”]</b>
#     
# A validação é um processo para separar uma coleção de dados de teste do modelo de dados treinados, aplicando um modelo selecionado no conjunto de testes, possibiliitando resutados mais precisos. 
# 
# Um erro comum é testar o classificador com dados que já foram trainados, neste caso sempre vai parecer que seu algoritimo está ótimo. 
# 
# Para validar minha analise, eu usei o método train_test_split, para separar um conjunto de dados de teste e após isso testar-los no meu classificador
# 
# <b>6. Give at least 2 evaluation metrics and your average performance for each of them. Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]</b>
# 
# Eu utilizei accuracy, precision e recall para verificar os meus dados, no teste final, usando DecisionTree, os resultados foram:
# 
# <!-- Fonte: http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html -->
# Precision é o numero de true positivos sobre o numero de true positivos mais o numero de falsos positivos.
# Recall é o numero de true positivos sobre o numero de true positivos mais o numero de falsos negtivos.
# 
# 
# <ul>
#     <li>Accuracy test:  0.83</li>
#     <li>Precision test:  0.36</li>
#     <li>Recall test:  0.34</li>
# </ul>
# 
# 
# A seguir é mostrado todo meu processo para realizar este trabalho

# In[352]:


import sys
import pickle
from pandas.tools.plotting import *
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

# separas as features entre financeiro e email
financialFeatures= ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
emailFeatures =  ['to_messages', 'from_poi_to_this_person', 'from_messages',
                   'from_this_person_to_poi', 'shared_receipt_with_poi']


# In[353]:


sys.path.append("tools/")
sys.path.append("final_project/")

from feature_format import featureFormat, targetFeatureSplit


# In[354]:


# criar um dataframe a partir do arquivo pickle
if os.path.getsize('final_project/final_project_dataset.pkl') > 0: 
    with open('final_project/final_project_dataset.pkl', "rb") as data_file:
        df = pd.DataFrame(pd.read_pickle(data_file)).transpose()
else:
    print('Erro no arquivo')
    
with open('final_project/final_project_dataset.pkl', "rb") as data_file:
    data_dict = pickle.load(data_file)


# In[355]:


df.head(5)


# ## Entendimento dos Dados e da Pergunta

# ### Exploração

# In[356]:


# substituir os Nan pelo objeto do Nan do numpy para identificar possiveis nulos
df = df.replace('NaN', np.NaN)

allFeatures = ['poi', 
'bonus','deferral_payments',
'deferred_income', 'director_fees','exercised_stock_options','expenses',
'from_messages','from_poi_to_this_person','from_this_person_to_poi',
'loan_advances','long_term_incentive','other','restricted_stock','restricted_stock_deferred', 'salary',
'shared_receipt_with_poi','to_messages','total_payments','total_stock_value']


# In[357]:


df.describe()


# In[358]:


#usar o info pra contar quantas informações nulas tem
df.info()


# Agora responder as seguintes perguntas:
# <ul>
#     <li>número total de data points</li>
#     <li>alocação entre classes (POI/non-POI)</li>
#     <li>número de características usadas</li>
#     <li>existem características com muitos valores faltando? etc.</li>
# </ul>

# In[359]:


# numero de datapoints
print("Nº de Datapoints: " + str(len(df)))
# numero de Caracteristicas
print("Nº de Caracteristicas: " + str(len(df.columns) - 1))
# numero de Poi
print("Nº de POI: " + str(len(df[df['poi'] == 1])))
#numero de npoi
print("Nº de non-POI: " + str(len(df[df['poi'] == 0])))
#numero de features usadas
print("Nº de caracteristicas usadas: " + str(len(df.columns) + 1))


# Pela analise dos dados que temos, Muitos dados estão faltando, e a maioria dos dados são numericos

# In[360]:


#plotar o numero de informações nulas do projeto
summary = df.describe().transpose()
summary['data_missing'] = (146 - summary['count'])/146
summary[['data_missing']].plot(kind='barh',figsize=(6, 6))


# In[361]:


def transform_null(dataframe,financialFeatures,emailFeatures):
    for feature in dataframe.columns:
        if feature in financialFeatures:
            dataframe[feature] = dataframe[feature].fillna(0)
        elif feature in emailFeatures:
            mean = dataframe[feature].mean()
            dataframe[feature] = dataframe[feature].fillna(mean)
    
    return dataframe

def convert_to_NaNs(dataframe):
    for col in dataframe.columns:
        dataframe[col] = dataframe[col].apply(lambda x: np.nan if str(x).strip()=='NaN' else x)
    
    nullByRow,nullByColumn = (nullValues(dataframe))
    
    return dataframe,nullByRow,nullByColumn
   
def nullValues(dataframe):
    nullByRow = dataframe.isnull().sum(axis=1)
    nullByColumn = dataframe.isnull().sum(axis=0)
    
    
    return nullByRow,nullByColumn
 
def inspect_data(dataframe):
    print('There are {} features, and {} persons in the enron data \n'.format(len(dataframe.columns),len(dataframe.index)))
    
    poi_npoi = dataframe['poi'].value_counts().to_dict()
    print('Of the {} persons, {} are classified as Persons of Interest \n'.format(len(dataframe.index),poi_npoi[True]))
    
    dataframe, nullrows,nullcolumns =convert_to_NaNs(df)
    print('The following columns: \n {} have null values greater than 100 \n'.format(nullcolumns[nullcolumns>100]))
    print('The following rows: \n {} \n have null values greater than 15 \n'.format(nullrows[nullrows>15]))

    return dataframe

# popular os nulos do projeto
data = transform_null(df,financialFeatures,emailFeatures)


# In[362]:


#plotar novamente para verificar se existe nulos 
summary = df.describe().transpose()
summary['data_missing'] = (146 - summary['count'])/146
summary[['data_missing']].plot(kind='barh',figsize=(6, 6))


# ### Identificação de outliers
# 
# Vamos comparar uns dados para encontrar possiveis outliers, primeiro ver salario e bonus

# In[363]:


#fazer um plot entre salário e bonus para indentificar possiveis outliers
df.plot.scatter('salary', 'bonus', figsize=(6, 6));


# Parece que existem outliser aqui, neste caso precisamos remove-los

# In[364]:


#Selecionar o outlier
df.loc[df['salary'] > 0.8 * 1e7]


# In[365]:


#Remover o outlier
df.drop(['LOCKHART EUGENE E','THE TRAVEL AGENCY IN THE PARK','TOTAL'],inplace=True)


# In[366]:


#Explorar novamente
df.plot.scatter('salary', 'bonus', figsize=(6, 6));


# ### Otimização da Seleção de Características/Engenharia

# In[367]:


PERF_FORMAT_STRING = "\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\tRecall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\tFalse negatives: {:4d}\tTrue negatives: {:4d}"
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedShuffleSplit

def my_test_classifier(clf, dataset, feature_list, folds = 1000):
    data = featureFormat(dataset, feature_list, sort_keys = True)
    X, y = make_classification()
    #y, X = targetFeatureSplit(data)
    #X = np.array(X)
    #y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    features_train = X_train
    features_test  = X_test
    labels_train   = y_train
    labels_test    = y_test
        
    ### fit the classifier using training set, and test on test set
    clf.fit(features_train, labels_train)
    predictions = clf.predict(features_test)
    for prediction, truth in zip(predictions, labels_test):
        if prediction == 0 and truth == 0:
            true_negatives += 1
        elif prediction == 0 and truth == 1:
            false_negatives += 1
        elif prediction == 1 and truth == 0:
            false_positives += 1
        elif prediction == 1 and truth == 1:
            true_positives += 1
        else:
            print ("Warning: Found a predicted label not == 0 or 1.")
            print ("All predictions should take value 0 or 1.")
            print ("Evaluating performance for processed predictions:")
            break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        print (clf)
        print (PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5))
        print (RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives))
        print ("")
    except:
        print ("Got a divide by zero when trying out:", clf)
        print ("Precision or recall may be undefined due to a lack of true positive predicitons.")

def test_classifier(clf, dataset, feature_list, folds = 0.5):
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    labels = np.array(labels)
    features = np.array(features)
    cv = StratifiedShuffleSplit(n_splits=1000, random_state=42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv.split(data, labels): 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print ("Warning: Found a predicted label not == 0 or 1.")
                print ("All predictions should take value 0 or 1.")
                print ("Evaluating performance for processed predictions:")
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        print (clf)
        print (PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5))
        print (RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives))
        print ("")
    except:
        print ("Got a divide by zero when trying out:", clf)
        print ("Precision or recall may be undefined due to a lack of true positive predicitons.")


# In[368]:


from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix,average_precision_score
from sklearn.feature_selection import SelectKBest, f_regression

features_list = np.array(allFeatures, copy=True)  

#converter o dataframe em um dicionário
data_dict = df.to_dict(orient='index')

#copiar o dicionário para nao mexer nos dados caso venha a ter algum problema
my_dataset = data_dict
data = featureFormat(my_dataset, features_list, sort_keys=True)

#Separar o target das features
y, X = targetFeatureSplit(data)
X = np.array(X)
y = np.array(y)

#treinar meus dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 100)


# ### Naive Bayes

# In[369]:


#naive bayes
clf = GaussianNB()
clf.fit(X, y)
y_pred = clf.predict(X)

# Avaliação, fonte: https://stackoverflow.com/questions/31421413/how-to-compute-precision-recall-accuracy-and-f1-score-for-the-multiclass-case
print('Accuracy: ', accuracy_score(clf.predict(X), y))
print('Precision: ', precision_score(y, clf.predict(X)))
print('Recall: ', recall_score(y, clf.predict(X)))   

test_classifier(clf, my_dataset, features_list)


# In[370]:


#Tune Gaussian
parameters = {}
grid = GridSearchCV(clf, parameters, cv=5)
grid.fit(X, y)
print(grid.best_estimator_)


# ### Arvore de decisão 

# In[371]:


from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()


# In[372]:


clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('Accuracy: ', accuracy_score(y_test, y_pred))
print('f1: ', f1_score(y_test, y_pred, average="macro"))
print('Precision: ', precision_score(y_test, y_pred, average="macro"))
print('Recall: ', recall_score(y_test, y_pred, average="macro"))    

test_classifier(clf, my_dataset, features_list)


# In[373]:


#Tune Decision Tree
parameters = {}
grid = GridSearchCV(clf, parameters, cv=5, scoring='f1')
grid.fit(X_train, y_train)
print(grid.best_estimator_)


# ### Seleção de features feita de forma inteligente

# In[374]:


#apresentar as caracteristicas mais importantes, fonte: 
#https://www.fabienplisson.com/choosing-right-features/
print (len(clf.feature_importances_))
print (len(features_list[1:19]))
importances = pd.DataFrame({'feature':features_list[1:20],'importance':np.round(clf.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
 
print (importances)
importances.plot.bar()


# In[375]:


#verificação de featurees
#fonte: https://stackoverflow.com/questions/40245277/visualize-feature-selection-in-descending-order-with-selectkbest
fit = SelectKBest(f_regression, k=19).fit(X, y)

indices = np.argsort(fit.scores_)[::-1]

features = []
for i in range(len(fit.scores_)):
    features.append(features_list[i])
    
#Now plot
importances = pd.DataFrame({'feature':features,'importance':indices})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
print (importances)
importances.plot.bar()


# In[376]:


#random forest, fonte: http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
from sklearn.ensemble import ExtraTreesClassifier
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)

indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()


# ### Criação de novas features
# Analisando o nosso DF, podemos trabalhar com uma ideia: <b>Tirar uma porcentagem das mensagens enviadas para um POI</b>, isso poderia ajudar a melhorar o classificador, entao podemos criar 2 novas caracteristicas
# <ul>
#     <li>from_this_person_to_poi: Relação da quantidades de emails enviados para um POI sobre o total de emails</li>
#     <li>from_poi_to_this_person: Relação da quantidades de emails recebidos de um POI sobre o total de emails</li>
# </ul>

# In[377]:


df['fraction_from_poi'] = df['from_poi_to_this_person'] / df['to_messages']
df['fraction_to_poi'] = df['from_this_person_to_poi'] / df['from_messages']

ax = df[df['poi'] == False].plot.scatter(x='fraction_from_poi', y='fraction_to_poi', color='blue', label='non-poi')
df[df['poi'] == True].plot.scatter(x='fraction_from_poi', y='fraction_to_poi', color='red', label='poi', ax=ax)


# ### Novo classificador com afinamento e seleção de features

# In[393]:


#selecionar 10 features do selecktkbest, o poi e as novas features criadas
features_list = ['poi', 'bonus', 'deferred_income', 'shared_receipt_with_poi', 'loan_advances', 
                'total_payments', 'expenses', 'long_term_incentive', 'exercised_stock_options', 
                'fraction_to_poi', 'fraction_from_poi']


#converter o dataframe em um dicionário
data_dict = df.to_dict(orient='index')

#copiar o dicionário para nao mexer nos dados caso venha a ter algum problema
my_dataset = data_dict
data = featureFormat(my_dataset, features_list, sort_keys = True)

y, X = targetFeatureSplit(data)
X = np.array(X)
y = np.array(y)

# alterar o teste 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


# In[394]:


#naive bayes
clf = GaussianNB(priors=None, var_smoothing=1e-09)
clf.fit(X_train, y_train)

# Avaliação
print('Accuracy test: ', accuracy_score(clf.predict(X_test), y_test))
print('Precision test: ', precision_score(y_test, clf.predict(X_test), average='macro'))
print('Recall test: ', recall_score(y_test, clf.predict(X_test), average='macro'))
test_classifier(clf, my_dataset, features_list)


# In[395]:


# arvore de decisão  pra testar após o tuning
clf = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

print('Accuracy test: ', accuracy_score(clf.predict(X_test), y_test))
print('Precision test: ', precision_score(y_test, clf.predict(X_test),average="macro"))
print('Recall test: ', recall_score(y_test, clf.predict(X_test), average="macro"))
test_classifier(clf, my_dataset, features_list)


# In[396]:


CLF_PICKLE_FILENAME = "my_classifier_tree.pkl"
DATASET_PICKLE_FILENAME = "my_dataset_tree.pkl"
FEATURE_LIST_FILENAME = "my_feature_list_tree.pkl"

def dump_classifier_and_data(clf, dataset, feature_list):
    with open(CLF_PICKLE_FILENAME, "wb") as clf_outfile:
        pickle.dump(clf, clf_outfile)
    with open(DATASET_PICKLE_FILENAME, "wb") as dataset_outfile:
        pickle.dump(dataset, dataset_outfile)
    with open(FEATURE_LIST_FILENAME, "wb") as featurelist_outfile:
        pickle.dump(feature_list, featurelist_outfile)

dump_classifier_and_data(clf,my_dataset,features_list)


# In[397]:


def load_classifier_and_data():
    with open(CLF_PICKLE_FILENAME, "rb") as clf_infile:
        ct = pickle.load(clf_infile)
    with open(DATASET_PICKLE_FILENAME, "rb") as dataset_infile:
        ds = pickle.load(dataset_infile)
    with open(FEATURE_LIST_FILENAME, "rb") as featurelist_infile:
        fl = pickle.load(featurelist_infile)
    return ct, ds, fl

ct, ds, fl = load_classifier_and_data()
### Run testing script
test_classifier(ct, ds, fl)

