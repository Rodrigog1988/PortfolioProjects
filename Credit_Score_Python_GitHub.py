# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 18:59:34 2019

@author: rodri
"""


"""

**************************************************************************
Book: Python for Data Analysis de Wes McKinney

CHAPTER 6: Data Loading, Storage, and File Formats
**************************************************************************

Sección: Interacting with Databases

"""

# region Conexion base de datos MySQL

"""
**************************************************************************
Conexion base de datos MySQL
**************************************************************************
"""

"""

pip install mysql-connector-python

"""

import mysql.connector
import pandas as pd

conexion1=mysql.connector.connect(host="localhost", 
                                  user="root", 
                                  passwd="", 
                                  database="datascience")
cursor1=conexion1.cursor()
cursor1.execute("select * from german_credit_normal")
rows = cursor1.fetchall()
#for fila in cursor1:
#    print(fila)
#conexion1.close() 

print(rows)
cursor1.description

"""

"""
pd.DataFrame(rows, columns=[x[0] for x in cursor1.description])

"""


    
pd.DataFrame(rows, columns=cursor1.description)

Es  mejor usar:
    
pd.DataFrame(rows, columns=[x[0] for x in cursor1.description])

Fuente: Pagina 189 Data Analysis Python

Si usaba directamente el código columns=cursor1.description la variable columns eran muy largas


"""

import pandas.io.sql as sql

sql.read_sql('select * from german_credit_normal', conexion1)



print(rows)

rows



credit = pd.DataFrame(rows, columns=[x[0] for x in cursor1.description])


credit

# endregion Conexion base de datos MySQL


# region Exploración y Operaciones con DataFrame: identificamos las variables continuas del dataset

"""
**************************************************************************
Operaciones DataFrame
**************************************************************************
"""

"""

Contar registros en un DataFrame, y aperturar por variables.

"""

credit.describe()

credit.count()



"""

Describir data types de columnas de un DataFrame:

"""



credit.dtypes

credit.info()

"""

Ver primeras 4 filas del DataFrame:

"""

credit.head()


"""

Ver últimas 4 filas del DataFrame:

"""

credit.tail()


"""

Mostrar todas las columnas del DataFrame:

"""

pd.set_option("display.max.columns", None)

credit

credit.head()

credit.tail()



"""

Mostrar cantidad de valores únicos que tiene cada variable o columna del DataFrame:

Show all the unique values for each column in the DataFrame:

https://pbpython.com/pandas_dtypes_cat.html

"""


unique_counts = pd.DataFrame.from_records([(col, credit[col].nunique()) for col in credit.columns],
                          columns=['Column_Name', 'Num_Unique']).sort_values(by=['Num_Unique'])


unique_counts

"""

Observamos que las variables: DURATION_OF_CREDIT_MONTH, AGE_YEARS y CREDIT_AMOUNT adquieren muchos valores, en otras palabras, son las 3 variables continuas 
a discretizar en el dataset. 


"""


"""

Más detalle en libro Data Anlysis pagina 8.1 Hierarchical Indexing  221


"""

credit.index


"""
Para elegir columnas: 
    
cursor1.description


credit2 = credit.set_index([('Creditability', 3, None, None, None, None, 1, 0),
 ('Account Balance', 3, None, None, None, None, 1, 0)])
    
"""
credit.columns


"""
página 226 Data Analysis

"""
credit2 = credit.set_index(['CREDITABILITY','ACCOUNT_BALANCE'])


credit2 


credit2.count(level="CREDITABILITY")


# endregion Operaciones DataFrame


# region Discretization and Binning

"""
**************************************************************************
Discretization and Binning
**************************************************************************
"""


"""

Discretization and Binning:
***************************

Libro: "Python for Data Analysis" pagina 203

"""
credit.columns

credit.CREDITABILITY
credit.AGE_YEARS

import numpy as np
data = np.random.randn(1000) # Normally distributed

data

import pandas as pd
cats = pd.qcut(data, 4) # Cut into quartiles


cats



data_age=credit.AGE_YEARS

age_bin_quartiles = pd.qcut(data_age, 4) # Cut into quartiles

age_bin_quartiles





"""

Poner nombres a las categorías:

"""

group_names = ['Youth', 'YoungAdult', 'MiddleAged', 'Senior']

age_bin_quartiles2 = pd.qcut(data_age, 4, labels=group_names) # Cut into quartiles


age_bin_quartiles2


"""

Deciles

"""

credit['AGE_YEARS']

data_age=credit.AGE_YEARS

age_bin_deciles = pd.qcut(data_age, 10) # Cut into deciles

age_bin_deciles



"""

Agregar campo derivado a dataframe:
***********************************

google: how to add to dataframe in python pd.qcut variable

https://stackoverflow.com/questions/28442991/python-pandas-create-new-bin-bucket-variable-with-pd-qcut

"""

credit['age_bin_deciles'] = pd.qcut(credit['AGE_YEARS'], 10, labels=False)


credit

credit['age_bin_deciles'] 

credit['age_bin_deciles2'] = pd.qcut(credit['AGE_YEARS'], 10)

credit


credit['age_bin_deciles2'] 


"""

Manualmente agrupar o hacer bins:

Usamos en vez de qcut, la funcion cut.

Agrupamos de <30, 30 a 65 años, 65 a 100 años:

"""



credit['age_bin_deciles3'] = pd.cut(credit['AGE_YEARS'], [0,30, 65, 100])

credit

credit['age_bin_deciles3'] 


"""

Invertir el valor inclusivo () o [] 

[(0, 30] < (30, 65] < (65, 100]]
    
"""


credit['age_bin_deciles4'] = pd.cut(credit['AGE_YEARS'], [0,30, 65, 100],right=True)

credit['age_bin_deciles4'] 


"""

[[0, 30) < [30, 65) < [65, 100)]
    
"""


credit['age_bin_deciles5'] = pd.cut(credit['AGE_YEARS'], [0,30, 65, 100],right=False)

credit['age_bin_deciles5'] 




# endregion Discretization and Binning



# region Graficamos variable categórica

"""
**************************************************************************
Graficamos variable categórica
**************************************************************************
"""


"""

Graficamos variable categórica

"""

credit

credit['age_bin_deciles2'] 

credit['CREDITABILITY'] 


"""
Contingency Table:
    
"""

AGE_BIN_X_CREDIT = pd.crosstab(credit['age_bin_deciles2'], credit['CREDITABILITY'])

AGE_BIN_X_CREDIT


"""

GRAFICO CANTIDAD

"""

AGE_BIN_X_CREDIT.plot.bar()

"""

GRAFICO CANTIDAD APILADO!!!!


"""


AGE_BIN_X_CREDIT.plot.bar(stacked=True)

# Normalize to sum to 1

AGE_BIN_X_CREDIT_NORM = AGE_BIN_X_CREDIT.div(AGE_BIN_X_CREDIT.sum(1), axis=0)

AGE_BIN_X_CREDIT_NORM


AGE_BIN_X_CREDIT_NORM.plot.bar()

"""

GRAFICO NORMALIZADO APILADO!!!!


"""

AGE_BIN_X_CREDIT_NORM.plot.bar(stacked=True)




# endregion Graficamos variable categórica




# region Chi-Square Test for Independence

"""
**************************************************************************
Chi-Square Test for Independence
**************************************************************************
"""


"""
Contingency Table:
    
"""

AGE_BIN_X_CREDIT = pd.crosstab(credit['age_bin_deciles2'], credit['CREDITABILITY'])

AGE_BIN_X_CREDIT


"""

Chi-Square Test for Independence:
*********************************

Para utilizar este test en Python hay que armar la tabla de contigencia previamente: AGE_BIN_X_CREDIT

https://towardsdatascience.com/gentle-introduction-to-chi-square-test-for-independence-7182a7414a95

SciPy’s chi2_contingency() returns four values:
    a) 𝜒2 value
    b) p-value 
    c) degree of freedom and expected values.
    
"""

from scipy.stats import chi2_contingency
chi2_contingency(AGE_BIN_X_CREDIT)


# endregion Chi-Square Test for Independence



# region Arboles CHAID: Discretización de la Variable Continuas

"""
**************************************************************************
Arboles CHAID
**************************************************************************
"""



"""
Arboles CHAID:
*************

    
    
    
Segun las 2 siguientes fuentes, por ahora sólo existe Arboles CHAID en Python con
variables explicativas nominales y ordinales:

https://github.com/Rambatino/CHAID
https://stackoverflow.com/questions/61747037/is-there-any-way-to-code-decision-tree-chaid-in-python-where-independent-varia    

Por ahora no existe Binning of continuous independent variables con el método CHAID en Python.
Pero según la pagina https://github.com/Rambatino/CHAID proximamente estará disponible "Upcoming Features".


Tengo dificultades para graficar árboles CHAID tal como lo explican y ya invertí mucho tiempo.
intentandolo. 


Feature importances en libro "Introduction to Machine Learning with Python"


"""




from CHAID import Tree


import numpy as np


credit.columns

credit



independent_variable_columns = ['age_bin_deciles']
dep_variable = 'CREDITABILITY'

independent_variable_columns
dep_variable

tree = Tree.from_pandas_df(credit, dict(zip(independent_variable_columns, ['ordinal'] * 1)), dep_variable)

tree

"""
De acuerdo a lo indicado por el arbol CHAID: hacemos las siguientes 2 agrupaciones:

groups=[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])


A continuación, generamos la tabla de contigencia para entender qué intervalos de edad continenes los conjuntos [0, 1, 2, 3, 4] y [5, 6, 7, 8, 9]
    
"""


AGE_BIN_X_AGE_BIN2= pd.crosstab([credit['age_bin_deciles2'], credit['age_bin_deciles']], credit['CREDITABILITY'], margins=True)


AGE_BIN_X_AGE_BIN2


"""

El conjunto [0, 1, 2, 3, 4] abarca el intervalo de edades de 0 a 33 años y el conjunto [5, 6, 7, 8, 9] abarca el intervalo de edades de 34 a 100 años.

Generamos la nueva variable discreta "age_bin_optimal":

"""

credit['age_bin_optimal'] = pd.cut(credit['AGE_YEARS'], [0,33,100])

credit['age_bin_optimal'] 

AGE_BIN_OPT_X_CREDIT= pd.crosstab(credit['age_bin_optimal'], credit['CREDITABILITY'], margins=False)


AGE_BIN_OPT_X_CREDIT




"""

GRAFICO CANTIDAD

"""

AGE_BIN_OPT_X_CREDIT.plot.bar()

"""

GRAFICO CANTIDAD APILADO!!!!


"""


AGE_BIN_OPT_X_CREDIT.plot.bar(stacked=True)

# Normalize to sum to 1

AGE_BIN_OPT_X_CREDIT_NORM = AGE_BIN_OPT_X_CREDIT.div(AGE_BIN_OPT_X_CREDIT.sum(1), axis=0)

AGE_BIN_OPT_X_CREDIT_NORM


"""

GRAFICO NORMALIZADO APILADO!!!!


"""

AGE_BIN_OPT_X_CREDIT_NORM.plot.bar(stacked=True)



"""
Calculamos nuevamente el P-Value de Chi Cuadrado:
    1) Da parecido al arbol CHAID!
    2) P-Value menor a 0.05--> Rechazo H0: hay dependencia de atributos y capturamos efectos no lineales.
    
"""


from scipy.stats import chi2_contingency
chi2_contingency(AGE_BIN_OPT_X_CREDIT)



# endregion Arboles CHAID



# region Discretización de la Variable 'AGE_YEARS'


"""

***********************************************************************
***********************************************************************
***********************************************************************

En vez de aplicar un arbol CHAID sobre una variable ordinal "age_bin_deciles" (que es la variable AGE en deciles), podemos
aplicarlo sobre una variable AGE particionada en 24 percentiles que llamaremos "age_bin_veinte" y así obtener resultados más parecidos
a los que indica SPSS o Tuffery: ambas fuentes indican que la mejor partición de la variable AGE (o donde se nota de manera más evidente la diferencia en la tasa de malos)
son en los intevalos de 0-25 años y en el intervalo de 25-100 años.

Aclaración: Decidí que sea en 24 percentiles porque si ampliaba a 100 percentiles la función qcut no funcionaba.
De esta manera obtendré particiones más precisas y parecidas a los resultados que arrojaría un Arbol CHAID en SPSS a partir de una variable continua, o las
del algoritmo Optimal Binning.

***********************************************************************
***********************************************************************
***********************************************************************

"""

"""

Discretizo variable continua AGE en 24 percentiles

"""




credit['age_bin_veinte'] = pd.qcut(credit['AGE_YEARS'], 24, labels=False)



credit['age_bin_veinte'] 

credit['age_bin_veinte2'] = pd.qcut(credit['AGE_YEARS'], 24)


credit['age_bin_veinte2'] 


AGE_BIN_X_AGE_BIN2= pd.crosstab([credit['age_bin_veinte2'], credit['age_bin_veinte']], credit['CREDITABILITY'], margins=True)


AGE_BIN_X_AGE_BIN2





from CHAID import Tree


import numpy as np


credit.columns

credit



independent_variable_columns = ['age_bin_veinte']
dep_variable = 'CREDITABILITY'

independent_variable_columns
dep_variable

tree = Tree.from_pandas_df(credit, dict(zip(independent_variable_columns, ['ordinal'] * 1)), dep_variable)

tree

"""
De acuerdo a lo indicado por el arbol CHAID: hacemos las siguientes 4 agrupaciones:

groups=[[0, 1, 2, 3], [4, 5, 6, 7, 8, 9, 10, 11, 12], [13, 14], [15, 16, 17, 18, 19, 20, 21, 22, 23]])


A continuación, generamos la tabla de contigencia para entender qué intervalos de edad continenes los conjuntos [0, 1, 2, 3], [4, 5, 6, 7, 8, 9, 10, 11, 12], 
[13, 14] y [15, 16, 17, 18, 19, 20, 21, 22, 23]
    
    
"""

AGE_BIN_X_AGE_BIN2


"""

Intervalos de Edad: de 0-25,26-34,35-36,37-100

"""




credit['age_bin_optimal'] = pd.cut(credit['AGE_YEARS'], [0,25,34,36,100])

credit['age_bin_optimal'] 

AGE_BIN_OPT_X_CREDIT= pd.crosstab(credit['age_bin_optimal'], credit['CREDITABILITY'], margins=False)


AGE_BIN_OPT_X_CREDIT




"""

GRAFICO CANTIDAD

"""

AGE_BIN_OPT_X_CREDIT.plot.bar()

"""

GRAFICO CANTIDAD APILADO!!!!


"""


AGE_BIN_OPT_X_CREDIT.plot.bar(stacked=True)


"""

Notar: como el intervalo 34-36 de edad abarca solo 40 clientes (muy pocos casos), conviene
agruparlo con el intervalo: 37-100.

De este modo las agrupaciones son iguales a las que produjo SPSS Modeler con el Arbol CHAID:
    
Intervalos de Edad optimos: de 0-25, 26-34, 35-100

"""




credit['age_bin_optimal'] = pd.cut(credit['AGE_YEARS'], [0,25,34,100])

credit['age_bin_optimal'] 

AGE_BIN_OPT_X_CREDIT= pd.crosstab(credit['age_bin_optimal'], credit['CREDITABILITY'], margins=False)


AGE_BIN_OPT_X_CREDIT




"""

GRAFICO CANTIDAD

"""

AGE_BIN_OPT_X_CREDIT.plot.bar()

"""

GRAFICO CANTIDAD APILADO!!!!


"""


AGE_BIN_OPT_X_CREDIT.plot.bar(stacked=True)





# Normalize to sum to 1

AGE_BIN_OPT_X_CREDIT_NORM = AGE_BIN_OPT_X_CREDIT.div(AGE_BIN_OPT_X_CREDIT.sum(1), axis=0)

AGE_BIN_OPT_X_CREDIT_NORM


"""

GRAFICO NORMALIZADO APILADO!!!!


"""

AGE_BIN_OPT_X_CREDIT_NORM.plot.bar(stacked=True)






from scipy.stats import chi2_contingency
chi2_contingency(AGE_BIN_OPT_X_CREDIT)



"""
Aunque los Intervalos de Edad optimos coinciden con el Arbol CHAID de SPSS: 
de 0-25, 26-34, 35-100, según Tuffery (páginas 572 y 57) sostiene que la 
difencia más evidente de tasas de malos va de 0-25 Años y de 26-100 Años.

Para tener resultados iguales a Tuffery decidí particionar la varaible AGE 
de la misma manera.
 
"""



credit['age_bin_optimal'] = pd.cut(credit['AGE_YEARS'], [0,25,100])

credit['age_bin_optimal'] 

AGE_BIN_OPT_X_CREDIT= pd.crosstab(credit['age_bin_optimal'], credit['CREDITABILITY'], margins=False)


AGE_BIN_OPT_X_CREDIT




"""

GRAFICO CANTIDAD

"""

AGE_BIN_OPT_X_CREDIT.plot.bar()

"""

GRAFICO CANTIDAD APILADO!!!!


"""


AGE_BIN_OPT_X_CREDIT.plot.bar(stacked=True)





# Normalize to sum to 1

AGE_BIN_OPT_X_CREDIT_NORM = AGE_BIN_OPT_X_CREDIT.div(AGE_BIN_OPT_X_CREDIT.sum(1), axis=0)

AGE_BIN_OPT_X_CREDIT_NORM


"""

GRAFICO NORMALIZADO APILADO!!!!


"""

AGE_BIN_OPT_X_CREDIT_NORM.plot.bar(stacked=True)



"""

Incluso el P-Value del Test de Independencia sigue siendo inferior a 0.05:

"""


from scipy.stats import chi2_contingency
chi2_contingency(AGE_BIN_OPT_X_CREDIT)

"""
Por lo tanto utilizaremos como variable explicativa la variable 'age_bin_optimal'
que va de 0-25 Años y de 26-100 Años, en vez de la variable continua original AGE.

Utilizar la variable discretizada 'age_bin_optimal' nos permite capturar efectos no lineales que existen 
entre la variable dependiente e independiente y evitar validar supuestos de normalidad, outliers, etc.

"""

credit['age_bin_optimal'] 

AGE_BIN_OPT_X_CREDIT= pd.crosstab(credit['age_bin_optimal'], credit['CREDITABILITY'], margins=False)


AGE_BIN_OPT_X_CREDIT


# endregion Discretización de la Variable 'AGE_YEARS'



# region Discretización de la Variable 'CREDIT_AMOUNT'


credit.columns

credit.CREDITABILITY

credit.CREDIT_AMOUNT




"""

Discretizo variable continua CREDIT_AMOUNT en 10 deciles

"""




credit['credit_bin'] = pd.qcut(credit['CREDIT_AMOUNT'], 10, labels=False)



credit['credit_bin'] 

credit['credit_bin2'] = pd.qcut(credit['CREDIT_AMOUNT'], 10)


credit['credit_bin2'] 


AMOUNT_BIN_X_AMOUNT_BIN2= pd.crosstab([credit['credit_bin2'], credit['credit_bin']], credit['CREDITABILITY'], margins=True)


AMOUNT_BIN_X_AMOUNT_BIN2





from CHAID import Tree


import numpy as np


credit.columns

credit



independent_variable_columns = ['credit_bin']
dep_variable = 'CREDITABILITY'

independent_variable_columns
dep_variable

tree = Tree.from_pandas_df(credit, dict(zip(independent_variable_columns, ['ordinal'] * 1)), dep_variable)

tree

"""
De acuerdo a lo indicado por el arbol CHAID: hacemos las siguientes 2 agrupaciones:


groups=[[0, 1, 2, 3, 4, 5, 6, 7], [8, 9]])   
    
A continuación, generamos la tabla de contigencia para entender qué intervalos de Credit AMOUNT 
que continenes los conjuntos.

    
    
"""

AMOUNT_BIN_X_AMOUNT_BIN2


"""

Intervalos de CREDIT_AMOUNT son de: de 0-4720, y más de 4720.

"""




credit['cred_amount_bin_optimal'] = pd.cut(credit['CREDIT_AMOUNT'], [0,4720,20000])

credit['cred_amount_bin_optimal'] 

AMOUNT_BIN_OPT_X_CREDIT= pd.crosstab(credit['cred_amount_bin_optimal'], credit['CREDITABILITY'], margins=False)


AMOUNT_BIN_OPT_X_CREDIT




"""

GRAFICO CANTIDAD

"""

AMOUNT_BIN_OPT_X_CREDIT.plot.bar()

"""

GRAFICO CANTIDAD APILADO!!!!


"""


AMOUNT_BIN_OPT_X_CREDIT.plot.bar(stacked=True)




# Normalize to sum to 1

AMOUNT_BIN_OPT_X_CREDIT_NORM = AMOUNT_BIN_OPT_X_CREDIT.div(AMOUNT_BIN_OPT_X_CREDIT.sum(1), axis=0)

AMOUNT_BIN_OPT_X_CREDIT_NORM


"""

GRAFICO NORMALIZADO APILADO!!!!


"""

AMOUNT_BIN_OPT_X_CREDIT_NORM.plot.bar(stacked=True)



"""

Incluso el P-Value del Test de Independencia sigue siendo inferior a 0.05:

"""


from scipy.stats import chi2_contingency
chi2_contingency(AMOUNT_BIN_OPT_X_CREDIT)

"""
Por lo tanto utilizaremos como variable explicativa la variable 'cred_amount_bin_optimal'
que va de 0-4720, y más de 4720, en vez de la variable continua original CREDIT_AMOUNT.

Utilizar la variable discretizada 'cred_amount_bin_optimal' nos permite capturar efectos no lineales que existen 
entre la variable dependiente e independiente y evitar validar supuestos de normalidad, outliers, etc.

"""

credit['cred_amount_bin_optimal'] 

AMOUNT_BIN_OPT_X_CREDIT= pd.crosstab(credit['cred_amount_bin_optimal'], credit['CREDITABILITY'], margins=False)


AMOUNT_BIN_OPT_X_CREDIT




# endregion Discretización de la Variable 'CREDIT_AMOUNT'



# region Discretización de la Variable 'DURATION_OF_CREDIT_MONTH'


credit.columns

credit.CREDITABILITY

credit.DURATION_OF_CREDIT_MONTH

credit['DURATION_OF_CREDIT_MONTH']


"""

Discretizo variable continua DURATION_OF_CREDIT_MONTH en 7 percentiles

"""



credit['credit_dur_bin'] = pd.qcut(credit['DURATION_OF_CREDIT_MONTH'], 7, labels=False)



credit['credit_dur_bin'] 

credit['credit_dur_bin2'] = pd.qcut(credit['DURATION_OF_CREDIT_MONTH'], 7)


credit['credit_dur_bin2'] 


DURATION_BIN_X_DURATION_BIN2= pd.crosstab([credit['credit_dur_bin2'], credit['credit_dur_bin']], credit['CREDITABILITY'], margins=True)


DURATION_BIN_X_DURATION_BIN2





from CHAID import Tree


import numpy as np


credit.columns

credit



independent_variable_columns = ['credit_dur_bin']
dep_variable = 'CREDITABILITY'

independent_variable_columns
dep_variable

tree = Tree.from_pandas_df(credit, dict(zip(independent_variable_columns, ['ordinal'] * 1)), dep_variable)

tree

"""
De acuerdo a lo indicado por el arbol CHAID: hacemos las siguientes 3 agrupaciones:


groups=[[0, 1, 2], [3, 4, 5], [6]]) 
    
A continuación, generamos la tabla de contigencia para entender qué intervalos de Credit AMOUNT 
que continenes los conjuntos.

    
    
"""

DURATION_BIN_X_DURATION_BIN2


"""

Intervalos de DURATION_OF_CREDIT_MONTH son de: de 0-15, 16-36 y más de 36 meses.

"""




credit['cred_duration_bin_optimal'] = pd.cut(credit['DURATION_OF_CREDIT_MONTH'], [0,15,36,100])

credit['cred_duration_bin_optimal'] 

DURATION_BIN_OPT_X_CREDIT= pd.crosstab(credit['cred_duration_bin_optimal'], credit['CREDITABILITY'], margins=False)


DURATION_BIN_OPT_X_CREDIT




"""

GRAFICO CANTIDAD

"""

DURATION_BIN_OPT_X_CREDIT.plot.bar()

"""

GRAFICO CANTIDAD APILADO!!!!


"""


DURATION_BIN_OPT_X_CREDIT.plot.bar(stacked=True)




# Normalize to sum to 1

DURATION_BIN_OPT_X_CREDIT_NORM = DURATION_BIN_OPT_X_CREDIT.div(DURATION_BIN_OPT_X_CREDIT.sum(1), axis=0)

DURATION_BIN_OPT_X_CREDIT_NORM


"""

GRAFICO NORMALIZADO APILADO!!!!


"""

DURATION_BIN_OPT_X_CREDIT_NORM.plot.bar(stacked=True)



"""

Incluso el P-Value del Test de Independencia sigue siendo inferior a 0.05:

"""


from scipy.stats import chi2_contingency
chi2_contingency(DURATION_BIN_OPT_X_CREDIT)

"""
Por lo tanto utilizaremos como variable explicativa la variable 'cred_duration_bin_optimal'
que va de 0-15, 16-36 y más de 36 meses, en vez de la variable continua original DURATION_OF_CREDIT_MONTH.

Utilizar la variable discretizada 'cred_duration_bin_optimal' nos permite capturar efectos no lineales que existen 
entre la variable dependiente e independiente y evitar validar supuestos de normalidad, outliers, etc.

"""

credit['cred_duration_bin_optimal'] 


DURATION_BIN_OPT_X_CREDIT= pd.crosstab(credit['cred_duration_bin_optimal'], credit['CREDITABILITY'], margins=False)


DURATION_BIN_OPT_X_CREDIT



# endregion Discretización de la Variable 'DURATION_OF_CREDIT_MONTH'






# region Cramer V en Python

"""

Utilizar o Cramer V o Arbol de decision sólo utilizando las variables discretas.

https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9

https://github.com/shakedzy/dython

http://shakedzy.xyz/dython/

http://shakedzy.xyz/dython/getting_started/examples/


Copado tiene para graficar Curva de ROC!!!!!

http://shakedzy.xyz/dython/getting_started/examples/


from dython.nominal import associations
associations(data)

"""

import scipy.stats as ss

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

"""

Ejemplo 1: Mapa de calor entre variable dependiente categorica 
y variables explicativas continuas. Creo que se calculan coeficientes de correlación.
    
Plot an example of an associations heat-map of the Iris dataset features. All features of this dataset are numerical (except for the target

"""


# pip install dython

import pandas as pd
from sklearn import datasets
from dython.nominal import associations

# Load data 
iris = datasets.load_iris()

# Convert int classes to strings to allow associations 
# method to automatically recognize categorical columns
target = ['C{}'.format(i) for i in iris.target]

# Prepare data
X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
y = pd.DataFrame(data=target, columns=['target'])
df = pd.concat([X, y], axis=1)

# Plot features associations
associations(df)



"""

Ejemplo 2: Theil’s U y Cramer’s V 

"""


import pandas as pd
from dython.nominal import associations

# Download and load data from UCI
df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data')
df.columns = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment',
              'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
              'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type',
              'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']


df.columns

df['gill-size']

# Plot features associations Theil’s U calculated for the mushrooms data-set
associations(df, theil_u=True, figsize=(15, 15))


# Cramer’s V calculated for the mushrooms data-set
associations(df, theil_u=False, figsize=(25, 15))

# endregion Cramer V en Python





# region SELECCION DE VARIABLES EXPLICATIVAS DEL MODELO y DEPENDENCIAS PROBLEMÁTICAS ENTRE VARIABLES EXPLICATIVAS (Multicolinealidad):



# region SELECCION DE VARIABLES EXPLICATIVAS DEL MODELO



"""

V-CRAMER: SELECCION DE VARIABLES EXPLICATIVAS DEL MODELO. 

Criterio de Cutoff / Umbral para la selección de Tuffery (pagina 580):  Variables con Cramer V INFERIOR a 0.05  se descartan.

Según Tuffery, con este criterio sólo se seleccionan 14 variables.


"""


"""

Generamos Dataframe equivalente a credit , pero dejando sólo las variables discretas:

"""


"""

Ejemplo 1: generamos copia identica de credit  con todas sus columnas o variables
    
"""

credit2 = credit.copy()

credit2

credit2.columns



"""

Ejemplo 2: generamos copia de credit  pero sólo con las columnas o variables
que nos interesan.
    
"""

credit.columns

selected_columns = credit[['CREDITABILITY', 'ACCOUNT_BALANCE',
                           'PAYMENT_STATUS_OF_PREVIOUS_CREDIT', 'PURPOSE',
       'VALUE_SAVINGS_STOCKS', 'LENGTH_OF_CURRENT_EMPLOYMENT',
       'INSTALMENT_PER_CENT', 'SEX_AND_MARITAL_STATUS', 'GUARANTORS',
       'DURATION_IN_CURRENT_ADDRESS', 'MOST_VALUABLE_AVAILABLE_ASSET',
       'CONCURRENT_CREDITS', 'TYPE_OF_APARTMENT',
       'NO_OF_CREDITS_AT_THIS_BANK', 'OCCUPATION', 'NO_OF_DEPENDENTS',
       'TELEPHONE', 'FOREIGN_WORKER',
       'age_bin_optimal',
       'cred_amount_bin_optimal',
       'cred_duration_bin_optimal'    
       ]]


credit3= selected_columns.copy()


credit3


credit3.columns



"""

Información de las variables del dataset:
number of rows, columns, column data types, memory usage, etc.
    

"""

print(credit3.info())


credit3.CREDITABILITY

credit3.ACCOUNT_BALANCE




"""
Hacemos conversión a variables categóricas

"""


credit3['CREDITABILITY'] = credit3['CREDITABILITY'].astype('object')
credit3['ACCOUNT_BALANCE'] = credit3['ACCOUNT_BALANCE'].astype('object')
credit3['PAYMENT_STATUS_OF_PREVIOUS_CREDIT'] = credit3['PAYMENT_STATUS_OF_PREVIOUS_CREDIT'].astype('object')
credit3['PURPOSE'] = credit3['PURPOSE'].astype('object')
credit3['VALUE_SAVINGS_STOCKS'] = credit3['VALUE_SAVINGS_STOCKS'].astype('object')
credit3['LENGTH_OF_CURRENT_EMPLOYMENT'] = credit3['LENGTH_OF_CURRENT_EMPLOYMENT'].astype('object')
credit3['INSTALMENT_PER_CENT'] = credit3['INSTALMENT_PER_CENT'].astype('object')
credit3['SEX_AND_MARITAL_STATUS'] = credit3['SEX_AND_MARITAL_STATUS'].astype('object')
credit3['GUARANTORS'] = credit3['GUARANTORS'].astype('object')
credit3['DURATION_IN_CURRENT_ADDRESS'] = credit3['DURATION_IN_CURRENT_ADDRESS'].astype('object')
credit3['MOST_VALUABLE_AVAILABLE_ASSET'] = credit3['MOST_VALUABLE_AVAILABLE_ASSET'].astype('object')
credit3['CONCURRENT_CREDITS'] = credit3['CONCURRENT_CREDITS'].astype('object')
credit3['TYPE_OF_APARTMENT'] = credit3['TYPE_OF_APARTMENT'].astype('object')
credit3['NO_OF_CREDITS_AT_THIS_BANK'] = credit3['NO_OF_CREDITS_AT_THIS_BANK'].astype('object')
credit3['OCCUPATION'] = credit3['OCCUPATION'].astype('object')
credit3['NO_OF_DEPENDENTS'] = credit3['NO_OF_DEPENDENTS'].astype('object')
credit3['TELEPHONE'] = credit3['TELEPHONE'].astype('object')
credit3['FOREIGN_WORKER'] = credit3['FOREIGN_WORKER'].astype('object')
credit3['age_bin_optimal'] = credit3['age_bin_optimal'].astype('object')
credit3['cred_amount_bin_optimal'] = credit3['cred_amount_bin_optimal'].astype('object')
credit3['cred_duration_bin_optimal'] = credit3['cred_duration_bin_optimal'].astype('object')




print(credit3.info())


"""

Mostrar todas las columnas del DataFrame:

"""

pd.set_option("display.max.columns", None)



"""

Calculamos y graficamos Cramer V en mapa de calor y hacemos Ranking de valores:

Fuente: https://www.kaggle.com/chrisbss1/cramer-s-v-correlation-matrix

"""


from scipy.stats import chi2_contingency
import numpy as np



"""

Esta función no arroja exactamentelos mismos valores de Tuffery:

def cramers_V(var1,var2) :
  crosstab =np.array(pd.crosstab(var1,var2, rownames=None, colnames=None)) # Cross table building
  stat = chi2_contingency(crosstab)[0] # Keeping of the test statistic of the Chi2 test
  obs = np.sum(crosstab) # Number of observations
  mini = min(crosstab.shape)-1 # Take the minimum value between the columns and the rows of the cross table
  return (stat/(obs*mini))

"""

import scipy.stats as ss

def cramers_V(var1, var2):
    confusion_matrix = pd.crosstab(var1,var2)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))




rows= []
for var1 in credit3:
  col = []
  for var2 in credit3 :
    cramers =cramers_V(credit3[var1], credit3[var2]) # Cramer's V test
    col.append(round(cramers,2)) # Keeping of the rounded value of the Cramer's V  
  rows.append(col)
  
cramers_results = np.array(rows)
df = pd.DataFrame(cramers_results, columns = credit3.columns, index =credit3.columns)


df

"""

Mostrar todas las columnas del DataFrame:

"""

pd.set_option("display.max.columns", None)


df


df['CREDITABILITY']


Serie_Cred=df['CREDITABILITY']

# sort Series the values in descending order 

Serie_Cred.sort_values(ascending = False) 

# sort Series the values in ascending order 

Serie_Cred.sort_values(ascending = True)



"""

Ranking de Variables explicativas de 'CREDITABILITY'

"""


Rank_Var_Explicativas=Serie_Cred.sort_values(ascending = False) 

Rank_Var_Explicativas


"""

V-CRAMER: SELECCION DE VARIABLES EXPLICATIVAS DEL MODELO. 

Criterio de Cutoff / Umbral para la selección de Tuffery (pagina 580):  Variables con Cramer V INFERIOR a 0.05  se descartan.

Según Tuffery, con este criterio sólo se seleccionan 14 variables.


"""



import matplotlib.pyplot as plt


fig, axes = plt.subplots(1, 1)
Rank_Var_Explicativas.plot.bar(color='k', alpha=0.7, figsize=(15, 15))


Rank_Var_Explicativas2=Serie_Cred.sort_values(ascending = True)

fig, axes = plt.subplots(1, 1)
Rank_Var_Explicativas2.plot.barh(color='k', alpha=0.7, figsize=(15, 15))


# endregion SELECCION DE VARIABLES EXPLICATIVAS DEL MODELO



# region IDENTIFICACIÓN DEPENDENCIAS PROBLEMÁTICAS ENTRE VARIABLES EXPLICATIVAS (Multicolinealidad)


"""

Según Tuffery (pagina 582): Las variables explicativas con Cramer V SUPERIOR a 0.4 son problemáticas, y puede haber Multicolinealidad, es decir 
que el comportamiento de ambas variables es similar y utilizar ambas variables en el modelo es redundante (además de generar otros problemas). 
Por lo tanto, hay que elegir sólo una de esas variables para el modelo, y así hacer un modelo simple, parsimonioso, con pocas variables explicativas. 


"""



"""

Hacemos mapa de calor para representar la matriz de valores de Cramer-V entre variables.

Seaborn Heatmap:
https://indianaiproduction.com/seaborn-heatmap/

"""


import seaborn as sns
import matplotlib.pyplot as plt

plt.rc('figure', figsize=(20, 20))

mask = np.zeros_like(df, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True




with sns.axes_style("white"):
  ax = sns.heatmap(df, mask=mask,vmin=0., vmax=1, square=True, annot = True)


"""

Heatmap top and bottom boxes are cut off:

Fuente: https://github.com/mwaskom/seaborn/issues/1773

"""  

# fix for mpl bug that cuts off top/bottom of seaborn viz
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values


plt.show()


"""

Cuadro Tuffery pagina 582 hace análisis de Dependencia entre TODAS las posibles variables explicativas, sin tener en cuenta el descarte que hace la selección de V-Cramer con respecto a la variable dependiente.

Tuffery hace ranking de dependencia y señala
3 dependencias fuertes entre variables explicativas con V Cramer mayor a 0.4 que son "problematicas":

1) Assets x status_residence || Most valuable available asset x Type of apartment

2) Credit_duration x Credit_amount

3) Job x Telephone || Occupation x Telephone

Tuffery en pagina 584 también analiza brevemente  otras dos dependencias fuertes que no superan 0.4:

4) Credit History X number of credits || Payment Status of Previous Credit X number of credits

5) Credit Amount X Purpose of Credit || Credit Amount X Purpose



"""


# endregion DEPENDENCIAS PROBLEMÁTICAS ENTRE VARIABLES EXPLICATIVAS (Multicolinealidad)



# region ANÁLISIS DEPENDENCIAS PROBLEMÁTICAS ENTRE VARIABLES EXPLICATIVAS (Multicolinealidad)


# region 1) Assets("Most valuable available asset"/"Valuables owned by the applicant") x status_residence



"""

Tuffery pagina 582: Hay una dependencia fuerte entre  Assets x status_residence 
por una sóla categoría en cada de estas variables: hay una asociación fuerte entre la categoría 3 ("Rent-Free")de Residential Status y 
categoría 4 de Asset ("Not known assets"). 

Más del 96% de los 3 ("Rent-Free") se encuentran en la categoría 4 de Asset ("Not known assets"). 

Según Tuffery página 582: "esto no justifica la exclusión de alguna de estas dos variables en esta etapa del analisis, 

sin embargo tenemos que tener en cuenta esta asociación en caso que se vuelva relevante".

Conclusión: Mantenemos las 2 variables Assets y status_residence por ahora.


"""


credit3.columns

credit3

credit3['MOST_VALUABLE_AVAILABLE_ASSET'] 

credit3['TYPE_OF_APARTMENT'] 


"""

Calculamos cantidad de registros que tiene el Dataset

"""

credit3.index

index = credit3.index

number_of_rows = len(index)

number_of_rows

"""
Contingency Table:
    
"""

ASSET_X_TYPE_APART = pd.crosstab(credit3['MOST_VALUABLE_AVAILABLE_ASSET'] , credit3['TYPE_OF_APARTMENT'] )

ASSET_X_TYPE_APART

(ASSET_X_TYPE_APART/number_of_rows)*100

"""

GRAFICO CANTIDAD

"""

ASSET_X_TYPE_APART.plot.bar()

"""

GRAFICO CANTIDAD APILADO!!!!


"""


ASSET_X_TYPE_APART.plot.bar(stacked=True)

# Normalize to sum to 1

ASSET_X_TYPE_APART_NORM = ASSET_X_TYPE_APART.div(ASSET_X_TYPE_APART.sum(1), axis=0)

ASSET_X_TYPE_APART_NORM


ASSET_X_TYPE_APART_NORM.plot.bar()

"""

GRAFICO NORMALIZADO APILADO!!!!


"""

ASSET_X_TYPE_APART_NORM.plot.bar(stacked=True)

ASSET_X_TYPE_APART_NORM

ASSET_X_TYPE_APART_NORM*100




TYPE_APART_X_ASSET= pd.crosstab(credit3['TYPE_OF_APARTMENT'],credit3['MOST_VALUABLE_AVAILABLE_ASSET'])


TYPE_APART_X_ASSET_NORM = TYPE_APART_X_ASSET.div(TYPE_APART_X_ASSET.sum(1), axis=0)


TYPE_APART_X_ASSET_NORM


TYPE_APART_X_ASSET_NORM.plot.bar()

"""

GRAFICO NORMALIZADO APILADO!!!!


"""

TYPE_APART_X_ASSET_NORM.plot.bar(stacked=True)

TYPE_APART_X_ASSET_NORM


"""

Más del 96% de los 3 ("Rent-Free") se encuentran en la categoría 4 de Asset ("Not known assets").


"""

TYPE_APART_X_ASSET_NORM*100





# endregion 1) Assets("Most valuable available asset"/"Valuables owned by the applicant") x status_residence


# region 2) Credit_duration x Credit_amount



"""

Dado que Bin_Duration tiene un valor de V Cramer superior a Amount con respecto a la variable dependiente Credibility, 
y que a la variable Bin_Duration "ofrecen una división más balanceada de la población (dado que ninguna de sus categoría 
excede la mitad de las solicitudes de créditos). 

Conclusión:  se elige a la variable Duration para el modelo.



Mi análisis también coincide que hay  una dependencia fuerte. Segun Tuffery, es una dependencia obvia dado que 
los montos en préstamos grandes tienden a otorgarse en plazo más grandes. Según Tuffery es extraño que en un modelo 
de scoring aparezcan juntas ambas variables. Dado que Duration tiene un valor de V Cramer superior a Amount con respecto 
a la variable dependendiente Credibility, y que ademas el analisis de Cross Tabulation indica que la variable Bin_Credit 
Duration "ofrecen una división más balanceada de la población (dado que ninguna categoría excede la mitad de las 
solicitudes de créditos). "Permitiendo distribuir satisfactoriamente las solicitudes de menor a mayor riesgo" --> se elige a la variable Duration para el modelo.


"""


credit3.columns

credit3

credit3['cred_amount_bin_optimal'] 

credit3['cred_duration_bin_optimal'] 


"""

Calculamos cantidad de registros que tiene el Dataset

"""

credit3.index

index = credit3.index

number_of_rows = len(index)

number_of_rows

"""
Contingency Table:
    
"""

DURATION_X_AMOUNT= pd.crosstab(credit3['cred_duration_bin_optimal'] , credit3['cred_amount_bin_optimal'] )

DURATION_X_AMOUNT

(DURATION_X_AMOUNT/number_of_rows)*100

"""

GRAFICO CANTIDAD

ATENCION!!! por alguna razon no puedo generar graficos con el dataframe DURATION_X_AMOUNT, aparentemente es un bug / error de Python.
Para solucionar esto, lo que hay que hacer es cambiar los nombres de la etiquetas originales (los cuales se generaban automáticamente a partir 
de la función pd.cut para generar cuartiles).

https://stackoverflow.com/questions/45424753/pandas-buffer-has-wrong-number-of-dimensions

"""

DURATION_X_AMOUNT.columns

DURATION_X_AMOUNT.columns = ['0-4720', '4721 o Más']


DURATION_X_AMOUNT.plot.bar()

"""

GRAFICO CANTIDAD APILADO!!!!


"""


DURATION_X_AMOUNT.plot.bar(stacked=True)

# Normalize to sum to 1

DURATION_X_AMOUNT_NORM = DURATION_X_AMOUNT.div(DURATION_X_AMOUNT.sum(1), axis=0)

DURATION_X_AMOUNT_NORM


DURATION_X_AMOUNT_NORM.plot.bar()

"""

GRAFICO NORMALIZADO APILADO!!!!


"""

DURATION_X_AMOUNT_NORM.plot.bar(stacked=True)

DURATION_X_AMOUNT_NORM

DURATION_X_AMOUNT_NORM*100






AMOUNT_X_DURATION= pd.crosstab(credit3['cred_amount_bin_optimal'] , credit3['cred_duration_bin_optimal'] )



AMOUNT_X_DURATION_NORM = AMOUNT_X_DURATION.div(AMOUNT_X_DURATION.sum(1), axis=0)


AMOUNT_X_DURATION_NORM


"""

GRAFICO

ATENCION!!! por alguna razon no puedo generar graficos con el dataframe AMOUNT_X_DURATION_NORM, aparentemente es un bug / error de Python.
Para solucionar esto, lo que hay que hacer es cambiar los nombres de la etiquetas originales (los cuales se generaban automáticamente a partir 
de la función pd.cut para generar cuartiles).

https://stackoverflow.com/questions/45424753/pandas-buffer-has-wrong-number-of-dimensions

"""

AMOUNT_X_DURATION_NORM.columns

AMOUNT_X_DURATION_NORM.columns = ['15 Meses o Menos', '15-36 Meses', '36 meses o Más']

AMOUNT_X_DURATION_NORM


AMOUNT_X_DURATION_NORM.plot.bar()

"""

GRAFICO NORMALIZADO APILADO!!!!


"""

AMOUNT_X_DURATION_NORM.plot.bar(stacked=True)

AMOUNT_X_DURATION_NORM


AMOUNT_X_DURATION_NORM*100



"""

Dado que Duration tiene un valor de V Cramer superior a Amount con respecto a la variable dependendiente Credibility, 
y que ademas el analisis de Cross Tabulation indica que la variable Bin_Credit Duration "ofrecen una división más 
balanceada de la población (dado que ninguna categoría excede la mitad de las solicitudes de créditos). 
"Permitiendo distribuir satisfactoriamente las solicitudes de menor a mayor riesgo", por lo tanto se elige a la variable Duration para el modelo.

Aunque Tuffery no lo menciona explícitamente, yo creo que la elección de variables explicativas que ofrezcan una distribución más 
balanceada de la población se relaciona con el problema de "Imbalance Data": modelos tienden a aprender reglas de clase mayoritaria.


"""


import matplotlib.pyplot as plt

Rank_Var_Explicativas

fig, axes = plt.subplots(1, 1)
Rank_Var_Explicativas.plot.bar(color='k', alpha=0.7, figsize=(15, 15))


Rank_Var_Explicativas2=Serie_Cred.sort_values(ascending = True)

fig, axes = plt.subplots(1, 1)
Rank_Var_Explicativas2.plot.barh(color='k', alpha=0.7, figsize=(15, 15))



"""

Distribucion de la variable CREDITABILITY respecto cred_duration_bin_optimal

"""


credit3.columns

credit3


credit3['cred_duration_bin_optimal'] 


"""

Calculamos cantidad de registros que tiene el Dataset

"""

credit3.index

index = credit3.index

number_of_rows = len(index)

number_of_rows

"""
Contingency Table:
    
"""

DURATION_X_CREDIT= pd.crosstab(credit3['cred_duration_bin_optimal'] , credit3['CREDITABILITY'] )

DURATION_X_CREDIT

(DURATION_X_CREDIT/number_of_rows)*100

"""

GRAFICO CANTIDAD


"""


DURATION_X_CREDIT.plot.bar()

"""

GRAFICO CANTIDAD APILADO!!!!


"""


DURATION_X_CREDIT.plot.bar(stacked=True)

# Normalize to sum to 1

DURATION_X_CREDIT_NORM = DURATION_X_CREDIT.div(DURATION_X_CREDIT.sum(1), axis=0)

DURATION_X_CREDIT_NORM


DURATION_X_CREDIT_NORM.plot.bar()

"""

GRAFICO NORMALIZADO APILADO!!!!


"""

DURATION_X_CREDIT_NORM.plot.bar(stacked=True)

DURATION_X_CREDIT_NORM

DURATION_X_CREDIT_NORM*100






AMOUNT_X_CREDIT= pd.crosstab(credit3['cred_amount_bin_optimal'] , credit3['CREDITABILITY'] )



AMOUNT_X_CREDIT_NORM = AMOUNT_X_CREDIT.div(AMOUNT_X_CREDIT.sum(1), axis=0)


AMOUNT_X_CREDIT_NORM


"""

GRAFICO


"""



AMOUNT_X_CREDIT_NORM.plot.bar()

"""

GRAFICO NORMALIZADO APILADO!!!!


"""

AMOUNT_X_CREDIT_NORM.plot.bar(stacked=True)

AMOUNT_X_CREDIT_NORM


AMOUNT_X_CREDIT_NORM*100




# endregion 2) Credit_duration x Credit_amount



# region 3) Job (Type of employment) x Telephone



"""

La variable Telephone no aparece entre las 14 variables explicativas más relevantes para explicar Credibility según el criterio de Cramer V (cutoff de 0,05). Por eso no hay problema con esta dependencia. 

Conclusión: se elige a la variable Job/"Type of employment" para el modelo
(mismo resultado que Tuffery pagina 584)



"""

import matplotlib.pyplot as plt

Rank_Var_Explicativas

fig, axes = plt.subplots(1, 1)
Rank_Var_Explicativas.plot.bar(color='k', alpha=0.7, figsize=(15, 15))


Rank_Var_Explicativas2=Serie_Cred.sort_values(ascending = True)

fig, axes = plt.subplots(1, 1)
Rank_Var_Explicativas2.plot.barh(color='k', alpha=0.7, figsize=(15, 15))





# endregion 3) Job (Type of employment) x Telephone



# region 4) Credit History X number of credits



"""

La variable "number of credits" / "Number of loans already recorded at the bank" no aparece entre las 14 variables explicativas más relevantes para explicar Credibility según el criterio de Cramer V. Por eso no hay problema con esta dependencia. 

Conclusión: se elige a la variable "Credit History "/"Repayment History" para el modelo
(mismo resultado que Tuffery pagina 584).



"""

import matplotlib.pyplot as plt

Rank_Var_Explicativas

fig, axes = plt.subplots(1, 1)
Rank_Var_Explicativas.plot.bar(color='k', alpha=0.7, figsize=(15, 15))


Rank_Var_Explicativas2=Serie_Cred.sort_values(ascending = True)

fig, axes = plt.subplots(1, 1)
Rank_Var_Explicativas2.plot.barh(color='k', alpha=0.7, figsize=(15, 15))





# endregion 4) Credit History X number of credits


# region 5) Credit Amount X Purpose of Credit



"""

La variable Bin_Credit_Amount se descartó en el análisis de dependencia 2. Por eso no hay problema con esta dependencia. 

Conclusión: se elige a la variable Purpose of Credit para el modelo
(mismo resultado que Tuffery pagina 584)


"""

import matplotlib.pyplot as plt

Rank_Var_Explicativas

fig, axes = plt.subplots(1, 1)
Rank_Var_Explicativas.plot.bar(color='k', alpha=0.7, figsize=(15, 15))


Rank_Var_Explicativas2=Serie_Cred.sort_values(ascending = True)

fig, axes = plt.subplots(1, 1)
Rank_Var_Explicativas2.plot.barh(color='k', alpha=0.7, figsize=(15, 15))





# endregion 5) Credit Amount X Purpose of Credit



# endregion ANÁLISIS DEPENDENCIAS PROBLEMÁTICAS ENTRE VARIABLES EXPLICATIVAS (Multicolinealidad)



# region Dataset resultante de la selección de variables explicativas y del descarte de dependencias problemáticas

"""
A partir de los criterios y análisis de Selección de Variables Explicativas del Modelo y de Identificación de Dependencias Problemáticas entre Variables explicativas
previamente desarrolladas, seleccionamos las variables con las que continuaremos trabajando, generando un nuevo dataset/dataframe llamado credit4:

"""

import matplotlib.pyplot as plt

Rank_Var_Explicativas

fig, axes = plt.subplots(1, 1)
Rank_Var_Explicativas.plot.bar(color='k', alpha=0.7, figsize=(15, 15))


Rank_Var_Explicativas2=Serie_Cred.sort_values(ascending = True)

fig, axes = plt.subplots(1, 1)
Rank_Var_Explicativas2.plot.barh(color='k', alpha=0.7, figsize=(15, 15))



credit3.columns

selected_columns = credit[['CREDITABILITY', 
                           'ACCOUNT_BALANCE', # 1) Mean Account Balance
                           'PAYMENT_STATUS_OF_PREVIOUS_CREDIT', # 2) Repayment history
                           'cred_duration_bin_optimal', # 3) Bin Credit Duration
                           'VALUE_SAVINGS_STOCKS', # 4) Savings Outstanding
                           'PURPOSE', # 5) Purpose of the credit
                           'MOST_VALUABLE_AVAILABLE_ASSET', # 6) Valuables owned by the applicant
                           'TYPE_OF_APARTMENT', # 7) Residential status
                           #'cred_amount_bin_optimal',  8) Bin Credit Amount: se descarta por dependencia problemática
                           # con la variable "Bin Credit Duration"
                           'LENGTH_OF_CURRENT_EMPLOYMENT', # 9) Length of current employment
                           'age_bin_optimal', # 10) Bin Age
                           'CONCURRENT_CREDITS', # 11) Other borrowings (outside the bank)
                           'SEX_AND_MARITAL_STATUS', # 12) Marital status

                           #'FOREIGN_WORKER', coinciden los resultados de Python con SPSS que 'FOREIGN_WORKER' tiene 
                           # un valor superior a 0.05 de Cramer de V, sin embargo en el libro de Tuffery está variable
                           # no es seleccionada entre las 14 más importantes.
                           
                           'GUARANTORS', # 13) Guarantors
                           'INSTALMENT_PER_CENT', # 14) Affordability ratio
       ]]


credit4= selected_columns.copy()


# endregion Dataset resultante de la selección de variables explicativas y del descarte de dependencias problemáticas




# region otra manera de generar matrix de Cramer V y mapa de calor

"""
Cramer’s V calculated 
"""


from dython.nominal import associations

associations(credit3, theil_u=False, figsize=(25, 15))

# endregion otra manera de generar matrix de Cramer V y mapa de calor


"""

1) hacer ejemplo de matriz de correlacion con dython!!! con coeficientes de correlacion
2) ver si puedo acotar matriz con coeficientes de cramer con condiciones: mostras aquellos cuyo
    coeficiente es superior a 0.3 por ejemplo


    
    
http://shakedzy.xyz/dython/modules/nominal/#compute_associations

compute_associations(credit3)

compute_associations¶
compute_associations(dataset, nominal_columns='auto', mark_columns=False, theil_u=False, clustering=False, bias_correction=True, nan_strategy=_REPLACE, nan_replace_value=_DEFAULT_REPLACE_VALUE)

"""

# endregion  SELECCION DE VARIABLES EXPLICATIVAS DEL MODELO y DEPENDENCIA PROBLEMÁTICAS ENTRE VARIABLES EXPLICATIVAS (Multicolinealidad):




# region CUÁNDO CONVIENTE O NO REGRUPAR UNA VARIABLE EXPLICATIVA CATEGÓRICA


"""

Relación Target-Variables Explicativas Categóricas: para poder determinar si es necesario o no reagrupar / modificar las variables explicativas categóricas seleccionadas.

Tuffery realiza la siguiente metodología con análisis bivariado:
Si la tasa de malos entre las distintas categorías es parecida, entonces las reagrupa en una sóla.
Si ciertas categorías abarcan muy pocos casos o el significado de negocio es parecido, entonces también reagrupa dichas categorías.
De lo contrario, deja las variables categóricas originales sin necesidad de reagrupar.

"""






# region 1) 'ACCOUNT_BALANCE'/Mean Account Balance x CREDITABILITY


credit4.columns

credit4

credit4['ACCOUNT_BALANCE'] 

credit4['CREDITABILITY'] 



"""

Calculamos cantidad de registros que tiene el Dataset

"""

credit4.index

index = credit4.index

number_of_rows = len(index)

number_of_rows

"""
Contingency Table:
    
"""

ACCOUNT_X_CREDIT= pd.crosstab(credit4['ACCOUNT_BALANCE'] , credit4['CREDITABILITY'] )

ACCOUNT_X_CREDIT

(ACCOUNT_X_CREDIT/number_of_rows)*100

"""

GRAFICO CANTIDAD

"""

ACCOUNT_X_CREDIT.plot.bar()

"""

GRAFICO CANTIDAD APILADO!!!!


"""


ACCOUNT_X_CREDIT.plot.bar(stacked=True)

# Normalize to sum to 1

ACCOUNT_X_CREDIT_NORM = ACCOUNT_X_CREDIT.div(ACCOUNT_X_CREDIT.sum(1), axis=0)

ACCOUNT_X_CREDIT_NORM


ACCOUNT_X_CREDIT_NORM.plot.bar()

"""

GRAFICO NORMALIZADO APILADO!!!!


"""


"""

La tasa de malos es distinta entre las distintas categorías de la variable original:
    
Relación Negativa: Mientras mayor es el Account Balance, menor es la tasa de malos.

 

Attibute 1 (A11): (qualitative)

 

Status of existing checking account

 

A11 : ... < 0 DM

A12 : 0 <= ... < 2 00 DM

A13 : ... >= 200 DM / salary assignments for at least 1 year

A14 : no checking account

 

No es necesario agrupar ninguna de estas categorías, ni hacer modificaciones en la variable Account Balance.

"""

ACCOUNT_X_CREDIT_NORM.plot.bar(stacked=True, figsize=(10, 10))

ACCOUNT_X_CREDIT_NORM

ACCOUNT_X_CREDIT_NORM*100




CREDIT_X_ACCOUNT= pd.crosstab(credit4['CREDITABILITY'],credit4['ACCOUNT_BALANCE'])


CREDIT_X_ACCOUNT_NORM = CREDIT_X_ACCOUNT.div(CREDIT_X_ACCOUNT.sum(1), axis=0)


CREDIT_X_ACCOUNT_NORM


CREDIT_X_ACCOUNT_NORM.plot.bar()

"""

GRAFICO NORMALIZADO APILADO!!!!


"""

CREDIT_X_ACCOUNT_NORM.plot.bar(stacked=True, figsize=(10, 10))

CREDIT_X_ACCOUNT_NORM

CREDIT_X_ACCOUNT_NORM*100


"""

El P Value de Chi Cuadrado es menor 0,05 = atributos son "dependientes".

"""
from scipy.stats import chi2_contingency
chi2_contingency(CREDIT_X_ACCOUNT)

"""

El P Value de Chi Cuadrado es menor 0,05 = atributos son "dependientes".

"""
from scipy.stats import chi2_contingency
chi2_contingency(ACCOUNT_X_CREDIT)



# endregion 1) Assets("Most valuable available asset"/"Valuables owned by the applicant") x status_residence



# region 2) 'PAYMENT_STATUS_OF_PREVIOUS_CREDIT'/Repayment history x CREDITABILITY


credit4.columns

credit4

credit4['PAYMENT_STATUS_OF_PREVIOUS_CREDIT'] 

credit4['CREDITABILITY'] 



"""

Calculamos cantidad de registros que tiene el Dataset

"""

credit4.index

index = credit4.index

number_of_rows = len(index)

number_of_rows

"""
Contingency Table:
    
"""

PAYMENT_X_CREDIT= pd.crosstab(credit4['PAYMENT_STATUS_OF_PREVIOUS_CREDIT'] , credit4['CREDITABILITY'] )

PAYMENT_X_CREDIT

(PAYMENT_X_CREDIT/number_of_rows)*100

"""

GRAFICO CANTIDAD

"""

PAYMENT_X_CREDIT.plot.bar()

"""

GRAFICO CANTIDAD APILADO!!!!


"""


PAYMENT_X_CREDIT.plot.bar(stacked=True)

# Normalize to sum to 1

PAYMENT_X_CREDIT_NORM = PAYMENT_X_CREDIT.div(PAYMENT_X_CREDIT.sum(1), axis=0)

PAYMENT_X_CREDIT_NORM


PAYMENT_X_CREDIT_NORM.plot.bar()

"""

GRAFICO NORMALIZADO APILADO!!!!


"""


"""

Attribute 3 (A3): 'PAYMENT_STATUS_OF_PREVIOUS_CREDIT'/Repayment history

A30 : no credits taken/ all credits paid back duly
A31 : all credits at this bank paid back duly
A32 : existing credits paid back duly till now
A33 : delay in paying off in the past
A34 : critical account/ other credits existing (not at this bank)

---

Dado que 

1) categorías 2 y 3 tienen igual tasa de malos

2) la categoría 3 pocos registros (tiene 88 registros) 

3) categorías 2 y 3 tienen igual "significado de negocio", son "Credits without delay"

Tuffery AGRUPA categorías 2 y 3.


"""

PAYMENT_X_CREDIT_NORM.plot.bar(stacked=True, figsize=(10, 10))

PAYMENT_X_CREDIT_NORM

PAYMENT_X_CREDIT_NORM*100




CREDIT_X_PAYMENT= pd.crosstab(credit3['CREDITABILITY'],credit3['PAYMENT_STATUS_OF_PREVIOUS_CREDIT'])


CREDIT_X_PAYMENT_NORM = CREDIT_X_PAYMENT.div(CREDIT_X_PAYMENT.sum(1), axis=0)


CREDIT_X_PAYMENT_NORM


CREDIT_X_PAYMENT_NORM.plot.bar()

"""

GRAFICO NORMALIZADO APILADO!!!!


"""

CREDIT_X_PAYMENT_NORM.plot.bar(stacked=True, figsize=(10, 10))

CREDIT_X_PAYMENT_NORM

CREDIT_X_PAYMENT_NORM*100




"""

Generamos nuevo campo calculado 'AGRUP_REPAYMENT_HIST' y agrupamos categorías 2 y 3:

Para generar campos calculados (equivalente al CASE WHEN de SQL en Python Pandas): crear una función y aplicarla sobre
el dataframe.

Creating a new column based on if-elif-else condition:
https://stackoverflow.com/questions/21702342/creating-a-new-column-based-on-if-elif-else-condition


"""


def f(row):
    if row['PAYMENT_STATUS_OF_PREVIOUS_CREDIT'] == 3:
        val = 2
    elif row['PAYMENT_STATUS_OF_PREVIOUS_CREDIT'] == 2:
        val = 2
    else:
        val = row['PAYMENT_STATUS_OF_PREVIOUS_CREDIT']
    return val


credit4['AGRUP_REPAYMENT_HIST'] = credit4.apply(f, axis=1)

credit4['AGRUP_REPAYMENT_HIST'] 


credit4.columns





"""

Calculamos cantidad de registros que tiene el Dataset

"""

credit4.index

index = credit4.index

number_of_rows = len(index)

number_of_rows

"""
Contingency Table:
    
"""

PAYAGRUP_X_CREDIT= pd.crosstab(credit4['AGRUP_REPAYMENT_HIST'] , credit4['CREDITABILITY'] )

PAYAGRUP_X_CREDIT

(PAYAGRUP_X_CREDIT/number_of_rows)*100

"""

GRAFICO CANTIDAD

"""

PAYAGRUP_X_CREDIT.plot.bar()

"""

GRAFICO CANTIDAD APILADO!!!!


"""


PAYAGRUP_X_CREDIT.plot.bar(stacked=True, figsize=(10, 10))

# Normalize to sum to 1

PAYAGRUP_X_CREDIT_NORM = PAYAGRUP_X_CREDIT.div(PAYAGRUP_X_CREDIT.sum(1), axis=0)

PAYAGRUP_X_CREDIT_NORM


PAYAGRUP_X_CREDIT_NORM.plot.bar(figsize=(10, 10))

"""

GRAFICO NORMALIZADO APILADO!!!!


"""

"""

La tasa de malos es distinta entre las distintas categorías de la variable original:
    
"""

PAYAGRUP_X_CREDIT_NORM.plot.bar(stacked=True, figsize=(10, 10))

PAYAGRUP_X_CREDIT_NORM

PAYAGRUP_X_CREDIT_NORM*100




CREDIT_X_PAYAGRUP= pd.crosstab(credit4['CREDITABILITY'],credit4['AGRUP_REPAYMENT_HIST'])


CREDIT_X_PAYAGRUP


CREDIT_X_PAYAGRUP_NORM = CREDIT_X_PAYAGRUP.div(CREDIT_X_PAYAGRUP.sum(1), axis=0)


CREDIT_X_PAYAGRUP_NORM


CREDIT_X_PAYAGRUP_NORM.plot.bar(figsize=(10, 10))

"""

GRAFICO NORMALIZADO APILADO!!!!


"""

CREDIT_X_PAYAGRUP_NORM.plot.bar(stacked=True, figsize=(10, 10))

CREDIT_X_PAYAGRUP_NORM

CREDIT_X_PAYAGRUP_NORM*100




"""

El P Value de Chi Cuadrado es menor 0,05 = atributos son "dependientes".

"""
from scipy.stats import chi2_contingency
chi2_contingency(PAYAGRUP_X_CREDIT)

"""

El P Value de Chi Cuadrado es menor 0,05 = atributos son "dependientes".

"""
from scipy.stats import chi2_contingency
chi2_contingency(CREDIT_X_PAYAGRUP)










# endregion 2) 'PAYMENT_STATUS_OF_PREVIOUS_CREDIT'/Repayment history x CREDITABILITY



# region 3) 'VALUE_SAVINGS_STOCKS'/Savings Outstanding x CREDITABILITY


credit4.columns

credit4

credit4['VALUE_SAVINGS_STOCKS'] 

credit4['CREDITABILITY'] 



"""

Calculamos cantidad de registros que tiene el Dataset

"""

credit4.index

index = credit4.index

number_of_rows = len(index)

number_of_rows

"""
Contingency Table:
    
"""

SAVINGS_X_CREDIT= pd.crosstab(credit4['VALUE_SAVINGS_STOCKS'] , credit4['CREDITABILITY'] )

SAVINGS_X_CREDIT

(SAVINGS_X_CREDIT/number_of_rows)*100

"""

GRAFICO CANTIDAD

"""

SAVINGS_X_CREDIT.plot.bar()

"""

GRAFICO CANTIDAD APILADO!!!!


"""


SAVINGS_X_CREDIT.plot.bar(stacked=True)

# Normalize to sum to 1

SAVINGS_X_CREDIT_NORM = SAVINGS_X_CREDIT.div(SAVINGS_X_CREDIT.sum(1), axis=0)

SAVINGS_X_CREDIT_NORM


SAVINGS_X_CREDIT_NORM.plot.bar()

"""

GRAFICO NORMALIZADO APILADO!!!!


"""


"""

Savings account/bonds (A6):

A6 1 : ... < 100 DM
A6 2 : 100 <= ... < 5 00 DM
A6 3 : 500 <= ... < 1000 DM
A6 4 : .. >= 1000 DM
A6 5 : unknown/ no savings account

---

Dado que 

1)categorias 1 y 2 tienene igual tasa de malos

2)categorias 3 y 4 tienen muy pocos casos ambos (y las tasas de malos es algo parecida)


Tuffery AGRUPA categorías 1 y 2 ,y por otro lado agrupa categorías 3 y 4 (pagina 585 libro Tuffery)



"""

SAVINGS_X_CREDIT_NORM.plot.bar(stacked=True, figsize=(10, 10))

SAVINGS_X_CREDIT_NORM

SAVINGS_X_CREDIT_NORM*100




CREDIT_X_SAVINGS= pd.crosstab(credit3['CREDITABILITY'],credit3['VALUE_SAVINGS_STOCKS'])


CREDIT_X_SAVINGS_NORM = CREDIT_X_SAVINGS.div(CREDIT_X_SAVINGS.sum(1), axis=0)


CREDIT_X_SAVINGS_NORM


CREDIT_X_SAVINGS_NORM.plot.bar()

"""

GRAFICO NORMALIZADO APILADO!!!!


"""

CREDIT_X_SAVINGS_NORM.plot.bar(stacked=True, figsize=(10, 10))

CREDIT_X_SAVINGS_NORM

CREDIT_X_SAVINGS_NORM*100




"""

Generamos nuevo campo calculado 'AGRUP_SAVINGS' y agrupamos categorías 1 y 2 ,y por otro lado agrupa categorías 3 y 4:


Para generar campos calculados (equivalente al CASE WHEN de SQL en Python Pandas): crear una función y aplicarla sobre
el dataframe.

Creating a new column based on if-elif-else condition:
https://stackoverflow.com/questions/21702342/creating-a-new-column-based-on-if-elif-else-condition


"""


def f(row):
    if row['VALUE_SAVINGS_STOCKS'] == 2:
        val = 1
    elif row['VALUE_SAVINGS_STOCKS'] == 4:
        val = 3
    else:
        val = row['VALUE_SAVINGS_STOCKS']
    return val


credit4['AGRUP_SAVINGS'] = credit4.apply(f, axis=1)

credit4['AGRUP_SAVINGS'] 


credit4.columns





"""

Calculamos cantidad de registros que tiene el Dataset

"""

credit4.index

index = credit4.index

number_of_rows = len(index)

number_of_rows

"""
Contingency Table:
    
"""

SAVEAGRUP_X_CREDIT= pd.crosstab(credit4['AGRUP_SAVINGS'] , credit4['CREDITABILITY'] )

SAVEAGRUP_X_CREDIT

(SAVEAGRUP_X_CREDIT/number_of_rows)*100

"""

GRAFICO CANTIDAD

"""

SAVEAGRUP_X_CREDIT.plot.bar()

"""

GRAFICO CANTIDAD APILADO!!!!


"""


SAVEAGRUP_X_CREDIT.plot.bar(stacked=True, figsize=(10, 10))

# Normalize to sum to 1

SAVEAGRUP_X_CREDIT_NORM = SAVEAGRUP_X_CREDIT.div(SAVEAGRUP_X_CREDIT.sum(1), axis=0)

SAVEAGRUP_X_CREDIT_NORM


SAVEAGRUP_X_CREDIT_NORM.plot.bar(figsize=(10, 10))

"""

GRAFICO NORMALIZADO APILADO!!!!


"""

"""

La tasa de malos es distinta entre las distintas categorías de la variable original:
    
"""

SAVEAGRUP_X_CREDIT_NORM.plot.bar(stacked=True, figsize=(10, 10))

SAVEAGRUP_X_CREDIT_NORM

SAVEAGRUP_X_CREDIT_NORM*100




CREDIT_X_SAVEAGRUP= pd.crosstab(credit4['CREDITABILITY'],credit4['AGRUP_SAVINGS'])


CREDIT_X_SAVEAGRUP


CREDIT_X_SAVEAGRUP_NORM = CREDIT_X_SAVEAGRUP.div(CREDIT_X_SAVEAGRUP.sum(1), axis=0)


CREDIT_X_SAVEAGRUP_NORM


CREDIT_X_SAVEAGRUP_NORM.plot.bar(figsize=(10, 10))

"""

GRAFICO NORMALIZADO APILADO!!!!


"""

CREDIT_X_SAVEAGRUP_NORM.plot.bar(stacked=True, figsize=(10, 10))

CREDIT_X_SAVEAGRUP_NORM

CREDIT_X_SAVEAGRUP_NORM*100




"""

El P Value de Chi Cuadrado es menor 0,05 = atributos son "dependientes".

"""
from scipy.stats import chi2_contingency
chi2_contingency(SAVEAGRUP_X_CREDIT)

"""

El P Value de Chi Cuadrado es menor 0,05 = atributos son "dependientes".

"""
from scipy.stats import chi2_contingency
chi2_contingency(CREDIT_X_SAVEAGRUP)





# endregion 3) 'VALUE_SAVINGS_STOCKS'/Savings Outstanding x CREDITABILITY



# region 4) 'PURPOSE'/Purpose of the credit x CREDITABILITY


credit4.columns

credit4

"""

NOTAR: la variable PURPOSE of the credit tiene 10 categorías  en el dataset
mientras que en el libro de Tuffery según el cuadro de la página 587 hay 9 categorías.

La categoría "2nd hand vehicle" que señala Tuffery es el resultado de combinar las
categorías 1 y 10: car (used) y  others.

    
Purpose (A4).

A40 : car (new)
A41 : car (used)
A42 : furniture/eq uipment
A43 : radio/television
A44 : domestic appliances
A45 : repairs
A46 : education
A47 : (vacation - does not exist?)
A48 : retraining
A49 : business
A410 : others
    
    
    
"""
credit4['PURPOSE'].value_counts()

credit4['PURPOSE'] 

credit4['CREDITABILITY'] 



"""

Calculamos cantidad de registros que tiene el Dataset

"""

credit4.index

index = credit4.index

number_of_rows = len(index)

number_of_rows

"""
Contingency Table:
    
"""

PURPOSE_X_CREDIT= pd.crosstab(credit4['PURPOSE'] , credit4['CREDITABILITY'] )

PURPOSE_X_CREDIT

(PURPOSE_X_CREDIT/number_of_rows)*100

"""

GRAFICO CANTIDAD

"""

PURPOSE_X_CREDIT.plot.bar()

"""

GRAFICO CANTIDAD APILADO!!!!


"""


PURPOSE_X_CREDIT.plot.bar(stacked=True)

# Normalize to sum to 1

PURPOSE_X_CREDIT_NORM = PURPOSE_X_CREDIT.div(PURPOSE_X_CREDIT.sum(1), axis=0)

PURPOSE_X_CREDIT_NORM


PURPOSE_X_CREDIT_NORM.plot.bar()

"""

GRAFICO NORMALIZADO APILADO!!!!


"""


"""

Purpose (A4).

A40 : car (new)
A41 : car (used)
A42 : furniture/eq uipment
A43 : radio/television
A44 : domestic appliances
A45 : repairs
A46 : education
A47 : (vacation - does not exist?)
A48 : retraining
A49 : business
A410 : others
---

Dado que 

Las categorías de la variable "Purpose", son numerosas y algunas de ellas son muy pequeñas. Por lo tanto es necesario según libro Tuffery pagina 586 crear combinaciones.

1) Agrupamos categorias 2, 4 y 5, dado que tienen tasas de malos parecidas, y porque podemos agruparlas como 
"accesorios internos del hogar"/"internal fittings".

2) Agrupamos categorias 8 y 6, aunque la tasa de malos es muy distinta entre estas categorías, sin embargo esta diferencia no es relevante dado que 
la categoria 8 tiene sólo 1 caso de Bad credit, lo que provoca que la tasa de malos sea muy alta y sensible. Agrupamos ambas categorías como "Educación".

3) Agrupamos categorias 1 y 10 para estar alineados con Tuffery y tener los mismos resultados.
Como mencionamos previamente, la categoría "2nd hand vehicle" que señala 
Tuffery es el resultado de combinar las
categorías 1 y 10: car (used) y  others.

"""

PURPOSE_X_CREDIT_NORM.plot.bar(stacked=True, figsize=(10, 10))

PURPOSE_X_CREDIT_NORM

PURPOSE_X_CREDIT_NORM*100




CREDIT_X_PURPOSE= pd.crosstab(credit3['CREDITABILITY'],credit3['PURPOSE'])


CREDIT_X_PURPOSE_NORM = CREDIT_X_PURPOSE.div(CREDIT_X_PURPOSE.sum(1), axis=0)


CREDIT_X_PURPOSE_NORM


CREDIT_X_PURPOSE_NORM.plot.bar()


"""

GRAFICO NORMALIZADO APILADO!!!!


"""

CREDIT_X_PURPOSE_NORM.plot.bar(stacked=True, figsize=(10, 10))

CREDIT_X_PURPOSE_NORM

CREDIT_X_PURPOSE_NORM*100




"""

Generamos nuevo campo calculado 'AGRUP_PURPOSE' y agrupamos categorías 2, 4 y 5 ,y por otro lado agrupa categorías 8 y 6:
También agrupamos las categorías 1 y 10.


Para generar campos calculados (equivalente al CASE WHEN de SQL en Python Pandas): crear una función y aplicarla sobre
el dataframe.

Creating a new column based on if-elif-else condition:
https://stackoverflow.com/questions/21702342/creating-a-new-column-based-on-if-elif-else-condition


"""



def f(row):
    if row['PURPOSE'] == 4:
        val = 2
    elif row['PURPOSE'] == 5:
        val = 2
    elif row['PURPOSE'] == 6:
        val = 8
    elif row['PURPOSE'] == 10:
        val = 1
    else:
        val = row['PURPOSE']
    return val



credit4['AGRUP_PURPOSE'] = credit4.apply(f, axis=1)

credit4['AGRUP_PURPOSE'] 


credit4.columns





"""

Calculamos cantidad de registros que tiene el Dataset

"""

credit4.index

index = credit4.index

number_of_rows = len(index)

number_of_rows

"""
Contingency Table:
    
"""

PURPOSEAGRUP_X_CREDIT= pd.crosstab(credit4['AGRUP_PURPOSE'] , credit4['CREDITABILITY'] )

PURPOSEAGRUP_X_CREDIT

(PURPOSEAGRUP_X_CREDIT/number_of_rows)*100

"""

GRAFICO CANTIDAD

"""

PURPOSEAGRUP_X_CREDIT.plot.bar()

"""

GRAFICO CANTIDAD APILADO!!!!


"""


PURPOSEAGRUP_X_CREDIT.plot.bar(stacked=True, figsize=(10, 10))

# Normalize to sum to 1

PURPOSEAGRUP_X_CREDIT_NORM = PURPOSEAGRUP_X_CREDIT.div(PURPOSEAGRUP_X_CREDIT.sum(1), axis=0)

PURPOSEAGRUP_X_CREDIT_NORM


PURPOSEAGRUP_X_CREDIT_NORM.plot.bar(figsize=(10, 10))

"""

GRAFICO NORMALIZADO APILADO!!!!


"""

"""

La tasa de malos es distinta entre las distintas categorías de la variable original:
    
"""

PURPOSEAGRUP_X_CREDIT_NORM.plot.bar(stacked=True, figsize=(10, 10))

PURPOSEAGRUP_X_CREDIT_NORM

PURPOSEAGRUP_X_CREDIT_NORM*100




CREDIT_X_PURPOSEAGRUP= pd.crosstab(credit4['CREDITABILITY'],credit4['AGRUP_PURPOSE'])


CREDIT_X_PURPOSEAGRUP


CREDIT_X_PURPOSEAGRUP_NORM = CREDIT_X_PURPOSEAGRUP.div(CREDIT_X_PURPOSEAGRUP.sum(1), axis=0)


CREDIT_X_PURPOSEAGRUP_NORM


CREDIT_X_PURPOSEAGRUP_NORM.plot.bar(figsize=(10, 10))

"""

GRAFICO NORMALIZADO APILADO!!!!


"""

CREDIT_X_PURPOSEAGRUP_NORM.plot.bar(stacked=True, figsize=(10, 10))

CREDIT_X_PURPOSEAGRUP_NORM

CREDIT_X_PURPOSEAGRUP_NORM*100




"""

El P Value de Chi Cuadrado es menor 0,05 = atributos son "dependientes".

"""
from scipy.stats import chi2_contingency
chi2_contingency(PURPOSEAGRUP_X_CREDIT)

"""

El P Value de Chi Cuadrado es menor 0,05 = atributos son "dependientes".

"""
from scipy.stats import chi2_contingency
chi2_contingency(CREDIT_X_PURPOSEAGRUP)





# endregion 4) 'PURPOSE'/Purpose of the credit x CREDITABILITY



# region 5) 'MOST_VALUABLE_AVAILABLE_ASSET'/Valuables owned by the applicant x CREDITABILITY


credit4.columns

credit4

credit4['MOST_VALUABLE_AVAILABLE_ASSET'] 

credit4['CREDITABILITY'] 



"""

Calculamos cantidad de registros que tiene el Dataset

"""

credit4.index

index = credit4.index

number_of_rows = len(index)

number_of_rows

"""
Contingency Table:
    
"""

ASSET_X_CREDIT= pd.crosstab(credit4['MOST_VALUABLE_AVAILABLE_ASSET'] , credit4['CREDITABILITY'] )

ASSET_X_CREDIT

(ASSET_X_CREDIT/number_of_rows)*100

"""

GRAFICO CANTIDAD

"""

ASSET_X_CREDIT.plot.bar()

"""

GRAFICO CANTIDAD APILADO!!!!


"""


ASSET_X_CREDIT.plot.bar(stacked=True)

# Normalize to sum to 1

ASSET_X_CREDIT_NORM = ASSET_X_CREDIT.div(ASSET_X_CREDIT.sum(1), axis=0)

ASSET_X_CREDIT_NORM


ASSET_X_CREDIT_NORM.plot.bar()

"""

GRAFICO NORMALIZADO APILADO!!!!


"""


"""

Property (A12) /Assets/"Most valuable available asset"/"Valuables owned by the applicant":
 
A121 : real estate
A122 : if not A121 : building society savings agreement/ life insurance
A123 : if not A121/A122 : car or other, not in attribute 6
A124 : unknown / no property
---
 
Dado que
 
Agrupamos categorias 2 y 3, dado que tienen igual tasas de malos. Agrupamos ambas categorias como "Other assets".
 


"""


ASSET_X_CREDIT_NORM.plot.bar(stacked=True, figsize=(10, 10))

ASSET_X_CREDIT_NORM

ASSET_X_CREDIT_NORM*100




CREDIT_X_ASSET= pd.crosstab(credit3['CREDITABILITY'],credit3['MOST_VALUABLE_AVAILABLE_ASSET'])


CREDIT_X_ASSET_NORM = CREDIT_X_ASSET.div(CREDIT_X_ASSET.sum(1), axis=0)


CREDIT_X_ASSET_NORM


CREDIT_X_ASSET_NORM.plot.bar()


"""

GRAFICO NORMALIZADO APILADO!!!!


"""

CREDIT_X_ASSET_NORM.plot.bar(stacked=True, figsize=(10, 10))

CREDIT_X_ASSET_NORM

CREDIT_X_ASSET_NORM*100




"""

Generamos nuevo campo calculado 'AGRUP_ASSET' y agrupamos categorías categorias 2 y 3:


Para generar campos calculados (equivalente al CASE WHEN de SQL en Python Pandas): crear una función y aplicarla sobre
el dataframe.

Creating a new column based on if-elif-else condition:
https://stackoverflow.com/questions/21702342/creating-a-new-column-based-on-if-elif-else-condition


"""



def f(row):
    if row['MOST_VALUABLE_AVAILABLE_ASSET'] == 3:
        val = 2
    elif row['MOST_VALUABLE_AVAILABLE_ASSET'] == 2:
        val = 2
    else:
        val = row['MOST_VALUABLE_AVAILABLE_ASSET']
    return val



credit4['AGRUP_ASSET'] = credit4.apply(f, axis=1)

credit4['AGRUP_ASSET'] 


credit4.columns





"""

Calculamos cantidad de registros que tiene el Dataset

"""

credit4.index

index = credit4.index

number_of_rows = len(index)

number_of_rows

"""
Contingency Table:
    
"""

AGRUP_ASSET_X_CREDIT= pd.crosstab(credit4['AGRUP_ASSET'] , credit4['CREDITABILITY'] )

AGRUP_ASSET_X_CREDIT

(AGRUP_ASSET_X_CREDIT/number_of_rows)*100

"""

GRAFICO CANTIDAD

"""

AGRUP_ASSET_X_CREDIT.plot.bar()

"""

GRAFICO CANTIDAD APILADO!!!!


"""


AGRUP_ASSET_X_CREDIT.plot.bar(stacked=True, figsize=(10, 10))

# Normalize to sum to 1

AGRUP_ASSET_X_CREDIT_NORM = AGRUP_ASSET_X_CREDIT.div(AGRUP_ASSET_X_CREDIT.sum(1), axis=0)

AGRUP_ASSET_X_CREDIT_NORM


AGRUP_ASSET_X_CREDIT_NORM.plot.bar(figsize=(10, 10))

"""

GRAFICO NORMALIZADO APILADO!!!!


"""

"""

La tasa de malos es distinta entre las distintas categorías de la variable original:
    
"""

AGRUP_ASSET_X_CREDIT_NORM.plot.bar(stacked=True, figsize=(10, 10))

AGRUP_ASSET_X_CREDIT_NORM

AGRUP_ASSET_X_CREDIT_NORM*100




CREDIT_X_AGRUP_ASSET= pd.crosstab(credit4['CREDITABILITY'],credit4['AGRUP_ASSET'])


CREDIT_X_AGRUP_ASSET


CREDIT_X_AGRUP_ASSET_NORM = CREDIT_X_AGRUP_ASSET.div(CREDIT_X_AGRUP_ASSET.sum(1), axis=0)


CREDIT_X_AGRUP_ASSET_NORM


CREDIT_X_AGRUP_ASSET_NORM.plot.bar(figsize=(10, 10))

"""

GRAFICO NORMALIZADO APILADO!!!!


"""

CREDIT_X_AGRUP_ASSET_NORM.plot.bar(stacked=True, figsize=(10, 10))

CREDIT_X_AGRUP_ASSET_NORM

CREDIT_X_AGRUP_ASSET_NORM*100




"""

El P Value de Chi Cuadrado es menor 0,05 = atributos son "dependientes".

"""
from scipy.stats import chi2_contingency
chi2_contingency(AGRUP_ASSET_X_CREDIT)

"""

El P Value de Chi Cuadrado es menor 0,05 = atributos son "dependientes".

"""
from scipy.stats import chi2_contingency
chi2_contingency(CREDIT_X_AGRUP_ASSET)





# endregion 5) 'MOST_VALUABLE_AVAILABLE_ASSET'/Valuables owned by the applicant x CREDITABILITY



# region 6) 'LENGTH_OF_CURRENT_EMPLOYMENT' x CREDITABILITY


credit4.columns

credit4

credit4['LENGTH_OF_CURRENT_EMPLOYMENT'] 

credit4['CREDITABILITY'] 



"""

Calculamos cantidad de registros que tiene el Dataset

"""

credit4.index

index = credit4.index

number_of_rows = len(index)

number_of_rows

"""
Contingency Table:
    
"""

LENGTH_EMPLOY_X_CREDIT= pd.crosstab(credit4['LENGTH_OF_CURRENT_EMPLOYMENT'] , credit4['CREDITABILITY'] )

LENGTH_EMPLOY_X_CREDIT

(LENGTH_EMPLOY_X_CREDIT/number_of_rows)*100

"""

GRAFICO CANTIDAD

"""

LENGTH_EMPLOY_X_CREDIT.plot.bar()

"""

GRAFICO CANTIDAD APILADO!!!!


"""


LENGTH_EMPLOY_X_CREDIT.plot.bar(stacked=True)

# Normalize to sum to 1

LENGTH_EMPLOY_X_CREDIT_NORM = LENGTH_EMPLOY_X_CREDIT.div(LENGTH_EMPLOY_X_CREDIT.sum(1), axis=0)

LENGTH_EMPLOY_X_CREDIT_NORM


LENGTH_EMPLOY_X_CREDIT_NORM.plot.bar()

"""

GRAFICO NORMALIZADO APILADO!!!!


"""


"""

 
Seniority/Length of current employment/"Present employment since" (A7):
 
A71 : unemployed
A72 : ... < 1 year
A73 : 1 <= ... < 4 years
A74 : 4 <= ... < 7 years
A75 : .. >= 7 years
---
 
Dado que
 
1) Agrupamos categorias 1 y 2 , dado que son categorías que contienen pocos datos.

 

"""


LENGTH_EMPLOY_X_CREDIT_NORM.plot.bar(stacked=True, figsize=(10, 10))

LENGTH_EMPLOY_X_CREDIT_NORM

LENGTH_EMPLOY_X_CREDIT_NORM*100




CREDIT_X_LENGTH_EMPLOY= pd.crosstab(credit3['CREDITABILITY'],credit3['LENGTH_OF_CURRENT_EMPLOYMENT'])


CREDIT_X_LENGTH_EMPLOY_NORM = CREDIT_X_LENGTH_EMPLOY.div(CREDIT_X_LENGTH_EMPLOY.sum(1), axis=0)


CREDIT_X_LENGTH_EMPLOY_NORM


CREDIT_X_LENGTH_EMPLOY_NORM.plot.bar()


"""

GRAFICO NORMALIZADO APILADO!!!!


"""

CREDIT_X_LENGTH_EMPLOY_NORM.plot.bar(stacked=True, figsize=(10, 10))

CREDIT_X_LENGTH_EMPLOY_NORM

CREDIT_X_LENGTH_EMPLOY_NORM*100




"""

Generamos nuevo campo calculado 'AGRUP_LENGTH_EMPLOY' y agrupamos categorias 1 y 2:


Para generar campos calculados (equivalente al CASE WHEN de SQL en Python Pandas): crear una función y aplicarla sobre
el dataframe.

Creating a new column based on if-elif-else condition:
https://stackoverflow.com/questions/21702342/creating-a-new-column-based-on-if-elif-else-condition


"""



def f(row):
    if row['LENGTH_OF_CURRENT_EMPLOYMENT'] == 1:
        val = 1
    elif row['LENGTH_OF_CURRENT_EMPLOYMENT'] == 2:
        val = 1
    else:
        val = row['LENGTH_OF_CURRENT_EMPLOYMENT']
    return val



credit4['AGRUP_LENGTH_EMPLOY'] = credit4.apply(f, axis=1)

credit4['AGRUP_LENGTH_EMPLOY'] 


credit4.columns





"""

Calculamos cantidad de registros que tiene el Dataset

"""

credit4.index

index = credit4.index

number_of_rows = len(index)

number_of_rows

"""
Contingency Table:
    
"""

AGRUP_LENGTH_EMPLOY_X_CREDIT= pd.crosstab(credit4['AGRUP_LENGTH_EMPLOY'] , credit4['CREDITABILITY'] )

AGRUP_LENGTH_EMPLOY_X_CREDIT

(AGRUP_LENGTH_EMPLOY_X_CREDIT/number_of_rows)*100

"""

GRAFICO CANTIDAD

"""

AGRUP_LENGTH_EMPLOY_X_CREDIT.plot.bar()

"""

GRAFICO CANTIDAD APILADO!!!!


"""


AGRUP_LENGTH_EMPLOY_X_CREDIT.plot.bar(stacked=True, figsize=(10, 10))

# Normalize to sum to 1

AGRUP_LENGTH_EMPLOY_X_CREDIT_NORM = AGRUP_LENGTH_EMPLOY_X_CREDIT.div(AGRUP_LENGTH_EMPLOY_X_CREDIT.sum(1), axis=0)

AGRUP_LENGTH_EMPLOY_X_CREDIT_NORM


AGRUP_LENGTH_EMPLOY_X_CREDIT_NORM.plot.bar(figsize=(10, 10))

"""

GRAFICO NORMALIZADO APILADO!!!!


"""

"""

La tasa de malos es distinta entre las distintas categorías de la variable original:
    
"""

AGRUP_LENGTH_EMPLOY_X_CREDIT_NORM.plot.bar(stacked=True, figsize=(10, 10))

AGRUP_LENGTH_EMPLOY_X_CREDIT_NORM

AGRUP_LENGTH_EMPLOY_X_CREDIT_NORM*100




CREDIT_X_AGRUP_LENGTH_EMPLOY= pd.crosstab(credit4['CREDITABILITY'],credit4['AGRUP_LENGTH_EMPLOY'])


CREDIT_X_AGRUP_LENGTH_EMPLOY


CREDIT_X_AGRUP_LENGTH_EMPLOY_NORM = CREDIT_X_AGRUP_LENGTH_EMPLOY.div(CREDIT_X_AGRUP_LENGTH_EMPLOY.sum(1), axis=0)


CREDIT_X_AGRUP_LENGTH_EMPLOY_NORM


CREDIT_X_AGRUP_LENGTH_EMPLOY_NORM.plot.bar(figsize=(10, 10))

"""

GRAFICO NORMALIZADO APILADO!!!!


"""

CREDIT_X_AGRUP_LENGTH_EMPLOY_NORM.plot.bar(stacked=True, figsize=(10, 10))

CREDIT_X_AGRUP_LENGTH_EMPLOY_NORM

CREDIT_X_AGRUP_LENGTH_EMPLOY_NORM*100




"""

El P Value de Chi Cuadrado es menor 0,05 = atributos son "dependientes".

"""
from scipy.stats import chi2_contingency
chi2_contingency(AGRUP_LENGTH_EMPLOY_X_CREDIT)

"""

El P Value de Chi Cuadrado es menor 0,05 = atributos son "dependientes".

"""
from scipy.stats import chi2_contingency
chi2_contingency(CREDIT_X_AGRUP_LENGTH_EMPLOY)





# endregion 6) 'LENGTH_OF_CURRENT_EMPLOYMENT' x CREDITABILITY



# region 7) 'TYPE_OF_APARTMENT'/Residential status x CREDITABILITY


credit4.columns

credit4

credit4['TYPE_OF_APARTMENT'] 

credit4['CREDITABILITY'] 



"""

Calculamos cantidad de registros que tiene el Dataset

"""

credit4.index

index = credit4.index

number_of_rows = len(index)

number_of_rows

"""
Contingency Table:
    
"""

TYPE_APART_X_CREDIT= pd.crosstab(credit4['TYPE_OF_APARTMENT'] , credit4['CREDITABILITY'] )

TYPE_APART_X_CREDIT

(TYPE_APART_X_CREDIT/number_of_rows)*100

"""

GRAFICO CANTIDAD

"""

TYPE_APART_X_CREDIT.plot.bar()

"""

GRAFICO CANTIDAD APILADO!!!!


"""


TYPE_APART_X_CREDIT.plot.bar(stacked=True)

# Normalize to sum to 1

TYPE_APART_X_CREDIT_NORM = TYPE_APART_X_CREDIT.div(TYPE_APART_X_CREDIT.sum(1), axis=0)

TYPE_APART_X_CREDIT_NORM


TYPE_APART_X_CREDIT_NORM.plot.bar()

"""

GRAFICO NORMALIZADO APILADO!!!!


"""


"""

 
Residential status/Type of apartment/Housing (A15):
 
A151 : rent
A152 : own
A153 : for free
---
 
Dado que
 
1) Agrupamos categorias 1 y 3 , dado que son categorías que contienen pocos datos. Podemos agruparlas como  "Not Owner"/"No Dueños"

 

"""


TYPE_APART_X_CREDIT_NORM.plot.bar(stacked=True, figsize=(10, 10))

TYPE_APART_X_CREDIT_NORM

TYPE_APART_X_CREDIT_NORM*100




CREDIT_X_TYPE_APART= pd.crosstab(credit3['CREDITABILITY'],credit3['TYPE_OF_APARTMENT'])


CREDIT_X_TYPE_APART_NORM = CREDIT_X_TYPE_APART.div(CREDIT_X_TYPE_APART.sum(1), axis=0)


CREDIT_X_TYPE_APART_NORM


CREDIT_X_TYPE_APART_NORM.plot.bar()


"""

GRAFICO NORMALIZADO APILADO!!!!


"""

CREDIT_X_TYPE_APART_NORM.plot.bar(stacked=True, figsize=(10, 10))

CREDIT_X_TYPE_APART_NORM

CREDIT_X_TYPE_APART_NORM*100




"""

Generamos nuevo campo calculado 'AGRUP_TYPE_APART' y agrupamos categorias 1 y 3:


Para generar campos calculados (equivalente al CASE WHEN de SQL en Python Pandas): crear una función y aplicarla sobre
el dataframe.

Creating a new column based on if-elif-else condition:
https://stackoverflow.com/questions/21702342/creating-a-new-column-based-on-if-elif-else-condition


"""



def f(row):
    if row['TYPE_OF_APARTMENT'] == 3:
        val = 1
    elif row['TYPE_OF_APARTMENT'] == 1:
        val = 1
    else:
        val = row['TYPE_OF_APARTMENT']
    return val



credit4['AGRUP_TYPE_APART'] = credit4.apply(f, axis=1)

credit4['AGRUP_TYPE_APART'] 


credit4.columns





"""

Calculamos cantidad de registros que tiene el Dataset

"""

credit4.index

index = credit4.index

number_of_rows = len(index)

number_of_rows

"""
Contingency Table:
    
"""

AGRUP_TYPE_APART_X_CREDIT= pd.crosstab(credit4['AGRUP_TYPE_APART'] , credit4['CREDITABILITY'] )

AGRUP_TYPE_APART_X_CREDIT

(AGRUP_TYPE_APART_X_CREDIT/number_of_rows)*100

"""

GRAFICO CANTIDAD

"""

AGRUP_TYPE_APART_X_CREDIT.plot.bar()

"""

GRAFICO CANTIDAD APILADO!!!!


"""


AGRUP_TYPE_APART_X_CREDIT.plot.bar(stacked=True, figsize=(10, 10))

# Normalize to sum to 1

AGRUP_TYPE_APART_X_CREDIT_NORM = AGRUP_TYPE_APART_X_CREDIT.div(AGRUP_TYPE_APART_X_CREDIT.sum(1), axis=0)

AGRUP_TYPE_APART_X_CREDIT_NORM


AGRUP_TYPE_APART_X_CREDIT_NORM.plot.bar(figsize=(10, 10))

"""

GRAFICO NORMALIZADO APILADO!!!!


"""

"""

La tasa de malos es distinta entre las distintas categorías de la variable original:
    
"""

AGRUP_TYPE_APART_X_CREDIT_NORM.plot.bar(stacked=True, figsize=(10, 10))

AGRUP_TYPE_APART_X_CREDIT_NORM

AGRUP_TYPE_APART_X_CREDIT_NORM*100




CREDIT_X_AGRUP_TYPE_APART= pd.crosstab(credit4['CREDITABILITY'],credit4['AGRUP_TYPE_APART'])


CREDIT_X_AGRUP_TYPE_APART


CREDIT_X_AGRUP_TYPE_APART_NORM = CREDIT_X_AGRUP_TYPE_APART.div(CREDIT_X_AGRUP_TYPE_APART.sum(1), axis=0)


CREDIT_X_AGRUP_TYPE_APART_NORM


CREDIT_X_AGRUP_TYPE_APART_NORM.plot.bar(figsize=(10, 10))

"""

GRAFICO NORMALIZADO APILADO!!!!


"""

CREDIT_X_AGRUP_TYPE_APART_NORM.plot.bar(stacked=True, figsize=(10, 10))

CREDIT_X_AGRUP_TYPE_APART_NORM

CREDIT_X_AGRUP_TYPE_APART_NORM*100




"""

El P Value de Chi Cuadrado es menor 0,05 = atributos son "dependientes".

"""
from scipy.stats import chi2_contingency
chi2_contingency(AGRUP_TYPE_APART_X_CREDIT)

"""

El P Value de Chi Cuadrado es menor 0,05 = atributos son "dependientes".

"""
from scipy.stats import chi2_contingency
chi2_contingency(CREDIT_X_AGRUP_TYPE_APART)





# endregion 7) 'TYPE_OF_APARTMENT'/Residential status x CREDITABILITY



# region 8) 'age_bin_optimal'/ Bin Age x CREDITABILITY


credit4.columns

credit4

credit4['age_bin_optimal'] 

credit4['CREDITABILITY'] 



"""

Calculamos cantidad de registros que tiene el Dataset

"""

credit4.index

index = credit4.index

number_of_rows = len(index)

number_of_rows

"""
Contingency Table:
    
"""

AGE_BIN_OPT_X_CREDIT= pd.crosstab(credit4['age_bin_optimal'] , credit4['CREDITABILITY'] )

AGE_BIN_OPT_X_CREDIT

(AGE_BIN_OPT_X_CREDIT/number_of_rows)*100

"""

GRAFICO CANTIDAD

"""

AGE_BIN_OPT_X_CREDIT.plot.bar()

"""

GRAFICO CANTIDAD APILADO!!!!


"""


AGE_BIN_OPT_X_CREDIT.plot.bar(stacked=True)

# Normalize to sum to 1


AGE_BIN_OPT_X_CREDIT_NORM = AGE_BIN_OPT_X_CREDIT.div(AGE_BIN_OPT_X_CREDIT.sum(1), axis=0)


AGE_BIN_OPT_X_CREDIT_NORM


AGE_BIN_OPT_X_CREDIT_NORM.plot.bar()


"""

GRAFICO NORMALIZADO APILADO!!!!


"""


"""

 
A la variable Bin_Age no es necesario hacer cambios según libro Tuffery página 590.

 
"""


AGE_BIN_OPT_X_CREDIT_NORM.plot.bar(stacked=True, figsize=(10, 10))

AGE_BIN_OPT_X_CREDIT_NORM

AGE_BIN_OPT_X_CREDIT_NORM*100




CREDIT_X_AGE_BIN_OPT= pd.crosstab(credit3['CREDITABILITY'],credit3['age_bin_optimal'])


CREDIT_X_AGE_BIN_OPT_NORM = CREDIT_X_AGE_BIN_OPT.div(CREDIT_X_AGE_BIN_OPT.sum(1), axis=0)


CREDIT_X_AGE_BIN_OPT_NORM



"""

GRAFICO CANTIDAD

ATENCION!!! por alguna razon no puedo generar graficos con el dataframe CREDIT_X_AGE_BIN_OPT_NORM, aparentemente es un bug / error de Python.
Para solucionar esto, lo que hay que hacer es cambiar los nombres de la etiquetas originales (los cuales se generaban automáticamente a partir 
de la función pd.cut para generar cuartiles).

https://stackoverflow.com/questions/45424753/pandas-buffer-has-wrong-number-of-dimensions

"""

CREDIT_X_AGE_BIN_OPT_NORM.columns

CREDIT_X_AGE_BIN_OPT_NORM.columns = ['0-25', '26 o Más']




CREDIT_X_AGE_BIN_OPT_NORM.plot.bar()


"""

GRAFICO NORMALIZADO APILADO!!!!


"""

CREDIT_X_AGE_BIN_OPT_NORM.plot.bar(stacked=True, figsize=(10, 10))

CREDIT_X_AGE_BIN_OPT_NORM

CREDIT_X_AGE_BIN_OPT_NORM*100





"""

El P Value de Chi Cuadrado es menor 0,05 = atributos son "dependientes".

"""
from scipy.stats import chi2_contingency
chi2_contingency(AGE_BIN_OPT_X_CREDIT)

"""

El P Value de Chi Cuadrado es menor 0,05 = atributos son "dependientes".

"""
from scipy.stats import chi2_contingency
chi2_contingency(CREDIT_X_AGE_BIN_OPT)




# endregion 8) 'age_bin_optimal'/ Bin Age x CREDITABILITY



# region 9) 'CONCURRENT_CREDITS' / Other borrowings (outside the bank) x CREDITABILITY


credit4.columns

credit4

credit4['CONCURRENT_CREDITS'] 

credit4['CREDITABILITY'] 



"""

Calculamos cantidad de registros que tiene el Dataset

"""

credit4.index

index = credit4.index

number_of_rows = len(index)

number_of_rows

"""
Contingency Table:
    
"""

OTHER_CREDITS_X_CREDIT= pd.crosstab(credit4['CONCURRENT_CREDITS'] , credit4['CREDITABILITY'] )

OTHER_CREDITS_X_CREDIT

(OTHER_CREDITS_X_CREDIT/number_of_rows)*100

"""

GRAFICO CANTIDAD

"""

OTHER_CREDITS_X_CREDIT.plot.bar()

"""

GRAFICO CANTIDAD APILADO!!!!


"""


OTHER_CREDITS_X_CREDIT.plot.bar(stacked=True)

# Normalize to sum to 1

OTHER_CREDITS_X_CREDIT_NORM = OTHER_CREDITS_X_CREDIT.div(OTHER_CREDITS_X_CREDIT.sum(1), axis=0)

OTHER_CREDITS_X_CREDIT_NORM


OTHER_CREDITS_X_CREDIT_NORM.plot.bar()

"""

GRAFICO NORMALIZADO APILADO!!!!


"""


"""

Other_Credits/Other installment plans (A14):
 
A141 : bank
A142 : stores
A143 : none
 
---
 
Dado que
 
Agrupamos categorias 1 y 2 dado que tienen tasas de malos parecidas. La naturaleza de la 
institucion bancaria (convencional o especializada) no hace diferencia en las tasas de malos. Podemos agruparlas como
"Other banks or Institutions".


"""


OTHER_CREDITS_X_CREDIT_NORM.plot.bar(stacked=True, figsize=(10, 10))

OTHER_CREDITS_X_CREDIT_NORM

OTHER_CREDITS_X_CREDIT_NORM*100




CREDIT_X_OTHER_CREDITS= pd.crosstab(credit3['CREDITABILITY'],credit3['CONCURRENT_CREDITS'])


CREDIT_X_OTHER_CREDITS_NORM = CREDIT_X_OTHER_CREDITS.div(CREDIT_X_OTHER_CREDITS.sum(1), axis=0)


CREDIT_X_OTHER_CREDITS_NORM


CREDIT_X_OTHER_CREDITS_NORM.plot.bar()


"""

GRAFICO NORMALIZADO APILADO!!!!


"""

CREDIT_X_OTHER_CREDITS_NORM.plot.bar(stacked=True, figsize=(10, 10))

CREDIT_X_OTHER_CREDITS_NORM

CREDIT_X_OTHER_CREDITS_NORM*100




"""

Generamos nuevo campo calculado 'AGRUP_OTHER_CREDITS' y agrupamos categorias 1 y 2:


Para generar campos calculados (equivalente al CASE WHEN de SQL en Python Pandas): crear una función y aplicarla sobre
el dataframe.

Creating a new column based on if-elif-else condition:
https://stackoverflow.com/questions/21702342/creating-a-new-column-based-on-if-elif-else-condition


"""



def f(row):
    if row['CONCURRENT_CREDITS'] == 2:
        val = 1
    elif row['CONCURRENT_CREDITS'] == 1:
        val = 1
    else:
        val = row['CONCURRENT_CREDITS']
    return val



credit4['AGRUP_OTHER_CREDITS'] = credit4.apply(f, axis=1)

credit4['AGRUP_OTHER_CREDITS'] 


credit4.columns





"""

Calculamos cantidad de registros que tiene el Dataset

"""

credit4.index

index = credit4.index

number_of_rows = len(index)

number_of_rows

"""
Contingency Table:
    
"""

AGRUP_OTHER_CREDITS_X_CREDIT= pd.crosstab(credit4['AGRUP_OTHER_CREDITS'] , credit4['CREDITABILITY'] )

AGRUP_OTHER_CREDITS_X_CREDIT

(AGRUP_OTHER_CREDITS_X_CREDIT/number_of_rows)*100

"""

GRAFICO CANTIDAD

"""

AGRUP_OTHER_CREDITS_X_CREDIT.plot.bar()

"""

GRAFICO CANTIDAD APILADO!!!!


"""


AGRUP_OTHER_CREDITS_X_CREDIT.plot.bar(stacked=True, figsize=(10, 10))

# Normalize to sum to 1

AGRUP_OTHER_CREDITS_X_CREDIT_NORM = AGRUP_OTHER_CREDITS_X_CREDIT.div(AGRUP_OTHER_CREDITS_X_CREDIT.sum(1), axis=0)

AGRUP_OTHER_CREDITS_X_CREDIT_NORM


AGRUP_OTHER_CREDITS_X_CREDIT_NORM.plot.bar(figsize=(10, 10))

"""

GRAFICO NORMALIZADO APILADO!!!!


"""

"""

La tasa de malos es distinta entre las distintas categorías de la variable original:
    
"""

AGRUP_OTHER_CREDITS_X_CREDIT_NORM.plot.bar(stacked=True, figsize=(10, 10))

AGRUP_OTHER_CREDITS_X_CREDIT_NORM

AGRUP_OTHER_CREDITS_X_CREDIT_NORM*100




CREDIT_X_AGRUP_OTHER_CREDITS= pd.crosstab(credit4['CREDITABILITY'],credit4['AGRUP_OTHER_CREDITS'])


CREDIT_X_AGRUP_OTHER_CREDITS


CREDIT_X_AGRUP_OTHER_CREDITS_NORM = CREDIT_X_AGRUP_OTHER_CREDITS.div(CREDIT_X_AGRUP_OTHER_CREDITS.sum(1), axis=0)


CREDIT_X_AGRUP_OTHER_CREDITS_NORM


CREDIT_X_AGRUP_OTHER_CREDITS_NORM.plot.bar(figsize=(10, 10))

"""

GRAFICO NORMALIZADO APILADO!!!!


"""

CREDIT_X_AGRUP_OTHER_CREDITS_NORM.plot.bar(stacked=True, figsize=(10, 10))

CREDIT_X_AGRUP_OTHER_CREDITS_NORM

CREDIT_X_AGRUP_OTHER_CREDITS_NORM*100




"""

El P Value de Chi Cuadrado es menor 0,05 = atributos son "dependientes".

"""
from scipy.stats import chi2_contingency
chi2_contingency(AGRUP_OTHER_CREDITS_X_CREDIT)

"""

El P Value de Chi Cuadrado es menor 0,05 = atributos son "dependientes".

"""
from scipy.stats import chi2_contingency
chi2_contingency(CREDIT_X_AGRUP_OTHER_CREDITS)





# endregion 9) 'CONCURRENT_CREDITS' / Other borrowings (outside the bank) x CREDITABILITY



# region 10) 'SEX_AND_MARITAL_STATUS' / Marital status x CREDITABILITY


credit4.columns

credit4

credit4['SEX_AND_MARITAL_STATUS'] 

credit4['CREDITABILITY'] 



"""

Calculamos cantidad de registros que tiene el Dataset

"""

credit4.index

index = credit4.index

number_of_rows = len(index)

number_of_rows

"""
Contingency Table:
    
"""

MARITAL_STATUS_X_CREDIT= pd.crosstab(credit4['SEX_AND_MARITAL_STATUS'] , credit4['CREDITABILITY'] )

MARITAL_STATUS_X_CREDIT

(MARITAL_STATUS_X_CREDIT/number_of_rows)*100

"""

GRAFICO CANTIDAD

"""

MARITAL_STATUS_X_CREDIT.plot.bar()

"""

GRAFICO CANTIDAD APILADO!!!!


"""


MARITAL_STATUS_X_CREDIT.plot.bar(stacked=True)

# Normalize to sum to 1

MARITAL_STATUS_X_CREDIT_NORM = MARITAL_STATUS_X_CREDIT.div(MARITAL_STATUS_X_CREDIT.sum(1), axis=0)

MARITAL_STATUS_X_CREDIT_NORM


MARITAL_STATUS_X_CREDIT_NORM.plot.bar()

"""

GRAFICO NORMALIZADO APILADO!!!!


"""


"""


Marital Status / "Personal status and sex" (A9):
 
A91 : male : divorced/separated
A92 : female : divorced/separated/married
A93 : male : single
A94 : male : married/widowed
A95 : female : single
 
---
 
1) Agrupamos categorias 3 y 4, dado que tienen tasas de malos parecidas
 
Teniendo en cuenta la SELECCION DE VARIABLES CATEGORICAS que se hizo previamente con el criterio de V-CRAMER, la 
variable "Marital Status" tiene poco poder de discriminación: es poca la diferencia que hay en la proporción de 
clientes MALOS y BUENOS entre las distintas categorías de la variable "Marital Status". Las categorías con 
proporciones parecidas de clientes malos serán agrupadas: "male: single" (3) y "male : married/widowed" (4). 
No obstante, Tuffery advierte que es poco probable que la variable "Marital Status" juegue un rol importante en la predicción.
 
 
2)
Otra observación de Tuffery pagina 591: supuestamente la variable "Marital Status" tiene 5 categorías, sin 
embargo en la muestra de datos que utilizamos sólo aparecen 4 de estas 5 categorías: no hay datos que 
correspondan a la categoría "female : single" (5). Por lo tanto el modelo de score será desarrollado 
sin esta categoría, y no será posible scorear aquellas aplicaciones/solicitudes de créditos que 
correspondan a la categoría "female : single" (5). Para evitar esta situación, Tuffery sugiere 
aplicar una regla "a priori" para los casos "female : single" (5), que podrá ser validada por 
los analistas de créditos: esta regla establecerá que la categoría "female : single" (5) será 
combinada con una de las otras categorías existentes, ya sea "female : divorced/separated/married" (2) o "male : single" (3).


"""


MARITAL_STATUS_X_CREDIT_NORM.plot.bar(stacked=True, figsize=(10, 10))

MARITAL_STATUS_X_CREDIT_NORM

MARITAL_STATUS_X_CREDIT_NORM*100




CREDIT_X_MARITAL_STATUS= pd.crosstab(credit3['CREDITABILITY'],credit3['SEX_AND_MARITAL_STATUS'])


CREDIT_X_MARITAL_STATUS_NORM = CREDIT_X_MARITAL_STATUS.div(CREDIT_X_MARITAL_STATUS.sum(1), axis=0)


CREDIT_X_MARITAL_STATUS_NORM


CREDIT_X_MARITAL_STATUS_NORM.plot.bar()


"""

GRAFICO NORMALIZADO APILADO!!!!


"""

CREDIT_X_MARITAL_STATUS_NORM.plot.bar(stacked=True, figsize=(10, 10))

CREDIT_X_MARITAL_STATUS_NORM

CREDIT_X_MARITAL_STATUS_NORM*100




"""

Generamos nuevo campo calculado 'AGRUP_MARITAL_STATUS' y agrupamos categorias 3 y 4:


Para generar campos calculados (equivalente al CASE WHEN de SQL en Python Pandas): crear una función y aplicarla sobre
el dataframe.

Creating a new column based on if-elif-else condition:
https://stackoverflow.com/questions/21702342/creating-a-new-column-based-on-if-elif-else-condition


"""



def f(row):
    if row['SEX_AND_MARITAL_STATUS'] == 4:
        val = 3
    elif row['SEX_AND_MARITAL_STATUS'] == 3:
        val = 3
    else:
        val = row['SEX_AND_MARITAL_STATUS']
    return val



credit4['AGRUP_MARITAL_STATUS'] = credit4.apply(f, axis=1)

credit4['AGRUP_MARITAL_STATUS'] 


credit4.columns





"""

Calculamos cantidad de registros que tiene el Dataset

"""

credit4.index

index = credit4.index

number_of_rows = len(index)

number_of_rows

"""
Contingency Table:
    
"""

AGRUP_MARITAL_STATUS_X_CREDIT= pd.crosstab(credit4['AGRUP_MARITAL_STATUS'] , credit4['CREDITABILITY'] )

AGRUP_MARITAL_STATUS_X_CREDIT

(AGRUP_MARITAL_STATUS_X_CREDIT/number_of_rows)*100

"""

GRAFICO CANTIDAD

"""

AGRUP_MARITAL_STATUS_X_CREDIT.plot.bar()

"""

GRAFICO CANTIDAD APILADO!!!!


"""


AGRUP_MARITAL_STATUS_X_CREDIT.plot.bar(stacked=True, figsize=(10, 10))

# Normalize to sum to 1

AGRUP_MARITAL_STATUS_X_CREDIT_NORM = AGRUP_MARITAL_STATUS_X_CREDIT.div(AGRUP_MARITAL_STATUS_X_CREDIT.sum(1), axis=0)

AGRUP_MARITAL_STATUS_X_CREDIT_NORM


AGRUP_MARITAL_STATUS_X_CREDIT_NORM.plot.bar(figsize=(10, 10))

"""

GRAFICO NORMALIZADO APILADO!!!!


"""

"""

La tasa de malos es distinta entre las distintas categorías de la variable original:
    
"""

AGRUP_MARITAL_STATUS_X_CREDIT_NORM.plot.bar(stacked=True, figsize=(10, 10))

AGRUP_MARITAL_STATUS_X_CREDIT_NORM

AGRUP_MARITAL_STATUS_X_CREDIT_NORM*100




CREDIT_X_AGRUP_MARITAL_STATUS= pd.crosstab(credit4['CREDITABILITY'],credit4['AGRUP_MARITAL_STATUS'])


CREDIT_X_AGRUP_MARITAL_STATUS


CREDIT_X_AGRUP_MARITAL_STATUS_NORM = CREDIT_X_AGRUP_MARITAL_STATUS.div(CREDIT_X_AGRUP_MARITAL_STATUS.sum(1), axis=0)


CREDIT_X_AGRUP_MARITAL_STATUS_NORM


CREDIT_X_AGRUP_MARITAL_STATUS_NORM.plot.bar(figsize=(10, 10))

"""

GRAFICO NORMALIZADO APILADO!!!!


"""

CREDIT_X_AGRUP_MARITAL_STATUS_NORM.plot.bar(stacked=True, figsize=(10, 10))

CREDIT_X_AGRUP_MARITAL_STATUS_NORM

CREDIT_X_AGRUP_MARITAL_STATUS_NORM*100




"""

El P Value de Chi Cuadrado es menor 0,05 = atributos son "dependientes".

"""
from scipy.stats import chi2_contingency
chi2_contingency(AGRUP_MARITAL_STATUS_X_CREDIT)

"""

El P Value de Chi Cuadrado es menor 0,05 = atributos son "dependientes".

"""
from scipy.stats import chi2_contingency
chi2_contingency(CREDIT_X_AGRUP_MARITAL_STATUS)





# endregion 10) 'SEX_AND_MARITAL_STATUS' / Marital status x CREDITABILITY



# region 11) 'GUARANTORS' / Guarantors x CREDITABILITY


credit4.columns

credit4

credit4['GUARANTORS'] 

credit4['CREDITABILITY'] 



"""

Calculamos cantidad de registros que tiene el Dataset

"""

credit4.index

index = credit4.index

number_of_rows = len(index)

number_of_rows

"""
Contingency Table:
    
"""

GUARANTORS_X_CREDIT= pd.crosstab(credit4['GUARANTORS'] , credit4['CREDITABILITY'] )

GUARANTORS_X_CREDIT

(GUARANTORS_X_CREDIT/number_of_rows)*100

"""

GRAFICO CANTIDAD

"""

GUARANTORS_X_CREDIT.plot.bar()

"""

GRAFICO CANTIDAD APILADO!!!!


"""


GUARANTORS_X_CREDIT.plot.bar(stacked=True)

# Normalize to sum to 1


GUARANTORS_X_CREDIT_NORM = GUARANTORS_X_CREDIT.div(GUARANTORS_X_CREDIT.sum(1), axis=0)


GUARANTORS_X_CREDIT_NORM


GUARANTORS_X_CREDIT_NORM.plot.bar()


"""

GRAFICO NORMALIZADO APILADO!!!!


"""


"""

 
Other debtors / guarantors (A10)
 
A101 : none
A102 : co-applicant
A103 : guarantor
 
---
 
Tuffery: la existencia de un garante tiende a reducir la tasa de malos.
 
Tuffery deja esta variable tal como viene dada en la muestra.
 


"""


GUARANTORS_X_CREDIT_NORM.plot.bar(stacked=True, figsize=(10, 10))

GUARANTORS_X_CREDIT_NORM

GUARANTORS_X_CREDIT_NORM*100




CREDIT_X_GUARANTORS= pd.crosstab(credit3['CREDITABILITY'],credit3['GUARANTORS'])


CREDIT_X_GUARANTORS_NORM = CREDIT_X_GUARANTORS.div(CREDIT_X_GUARANTORS.sum(1), axis=0)


CREDIT_X_GUARANTORS_NORM



"""

GRAFICO CANTIDAD


"""



CREDIT_X_GUARANTORS_NORM.plot.bar()


"""

GRAFICO NORMALIZADO APILADO!!!!


"""

CREDIT_X_GUARANTORS_NORM.plot.bar(stacked=True, figsize=(10, 10))

CREDIT_X_GUARANTORS_NORM

CREDIT_X_GUARANTORS_NORM*100





"""

El P Value de Chi Cuadrado es menor 0,05 = atributos son "dependientes".

"""
from scipy.stats import chi2_contingency
chi2_contingency(GUARANTORS_X_CREDIT)

"""

El P Value de Chi Cuadrado es menor 0,05 = atributos son "dependientes".

"""
from scipy.stats import chi2_contingency
chi2_contingency(CREDIT_X_GUARANTORS)




# endregion 11) 'GUARANTORS' / Guarantors x CREDITABILITY



# region 12) 'INSTALMENT_PER_CENT' / Affordability ratio x CREDITABILITY


credit4.columns

credit4

credit4['INSTALMENT_PER_CENT'] 

credit4['CREDITABILITY'] 



"""

Calculamos cantidad de registros que tiene el Dataset

"""

credit4.index

index = credit4.index

number_of_rows = len(index)

number_of_rows

"""
Contingency Table:
    
"""

INSTALMENT_X_CREDIT= pd.crosstab(credit4['INSTALMENT_PER_CENT'] , credit4['CREDITABILITY'] )

INSTALMENT_X_CREDIT

(INSTALMENT_X_CREDIT/number_of_rows)*100

"""

GRAFICO CANTIDAD

"""

INSTALMENT_X_CREDIT.plot.bar()

"""

GRAFICO CANTIDAD APILADO!!!!


"""


INSTALMENT_X_CREDIT.plot.bar(stacked=True)

# Normalize to sum to 1


INSTALMENT_X_CREDIT_NORM = INSTALMENT_X_CREDIT.div(INSTALMENT_X_CREDIT.sum(1), axis=0)


INSTALMENT_X_CREDIT_NORM


INSTALMENT_X_CREDIT_NORM.plot.bar()


"""

GRAFICO NORMALIZADO APILADO!!!!


"""


"""

 
Tuffery pagina 592: La variable Installment_rate /Affordability_Rate tiene poco 
poder predictivo: las tasas de malos son parecidas entre las distintas categorías 
de la variable. Por lo tanto, decide eliminarla para el modelo.


"""


INSTALMENT_X_CREDIT_NORM.plot.bar(stacked=True, figsize=(10, 10))

INSTALMENT_X_CREDIT_NORM

INSTALMENT_X_CREDIT_NORM*100




CREDIT_X_INSTALMENT= pd.crosstab(credit3['CREDITABILITY'],credit3['INSTALMENT_PER_CENT'])


CREDIT_X_INSTALMENT_NORM = CREDIT_X_INSTALMENT.div(CREDIT_X_INSTALMENT.sum(1), axis=0)


CREDIT_X_INSTALMENT_NORM



"""

GRAFICO CANTIDAD


"""



CREDIT_X_INSTALMENT_NORM.plot.bar()


"""

GRAFICO NORMALIZADO APILADO!!!!


"""

CREDIT_X_INSTALMENT_NORM.plot.bar(stacked=True, figsize=(10, 10))

CREDIT_X_INSTALMENT_NORM

CREDIT_X_INSTALMENT_NORM*100





"""

El P Value de Chi Cuadrado es MAYOR a 0.05 = atributos son "independientes"!!!!!!!!!!!

"""

from scipy.stats import chi2_contingency
chi2_contingency(INSTALMENT_X_CREDIT)

"""

El P Value de Chi Cuadrado es MAYOR a 0.05 = atributos son "independientes"!!!!!!!!!!!

"""

from scipy.stats import chi2_contingency
chi2_contingency(CREDIT_X_INSTALMENT)




# endregion 12) 'INSTALMENT_PER_CENT' / Affordability ratio x CREDITABILITY




# region Dataset resultante de la selección de variables explicativas reagrupadas y descartadas

"""
A partir de los criterios y análisis para reagrupar  Variables Explicativas del Modelo
previamente desarrollados, seleccionamos las variables con las que continuaremos trabajando, generando 
un nuevo dataset/dataframe llamado credit5:

"""


credit4.columns


selected_columns = credit4[['CREDITABILITY', 
                           'ACCOUNT_BALANCE',
                           'cred_duration_bin_optimal',
                           'age_bin_optimal',
                           'GUARANTORS',

                           'AGRUP_REPAYMENT_HIST', 
                           'AGRUP_SAVINGS', 
                           'AGRUP_PURPOSE', 
                           'AGRUP_ASSET',
                           'AGRUP_LENGTH_EMPLOY', 
                           'AGRUP_TYPE_APART', 
                           'AGRUP_OTHER_CREDITS',
                           'AGRUP_MARITAL_STATUS'    
                           ]]


credit5= selected_columns.copy()


# endregion Dataset resultante de la selección de variables explicativas reagrupadas y descartadas





# endregion CUÁNDO CONVIENTE O NO REGRUPAR UNA VARIABLE EXPLICATIVA CATEGÓRICA




# region Medición del Cambio en el valor de Cramer V de las Variables Explicativas


"""

Notable: las nuevas variables explicativas (continuas binneadas y categóricas reagrupadas) no necesariamente aumentan los valores de V-Cramer. 
De hecho en este ejercicio se observa que varias disminuyen o mantienen su V-Cramer.
 
Medimos nuevamente cuánto variaron los valores de Cramer V con respecto a los valores obtenidos cuando hicimos la primera selección de variables explicativas.
 
En algunos casos, el valor de V-Cramer disminuyó, es decir que disminuyó la fuerza asociación entre dos variables categóricas (Variable Explicativa Vs.credibility).
En la gran mayoría sin embargo (según Tuffery pagina 593) el valor de V-Cramer se mantuvo o la variación es imperceptible.
A pesar de la disminución de la fuerza de asociación entre algunas Variable Explicativa Vs. Credibility, Tuffery decide utilizar las 
nuevas variables continuas binneadas y categóricas reagrupadas.
Tuffery (pagina 593) utiliza 12 principales variables explicativas con mayor fuerza de asociación a la variable Target Credibility

"""


import scipy.stats as ss

def cramers_V(var1, var2):
    confusion_matrix = pd.crosstab(var1,var2)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))




rows= []
for var1 in credit5:
  col = []
  for var2 in credit5 :
    cramers =cramers_V(credit5[var1], credit5[var2]) # Cramer's V test
    col.append(round(cramers,2)) # Keeping of the rounded value of the Cramer's V  
  rows.append(col)
  
cramers_results = np.array(rows)
df = pd.DataFrame(cramers_results, columns = credit5.columns, index =credit5.columns)


df

df['CREDITABILITY']


Serie_Cred2=df['CREDITABILITY']


"""

Nuevo Ranking de Variables explicativas de 'CREDITABILITY'

"""


Rank_Var_Explicativas3=Serie_Cred2.sort_values(ascending = False) 


Rank_Var_Explicativas3


"""

Ranking Anterior de Variables explicativas de 'CREDITABILITY'

"""

Rank_Var_Explicativas2


import matplotlib.pyplot as plt



fig, axes = plt.subplots(1, 1)
Rank_Var_Explicativas3.plot.bar(color='k', alpha=0.7, figsize=(15, 15))


Rank_Var_Explicativas3=Serie_Cred2.sort_values(ascending = True)

fig, axes = plt.subplots(1, 1)
Rank_Var_Explicativas3.plot.barh(color='k', alpha=0.7, figsize=(15, 15))

"""

Comparamos gráficamente el cambio en el valor de Cramer V de las Variables Explicativas:

"""


fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
Rank_Var_Explicativas3.plot.barh(color='k', alpha=0.7)
ax2 = fig.add_subplot(2, 2, 2)
Rank_Var_Explicativas2.plot.barh(color='k', alpha=0.7)


"""

Variación en los valores de V-Cramer entre las variables originales (para la SELECCION DE VARIABLES EXPLICATIVAS DEL MODELO)
Vs. las nuevas variables explicativas continuas binneadas y categóricas reagrupadas.
 
 
En algunos casos, el valor de V-Cramer disminuyó, es decir que disminuyó la fuerza
asociación entre dos variables categóricas (Variable Explicativa Vs.credibility).
En la gran mayoría sin embargo (según Tuffery pagina 593) el valor de V-Cramer
se mantuvo o la variación es imperceptible.
 
A pesar de la disminución de la fuerza de asociación entre algunas Variable Explicativa Vs. Credibility, Tuffery decide utilizar las 
nuevas variables continuas binneadas y categóricas reagrupadas.
 
Tuffery (pagina 593) utiliza 12 principales variables explicativas con mayor fuerza de asociación a la variable Target Credibility


"""

# endregion Medición del Cambio en el valor de Cramer V de las Variables Explicativas




# region GENERAR VARIABLES DUMMY


"""

Generar Variables Dummy a partir de variables categóricas

https://towardsdatascience.com/what-is-one-hot-encoding-and-how-to-use-pandas-get-dummies-function-922eb9bd4970


"""

credit5

credit5.columns

"""

Generar variables Dummy para una variable categórica específica del Dataset:

"""

"""

Cantidad de categorías

"""

credit5['ACCOUNT_BALANCE'].value_counts()

DUMMY_ACCOUNT_BALANCE=pd.get_dummies(credit5['ACCOUNT_BALANCE'], prefix='ACCOUNT_BALANCE')

DUMMY_ACCOUNT_BALANCE

DUMMY_ACCOUNT_BALANCE.columns


"""

Attaching to the DataFrame:

"""


credit6 = pd.concat([credit5['ACCOUNT_BALANCE'], DUMMY_ACCOUNT_BALANCE], axis=1)

credit6.head()



"""

Generar variables Dummy para todo el Dataset:

"""

credit5

credit5.columns

credit7=pd.get_dummies(credit5, columns=['ACCOUNT_BALANCE', 'cred_duration_bin_optimal',
       'age_bin_optimal', 'GUARANTORS', 'AGRUP_REPAYMENT_HIST',
       'AGRUP_SAVINGS', 'AGRUP_PURPOSE', 'AGRUP_ASSET', 'AGRUP_LENGTH_EMPLOY',
       'AGRUP_TYPE_APART', 'AGRUP_OTHER_CREDITS', 'AGRUP_MARITAL_STATUS'])

credit7

credit7.columns



# endregion GENERAR VARIABLES DUMMY



# region Logistic Regression in Python


"""

Paquetes StatsModels Vs. scikit-learn:

El paquete scikit-learn por ahora no ofrece muchos indicadores estadísticos 
relevantes como P-Value, desvío estandar, intervalos de confianza de parametros,
etc.

El paquete StatsModels sí ofrece más indicadores estadísticos.

"""


# region Logistic Regression in Python con StatsModels

"""

******************************************************************************
******************************************************************************

Logistic Regression in Python con StatsModels:

******************************************************************************
******************************************************************************

Fuentes:
    
https://datatofish.com/logistic-regression-python/

https://realpython.com/logistic-regression-python/


******************************************************************************
******************************************************************************

Selección de variables Dummy y de categorías base:
**************************************************
    
Tener en cuenta que SPSS Modeler selecciona como "categoría base" aquella 
categoría que es representada con el mayor valor númerico. 

Ejemplo: si la variable "Partido Politico" tiene 3 categorías: 
 "Democratas", "Republicanos" y "Verdes" y cada una de ellas es 
 representada respectivamente con los valores respectivos de 1, 2 y 3, 
 entonces la categoría "Verdes" será la "categoría base"

 
Esto se observa de la salida del nodo de regresión logistica en la 
parte "Codificación de los parámetros".

******************************************************************************
******************************************************************************

Para coincidir en resultados con las estimaciones de SPSS Modeler, descartamos 
las variables dummys correspondientes a las categorías con el mayor valor númerico.

Ejemplo: descartamos 'ACCOUNT_BALANCE_4', 'cred_duration_bin_optimal_(36, 100]','GUARANTORS_3',etc.

Set the independent variables (represented as X) and the dependent variable (represented as y):

"""

credit7.columns

X = credit7[[
        'ACCOUNT_BALANCE_1', 
        'ACCOUNT_BALANCE_2',
        'ACCOUNT_BALANCE_3',
        
        'cred_duration_bin_optimal_(0, 15]',
        'cred_duration_bin_optimal_(15, 36]',
       
        'age_bin_optimal_(0, 25]',
    
       'GUARANTORS_1', 
       'GUARANTORS_2',
       
       'AGRUP_REPAYMENT_HIST_0', 
       'AGRUP_REPAYMENT_HIST_1',
       'AGRUP_REPAYMENT_HIST_2', 
       
       'AGRUP_SAVINGS_1',
       'AGRUP_SAVINGS_3', 
       
       
       'AGRUP_PURPOSE_0',
       'AGRUP_PURPOSE_1', 
       'AGRUP_PURPOSE_2', 
       'AGRUP_PURPOSE_3',
       'AGRUP_PURPOSE_8', 
       
       
       'AGRUP_ASSET_1', 
       'AGRUP_ASSET_2', 
       
       
       'AGRUP_LENGTH_EMPLOY_1', 
       'AGRUP_LENGTH_EMPLOY_3',
       'AGRUP_LENGTH_EMPLOY_4', 
       
       
       'AGRUP_TYPE_APART_1',
       
       
       'AGRUP_OTHER_CREDITS_1', 
       
       
       'AGRUP_MARITAL_STATUS_1', 
       'AGRUP_MARITAL_STATUS_2',
       ]]
    
y = credit7['CREDITABILITY']


"""
Para paquete StatsModels:
Se agrega una columna con todos  los valores igual a 1 y así agregar 
intercepto en el modelo.

You can get the inputs and output the same way as you did 
with scikit-learn. However, StatsModels doesn’t take the 
intercept 𝑏₀ into account, and you need to include the 
additional column of ones in x. You do that with add_constant():
    
"""

import statsmodels.api as sm

X = sm.add_constant(X)


X



"""

Logistic Regression in Python With StatsModels



"""



model = sm.Logit(y, X)





"""

Now, you’ve created your model and you should fit it with the 
existing data. You do that with .fit() or, if you want to apply
L1 regularization, with .fit_regularized():

"""

LogitResults = model.fit(method='newton')


LogitResults.summary()

LogitResults.summary2()


"""

NOTAR: son los mismos resultados que obtuve con el programa de SPSS Modeler. Ver archivo Woord: "Score_Model"

C:\Users\rodri\Google Drive\Aplicaciones_herramientas_Libros\Python\Credit_Score_Python


"""


# endregion Logistic Regression in Python con StatsModels



#region Regresión Logística utilizando scikit-learn

"""

**************************************************************
**************************************************************

Regresión Logística utilizando scikit-learn

https://realpython.com/logistic-regression-python/

**************************************************************
**************************************************************

"""



credit7.columns

X = credit7[[
        'ACCOUNT_BALANCE_1', 
        'ACCOUNT_BALANCE_2',
        'ACCOUNT_BALANCE_3',
        
        'cred_duration_bin_optimal_(0, 15]',
        'cred_duration_bin_optimal_(15, 36]',
       
        'age_bin_optimal_(0, 25]',
    
       'GUARANTORS_1', 
       'GUARANTORS_2',
       
       'AGRUP_REPAYMENT_HIST_0', 
       'AGRUP_REPAYMENT_HIST_1',
       'AGRUP_REPAYMENT_HIST_2', 
       
       'AGRUP_SAVINGS_1',
       'AGRUP_SAVINGS_3', 
       
       
       'AGRUP_PURPOSE_0',
       'AGRUP_PURPOSE_1', 
       'AGRUP_PURPOSE_2', 
       'AGRUP_PURPOSE_3',
       'AGRUP_PURPOSE_8', 
       
       'AGRUP_ASSET_1', 
       'AGRUP_ASSET_2', 
       
       
       'AGRUP_LENGTH_EMPLOY_1', 
       'AGRUP_LENGTH_EMPLOY_3',
       'AGRUP_LENGTH_EMPLOY_4', 
       
       
       'AGRUP_TYPE_APART_1',
       
       
       'AGRUP_OTHER_CREDITS_1', 
       
       
       'AGRUP_MARITAL_STATUS_1', 
       'AGRUP_MARITAL_STATUS_2',
       ]]
    
y = credit7['CREDITABILITY']

"""

Then, apply train_test_split. For example, you can set the test size to 0.25, and 
therefore the model testing will be based on 25% of the dataset, while the model 
training will be based on 75% of the dataset:

"""



# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)



"""
Fuentes:
    
Libro: Practical Statistics for Data Scientists 50+ Essential Concepts Using R and Python
pagina 211



https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a

https://datatofish.com/logistic-regression-python/

"""


from sklearn.linear_model import LogisticRegression


logistic_regression = LogisticRegression(penalty='l2', C=1e42, solver='liblinear')


logistic_regression.fit(X,y)


logistic_regression.intercept_
logistic_regression.coef_


"""

NOTAR: son los mismos resultados que obtuve con el programa de SPSS Modeler. Ver archivo Woord: "Score_Model"

C:\Users\rodri\Google Drive\Aplicaciones_herramientas_Libros\Python\Credit_Score_Python


"""


#endregion Regresión Logística utilizando scikit-learn



# region Odds Ratios (OR)


"""


Odds Vs. Odds Ratios: 
*********************
*********************

Para mayor detalle y aclaraciones ver mi resumen de Regresión Logística escrito en Word en la siguiente ubicación:


C:\Users\rodri\Google Drive\Aplicaciones_herramientas_Libros\Estadistica\Resumenes mios\Regresion Logistica


"Odds" de una Variable Dependiente Y: 
*************************************

Los “Odds” de una variable dependiente Y, es la razón de Probabilidades en favor que el suceso (Y=1) tenga lugar 
sobre la probabilidad que el suceso no tenga lugar (Y=0):

Odds(Y=1)=Prob(Y=1)/Prob(Y=0)

En otras palabras, Odds es la cantidad de veces que un determinado evento es más probable que ocurra antes que no ocurra. 

Ejemplo:  Si las probabilidades que gane el partido Peronista son 0,75 y las probabilidades que no gane son 0,25, entonces:
Los Odds que gané el partido peronista son 3 (0,75/0,25=3). Es decir que es 3 veces más probable que gane el partido Peronista antes que no gane.


"Odds Ratio" de una Variable Independiente X: 
**********************************************

Los “Odds Ratio” de una variable independiente X mide cuánto varía el ratio de probabilidad de ocurrencia de Y=1 cuando 
la variable X cambia de 0 a 1.

Odds Ratios de X = Odds(Y=1|X=1) / Odds(Y=1|X=0) = [Prob(Y=1|X=1)⁄Prob(Y=0|X=1)] / [Prob(Y=1|X=0)⁄Prob(Y=0|X=0)] = e^(βi)

De manera equivalente, podemos decir el “Odds Ratios” de una variable independiente X mide cuánto varía el Odds de Y=1 cuando la variable X cambia de 0 a 1. 


Ejemplos:

1) Cuando X es variable categórica:
***********************************
Si investigamos la ocurrencia que una persona enferme (Y=1), y el Odds Ratios de la 
variable “Sexo” es igual a 1,5 (donde X=1=Hombre y X=0=Mujer), quiere decir que el ratio 
de probabilidad de enfermar de una persona es 1,5 veces mayor para un hombre que para una 
mujer. De manera equivalente podemos decir, el Odds de enfermar (Y=1) de una persona es 1,5 
veces mayor para un hombre que para una mujer.



Cuando X es variable continua: 
******************************

Si investigamos la ocurrencia que una persona enferme (Y=1), y el Odds Ratios de la 
variable “Edad” es igual a 1,11, quiere decir que el ratio de probabilidad de enfermar 
de una persona es 1,1 veces mayor cada año de edad que avanza. De manera equivalente 
podemos decir, el Odds de enfermar (Y=1) de una persona es 1,1 veces mayor cada año de edad que avanza.



Importancia de Discretizar Variables Continuas:
***********************************************
Stéphane Tufféry en el libro “DATA MINING AND STATISTICS FOR DECISION MAKING” página 444 menciona que el valor 
del Odds Ratio de una variable Continua no es del todo realista. Ejemplo: el proceso de enfermar no es necesariamente 
el mismo cada año que la edad avanza. Por lo tanto el Odds Ratio de una variable continua trae como consecuencia 2 
dificultades: 

A) la falla de no capturar o tener en cuenta los efectos no lineales entre la variable X e Y,  
B) La carencia de robustez de los modelos. 

Por lo tanto, Stéphane Tufféry recomienda que es mejor dividir una variable continua en clases, es decir discretizar 
la variable continua para capturar los efectos no lineales entre la variable X e Y.


De esta manera resolvemos todos los supuestos que se requieren para aplicar la regresión logística:

https://pythonfordatascienceorg.wordpress.com/logistic-regression-python/


Assumptions for logistic regression models:
*******************************************

-The dependent variable is categorical (binary)

-Independence of observations

-Independent variables are linearly related to the log odds

-Absence of multicollinearity: con el descate de Cramer V previamente hecho.

-Lack of outliers: gracias a la discretizacíon de las variables continuas.

Sumado a esto, recordar que la discretización de variables continuas tiene las siguientes Ventajas:

A) capturar o tiene en cuenta los efectos no lineales entre la variable X e Y.
 
B) Agrega robustez de los modelos.

C) Discretizar variables continuas y generar un dataset de sólo variables explicativas, permite 
seleccionar, comparar y rankear con un sólo indicador las mejores variables explicativas del modelo: con el valor de Cramer-V

D) Discretizar variables continuas y generar un dataset de sólo variables explicativas, permite 
identificar dependencias problemáticas entre variables explicativas (multicolinealidad).




La importancia de los “Odds Ratios”:
************************************

La importancia de los “Odds Ratios” es la interpretación del modelo: los “Odds Ratios” representanta 
el efecto constante de una variable predictora X en la probabilidad en que un suceso Y=1 ocurra.

La palabra clave aquí es “efecto constante”. En los modelos de regresión lineal, a menudo queremos 
una medición del efecto único de cada variable X sobre Y. Si tratamos de expresar el efecto de X en 
la probabilidad de que la variable categórica Y tenga un determinado valor (ejemplo: Y=1), el efecto no es constante.

Es decir, no hay manera de expresar en un número cómo X afecta a Y en términos de probabilidad. El efecto de X en 
la probabilidad de Y tiene diferentes valores dependiendo del valor que adquiera X. Ejemplo: en un modelo de 
regresión logística no es posible decir que el aumento unitario de la variable X aumentará (o disminuirá) en 
un determinado porcentaje constante la probabilidad que el evento Y=1 ocurra.

Por lo tanto, si se necesita comunicar el efecto de una variable X a una comunidad científica, es necesario 
manejar el concepto de “Odds Ratios”.

Stéphane Tufféry  en su Libro “DATA MINING AND STATISTICS FOR DECISION MAKING” señala que una de las ventajas de 
los modelos regresión logística (frente a otros tipos de modelos) es “The coefficients of the logit are easily interpreted in terms of odds ratios”. Ver página 478.



Significado o Interpretación del indicador de “Odds Ratios” de una variable Independiente X:
********************************************************************************************

Odds Ratios de X>1: 
*******************

Hay una relación o asociación positiva, el aumento de una unidad de la variable X aumenta el 
Odds de que un evento ocurra (Y=1).  Es decir, indica un incremento en la probabilidad que un evento ocurra (Y=1) 
en lugar que no ocurra, cuando X aumenta.
Ejemplo: el sexo de las personas aumenta el Odds de que una persona enferme, ceteris paribus las demás variables.


Odds Ratios de X<1: 
*******************

Hay una relación o asociación negativa, el aumento de una unidad de la variable X disminuye el Odds de que un evento 
ocurra (Y=1). Es decir, indica una disminución en la probabilidad que un evento ocurra (Y=1) en lugar que no ocurra, cuando X aumenta.
Ejemplo: el sexo de las personas disminuye el Odds de que una persona enferme, ceteris paribus las demás variables.


Odds Ratios de X=1: 
*******************

El Odds de que un evento ocurra (Y=1) se mantiene igual o fijo para cada valor que adquiere X. Es decir,  que 
indica la probabilidad que un evento ocurra (Y=1) y que no ocurra (Y=0) son iguales para cada valor que adquiere X.
Ejemplo: el Odds de que una persona enferme es el mismo para cada uno de los sexos.



"""





"""

Odds Ratios con StatsModels:
****************************

GETTING THE ODDS RATIOS, Z-VALUE, AND 95% CI

https://pythonfordatascienceorg.wordpress.com/logistic-regression-python/

"""

LogitResults.summary()

LogitResults.summary2()

LogitResults.params

LogitResults.pvalues

LogitResults.conf_int()


model_odds = pd.DataFrame(np.exp(LogitResults.params), columns= ['OR'])
model_odds['z-value']= LogitResults.pvalues
model_odds[['2.5%', '97.5%']] = np.exp(LogitResults.conf_int())
model_odds


"""

Odds Ratios con StatsModels:
****************************

https://www.datasklr.com/logistic-regression/logistic-regression-with-binary-target

"""

import numpy as np
np.exp(logistic_regression.intercept_)
np.exp(logistic_regression.coef_)


"""

NOTAR: son los mismos resultados de Odds Ratios que obtuve con el programa de SPSS Modeler. Ver archivo Woord: "Score_Model"

C:\Users\rodri\Google Drive\Aplicaciones_herramientas_Libros\Python\Credit_Score_Python


"""


# endregion Odds Ratios (OR)



#region Partición del Dataset


credit7.columns

X = credit7[[
        'ACCOUNT_BALANCE_1', 
        'ACCOUNT_BALANCE_2',
        'ACCOUNT_BALANCE_3',
        
        'cred_duration_bin_optimal_(0, 15]',
        'cred_duration_bin_optimal_(15, 36]',
       
        'age_bin_optimal_(0, 25]',
    
       'GUARANTORS_1', 
       'GUARANTORS_2',
       
       'AGRUP_REPAYMENT_HIST_0', 
       'AGRUP_REPAYMENT_HIST_1',
       'AGRUP_REPAYMENT_HIST_2', 
       
       'AGRUP_SAVINGS_1',
       'AGRUP_SAVINGS_3', 
       
       
       'AGRUP_PURPOSE_0',
       'AGRUP_PURPOSE_1', 
       'AGRUP_PURPOSE_2', 
       'AGRUP_PURPOSE_3',
       'AGRUP_PURPOSE_8', 
       
       'AGRUP_ASSET_1', 
       'AGRUP_ASSET_2', 
       
       
       'AGRUP_LENGTH_EMPLOY_1', 
       'AGRUP_LENGTH_EMPLOY_3',
       'AGRUP_LENGTH_EMPLOY_4', 
       
       
       'AGRUP_TYPE_APART_1',
       
       
       'AGRUP_OTHER_CREDITS_1', 
       
       
       'AGRUP_MARITAL_STATUS_1', 
       'AGRUP_MARITAL_STATUS_2',
       ]]
    
y = credit7['CREDITABILITY']



"""

With train_test_split(), you need to provide the sequences that you want to split as well as any optional arguments.

Estructura basica:

sklearn.model_selection.train_test_split(*arrays, **options) -> list


arrays: is the sequence of lists, NumPy arrays or pandas DataFrames.

options: are the optional keyword arguments that you can use to get desired behavior:


-train_size: is the number that defines the size of the training set. If you provide a float, then it must be between 0.0 and 1.0 and will 
define the share of the dataset used for testing. If you provide an int, then it will represent the total number of the training samples. 
The default value is None.

-test_size: is the number that defines the size of the test set. It’s very similar to train_size. You should provide either train_size or 
test_size. If neither is given, then the default share of the dataset that will be used for testing is 0.25, or 25 percent.

-random_state: Para generar siempre los mismos datasets de entrenamiento y testeo y obtener los mismos resultados.Is the object that controls randomization 
during splitting. It can be either an int or an instance of RandomState. The default value is None. Ejemplo si se explicita el parametro: random_state=4 o random_state=123 
siempre se genererán las mismas muestrar, independientemente que corramos una y otra vez la función train_test_split. Si no se explicita este parámetro, cada vez que volvamos a
correr la función train_test_split se generarán muestras distintas.

-shuffle: is the Boolean object (True by default) that determines whether to shuffle the dataset before applying the split.

-stratify: Para generar muestra de testeo estratificada, donde se respeten las proporciones de casos de una variable determinada, ejemplo de la variable y: stratify=y


https://realpython.com/train-test-split-python-data/



Then, apply train_test_split. For example, you can set the test size to 0.25, and therefore the 
model testing will be based on 25% of the dataset, while the model training will be based on 75% of the dataset:

https://datatofish.com/logistic-regression-python/


"""

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)


"""

Para generar siempre los mismos datasets de entrenamiento y testeo: random_state

Sometimes, to make your tests reproducible, you need a random split with the same output for each function call. 
You can do that with the parameter random_state. The value of random_state isn’t important, it can be any non-negative integer.

Para generar muestra de testeo estratificada, donde se respeten las proporciones de casos de una variable determinada, ejemplo de la variable y: stratify=y

If you want to (approximately) keep the proportion of y values through the training and test sets, then pass stratify=y

Ejemplo:

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4, stratify=y)

"""



"""

Apply the logistic regression as follows:

"""



logistic_regression = LogisticRegression(penalty='l2', C=1e42, solver='liblinear')


logistic_regression.fit(X_train,y_train)


logistic_regression.intercept_
logistic_regression.coef_


X.count()
y.count()

X_train.count()
X_test.count()

y_train.count()
y_test.count()


"""

Cantidad de registros y columnas de un dataframe:

"""

X.shape
X_train.shape
X_test.shape

y.shape.shape
y_train.shape
y_test.shape



"""

Logistic Regression in Python With StatsModels


"""


import statsmodels.api as sm

X_train = sm.add_constant(X_train)

model = sm.Logit(y_train, X_train)

LogitResults = model.fit(method='newton')

LogitResults.summary()

LogitResults.summary2()



#endregion Partición del Dataset



# region Tabla de Contingencia / Confusion Matrix



# region Confusion Matrix con el modelo de statsmodels






"""

Generar  Tabla de Contingencia / Confusion Matrix a partir del modelo de Statsmodels

https://realpython.com/logistic-regression-python/#logistic-regression-in-python-with-statsmodels-example
https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
https://datatofish.com/logistic-regression-python/

StatsModels:
    
Aplicamos el modelo estimado con el conjunto de datos de entrenamiento sobre el conjunto de datos de testeo.

Para poder aplicar el modelo de StatsModels es necesario que el conjunto de entrenamiento tenga la misma cantidad
de columnanas que el conjunto de testeo.
Hay que agregar al conjunto de testeo una columna con todos los 
valores igual a 1, al igual que lo hicimos con el conjunto de entrenamiento. De lo contrario aparece error por no
tener el conjunto de testeo la  misma dimensión que el conjunto de entrenamiento.


"""


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

import statsmodels.api as sm

X_train = sm.add_constant(X_train)

model = sm.Logit(y_train, X_train)

Stat_LogitResults = model.fit(method='newton')

Stat_LogitResults.summary()





X_train.shape

X_test.shape
 
X_test = sm.add_constant(X_test)



"""

Obtener las probabilidades estimadas sobre el conjunto de Testeo:

"""

y_Predicted_Test = Stat_LogitResults.predict(X_test)
print(y_Predicted_Test)



"""

Obtener los valores predecidos sobre el conjunto de Testeo segun punto de corte: 0,5:

"""

y_Predicted_Test=(Stat_LogitResults.predict(X_test) >= 0.5).astype(int)



"""

Tabla de Confusión sobre el conjunto de Entrenamiento, según el punto de corte que se establezca:

"""


Stat_LogitResults.pred_table(threshold=0.5)


"""

Otra manera de obtener la Tabla de Confusión con sklearn sobre el conjunto de Entrenamiento, 
según el punto de corte que se establezca:


"""

y_Predicted_Train=(Stat_LogitResults.predict(X_train) >= 0.5).astype(int)

from sklearn.metrics import confusion_matrix
cm_train = confusion_matrix(y_train, y_Predicted_Train)

cm_train


"""

Tabla de Confusión sobre el conjunto de Testeo: contrastamos valores predecidos 
sobre el conjunto de testeo Vs. valores reales del conjunto de testeo:

"""


from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_test, y_Predicted_Test)

cm_test


"""

Distintas maneras de visualizar tabla de Confusion:

"""


"""

Ejemplo 1: con matplotlib

"""

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm_test)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm_test[i, j], ha='center', va='center', color='red')
plt.show()




"""

Ejemplo 2: seaborn heatmap
    
"""

import seaborn as sns
sns.heatmap(cm_test, annot=True, fmt='g')

# fix for mpl bug that cuts off top/bottom of seaborn viz
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values
plt.show() # ta-da!




"""

Ejemplo 3: seaborn heatmap
    
"""


sns.heatmap(cm_test/np.sum(cm_test), annot=True, 
            fmt='.2%', cmap='Blues')
# fix for mpl bug that cuts off top/bottom of seaborn viz
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values
plt.show() # ta-da!




"""

Ejemplo 4: seaborn heatmap
    
"""



labels = ['True Neg','False Pos','False Neg','True Pos']
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cm_test, annot=labels, fmt='', cmap='Blues')

group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                cm_test.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cm_test.flatten()/np.sum(cm_test)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cm_test, annot=labels, fmt='', cmap='Blues')
# fix for mpl bug that cuts off top/bottom of seaborn viz
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values
plt.show() # ta-da!






# endregion Confusion Matrix con el modelo de statsmodels




# region Confusion Matrix con el modelo de scikit-learn



"""

Generar  Tabla de Contingencia / Confusion Matrix a partir del modelo de scikit-learn

https://realpython.com/logistic-regression-python/#logistic-regression-in-python-with-statsmodels-example
https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
https://datatofish.com/logistic-regression-python/


scikit-learn:

Aplicamos el modelo estimado con el conjunto de datos de entrenamiento sobre el conjunto de datos de testeo.


A diferencia de StatsModels, con el modelo de scikit-learn no es necesario agregar al conjunto de testeo ni de entrenamienta 
una columna con todos los valores igual a 1. Esto se debe a que scikit-learn genera modelo con intercepto automáticamente, sin necesidad
de explicitarlo como en StatsModels.


"""


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)


scikit_log_reg = LogisticRegression(penalty='l2', C=1e42, solver='liblinear')
scikit_log_reg.fit(X_train,y_train)


scikit_log_reg.intercept_
scikit_log_reg.coef_



X_train.shape

X_test.shape
 


"""

En scikit-learn, el punto de corte de la funcion .predict() por default es 0.5 (default threshold=0.5).

"""

y_pred_train = scikit_log_reg.predict(X_train)


y_pred_train




#Get the confusion matrix

from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(y_train, y_pred_train)
print(cf_matrix)


"""

Obtener las probabilidades estimadas sobre el conjunto de Testeo:

Each row corresponds to a single observation. The first column is 
the probability of the predicted output being zero, that is 1 - 𝑝(𝑥). 
The second column is the probability that the output is one, or 𝑝(𝑥). 

https://realpython.com/logistic-regression-python/#logistic-regression-python-packages   
    

"""



y_Predicted_Proba_test = scikit_log_reg.predict_proba(X_test)

print(y_Predicted_Proba_test)


"""

Obtener los valores predecidos sobre el conjunto de Testeo segun punto de corte: 

"""

threshold = 0.5


y_Predicted_test = (y_Predicted_Proba_test [:,1] >= threshold).astype('int')

y_Predicted_test


"""

Tabla de Confusión sobre el conjunto de Testeo: contrastamos valores predecidos 
sobre el conjunto de testeo Vs. valores reales del conjunto de testeo:

"""


from sklearn.metrics import confusion_matrix
scikit_cm_test = confusion_matrix(y_test, y_Predicted_test)

scikit_cm_test




"""

Obtener la Tabla de Confusión con sklearn sobre el conjunto de Entrenamiento, 
según el punto de corte que se establezca:
    
Ejemplo: threshold = 0.1


"""

y_Predicted_Proba_train = scikit_log_reg.predict_proba(X_train)

y_Predicted_Proba_train



threshold = 0.1


y_Predicted_train = (y_Predicted_Proba_train [:,1] >= threshold).astype('int')

y_Predicted_train


from sklearn.metrics import confusion_matrix
cm_train = confusion_matrix(y_train, y_Predicted_train)

cm_train





"""

Distintas maneras de visualizar tabla de Confusion:

"""


"""

Ejemplo 1: con matplotlib

"""

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(scikit_cm_test)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, scikit_cm_test[i, j], ha='center', va='center', color='red')
plt.show()




"""

Ejemplo 2: seaborn heatmap
    
"""

import seaborn as sns
sns.heatmap(scikit_cm_test, annot=True, fmt='g')

# fix for mpl bug that cuts off top/bottom of seaborn viz
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values
plt.show() # ta-da!




"""

Ejemplo 3: seaborn heatmap
    
"""


sns.heatmap(scikit_cm_test/np.sum(scikit_cm_test), annot=True, 
            fmt='.2%', cmap='Blues')
# fix for mpl bug that cuts off top/bottom of seaborn viz
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values
plt.show() # ta-da!




"""

Ejemplo 4: seaborn heatmap
    
"""



labels = ['True Neg','False Pos','False Neg','True Pos']
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(scikit_cm_test, annot=labels, fmt='', cmap='Blues')

group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                scikit_cm_test.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     scikit_cm_test.flatten()/np.sum(scikit_cm_test)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(scikit_cm_test, annot=labels, fmt='', cmap='Blues')
# fix for mpl bug that cuts off top/bottom of seaborn viz
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values
plt.show() # ta-da!






# endregion Confusion Matrix con el modelo de scikit-learn





# endregion Tabla de Contingencia / Confusion Matrix




# region Precision, recall, fscore, support, Specificity




from sklearn.metrics import confusion_matrix
scikit_cm_test = confusion_matrix(y_test, y_Predicted_test)

"""

Graficamos la tabla de Contigencia:

"""

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(scikit_cm_test)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, scikit_cm_test[i, j], ha='center', va='center', color='red')
plt.show()




"""
Precisión, recall, fscore, support:  de Y=0
"""
print('Precision', scikit_cm_test[0, 0] / sum(scikit_cm_test[:, 0]))
print('Recall', scikit_cm_test[0, 0] / sum(scikit_cm_test[0, :]))
print('Specificity', scikit_cm_test[1, 1] / sum(scikit_cm_test[1, :]))


"""
Precisión, recall, fscore, support:  de Y=1
"""
print('Precision', scikit_cm_test[1, 1] / sum(scikit_cm_test[:, 1]))
print('Recall', scikit_cm_test[1, 1] / sum(scikit_cm_test[1, :]))
print('Specificity', scikit_cm_test[0, 0] / sum(scikit_cm_test[0, :]))





"""

Otra manera de calcular: precision_recall_fscore_support

precision
recall
fscore
support


"""

from sklearn.metrics import precision_recall_fscore_support

"""
precision, recall, fscore, support:  de Y=0 
"""

precision,recall,fscore,support = precision_recall_fscore_support(y_test, y_Predicted_test,labels=[0])


precision

recall

fscore

support

"""
precision, recall, fscore, support:  de Y=0 
"""

precision,recall,fscore,support = precision_recall_fscore_support(y_test, y_Predicted_test,labels=[1])


precision

recall

fscore

support


"""
precision, recall, fscore, support:  de Y=1 y Y=0 en simultaneo
"""

precision,recall,fscore,support = precision_recall_fscore_support(y_test, y_Predicted_test,labels=[0,1])


precision

recall

fscore

support





"""

https://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/modules/generated/sklearn.metrics.precision_recall_fscore_support.html

macro:
Average over classes (does not take imbalance into account).

micro:
Average over instances (takes imbalance into account). This implies that precision == recall == f1

weighted:
Average weighted by support (takes imbalance into account). 
Can result in f1 score that is not between precision and recall.

"""

print(precision_recall_fscore_support(y_test, y_Predicted_test, average='macro'))
print(precision_recall_fscore_support(y_test, y_Predicted_test, average='micro'))
print(precision_recall_fscore_support(y_test, y_Predicted_test, average='weighted'))






# endregion Precision, recall, fscore, support, Specificity




# region Curva de ROC y AUC


"""

https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python

"""

"""

Curva de ROC aplicada al conjunto de datos de Entrenamiento:
    
OJO: con los parametros [:,0] y [:,1] elegimos distintos tipos de probabilidades.

Recordar cuando veíamos Tabla de Contingencia:
    

Each row corresponds to a single observation. The first column is 
the probability of the predicted output being zero, that is 1 - 𝑝(𝑥). 
The second column is the probability that the output is one, or 𝑝(𝑥). 

https://realpython.com/logistic-regression-python/#logistic-regression-python-packages   
    

y_Predicted_Proba_test = scikit_log_reg.predict_proba(X_test)

Elegimos las probabilidades de la primera columna:
scikit_log_reg.predict_proba(X_train)[:,0]

Elegimos las probabilidades de la segunda columna:
scikit_log_reg.predict_proba(X_train)[:,1]

"""

# Recordar distribucion de 'CREDITABILITY' entre 0 y 1: 0 son los malos y 1 los buenos clientes

credit7['CREDITABILITY'] .value_counts()



from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_train, scikit_log_reg.predict_proba(X_train)[:,1])

roc_df = pd.DataFrame({'recall': tpr, 'specificity': 1 - fpr})
ax = roc_df.plot(x='specificity', y='recall', figsize=(4, 4), legend=False)
ax.set_ylim(0, 1)
ax.set_xlim(1, 0)
ax.plot((1, 0), (0, 1))
ax.set_xlabel('specificity')
ax.set_ylabel('recall')

"""

AUC sobre el conjunto de datos de Entrenamiento:

"""

roc_auc = auc(fpr, tpr)
roc_auc


"""

Otra manera de calcular AUC:

"""

print(np.sum(roc_df.recall[:-1] * np.diff(1 - roc_df.specificity)))

print(roc_auc_score(y_train, scikit_log_reg.predict_proba(X_train)[:,1]))



"""

Conjunto de datos de Testeo:

"""


from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_test, scikit_log_reg.predict_proba(X_test)[:,1])

roc_df = pd.DataFrame({'recall': tpr, 'specificity': 1 - fpr})
ax = roc_df.plot(x='specificity', y='recall', figsize=(4, 4), legend=False)
ax.set_ylim(0, 1)
ax.set_xlim(1, 0)
ax.plot((1, 0), (0, 1))
ax.set_xlabel('specificity')
ax.set_ylabel('recall')


"""

AUC sobre el conjunto de datos de Testeo:

"""

roc_auc = auc(fpr, tpr)
roc_auc


"""

Otra manera de calcular AUC:

"""

print(np.sum(roc_df.recall[:-1] * np.diff(1 - roc_df.specificity)))

print(roc_auc_score(y_test, scikit_log_reg.predict_proba(X_test)[:,1]))






from sklearn.metrics import roc_curve, auc

roc_auc = auc(fpr, tpr)

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# endregion Curva de ROC y AUC




# region Lift y Cumulative Gains

"""


El Lift es una de las métricas más utilizadas en el Marketing Directo para medir el desempeño de un modelo de clasificación.

La tasa de respuesta de una campaña es el porcentaje de personas que efectivamente "responden" (por ejemplo: compran el producto) luego 
de haberse ejecutado dicha acción comercial.

El Lift es un ratio entre la tasa de respuesta de un grupo determinado (dentro de un conjunto de datos) dividida la tasa de respuesta promedio de 
todo el conjunto de datos. La tasa de respuesta promedio representa la tasa de respuesta que se obtendría si se seleccionara aleatoriamente a un 
grupo del conjunto sin tener un modelo predictivo. Usualmente el conjunto de datos utilizado y recomendado para calcular el Lift es el de Testeo.



DIFERENCIA ENTRE ROC/AUC y LIFT:
********************************

En el contexto de exponer resultados de un modelo a gente de negocios, Lift y Cumulative Gains son conceptos más sencillos de explicar y entender.

Por otro lado, hay que entender el objetivo de negocio / investigación: si nuestra investigación le da la misma importancia y valor a cometer Error de Tipo I y II, entonces
la métrica de ROC/AUC es más adecuada e integral.

Los problemas de negocio frecuentemente le dan mayor importancia a predecir una determinada clase, antes que a otra. Es decir, que cometer un determinado Tipo de Error
tiene un costo mayor que otro.

Cuando existen restricciones presupuestarias en una campaña comercial, y sólo se permite contactar, por ejemplo, a 5000 de los 25000 clientes,
entonces se busca generalmente identificar los 5000 clientes más propensos a comprar o responder la oferta. Lift y Cumulative Gains son métricas útiles
tanto para explicar cuánto mejor es el modelo predectivo, que simplemente seleccionar al azar clientes, e incluso permiten estimar cuántas ventas se obtendrían
de contactar a los 5000 mejores clientes.


Para más información leer mi tesis de maestría y las siguientes fuentes:
************************************************************************

https://towardsdatascience.com/evaluate-model-performance-with-cumulative-gains-and-lift-curves-1f3f8f79da01

https://towardsdatascience.com/model-lift-the-missing-link-70eb37460e67



"""


"""

Cumulative Gains:
*****************

"""



# pip install scikit-plot

from sklearn import linear_model
import matplotlib.pyplot as plt
import scikitplot as skplt

# get what the predicted probabilities are to use creating cumulative gains chart
predictions = scikit_log_reg.predict_proba(X_train)

skplt.metrics.plot_cumulative_gain(
    y_train, X_train, figsize=(12, 8), title_fontsize=20, text_fontsize=18
)
plt.show()



"""

Para contar la cantidad total de ventas obtenidas del conjunto de entrenamiento y luego poder estimar la cantidad de ventas que se obtendrían
de contactar, por ejemplo, al 20% de los clientes más propensos.

Ejemplo: si hay 500 ventas totales, y Cumulative Gains indica que contactar el 20% de los clientes más propensos captura el 40% de las ventas, entonces
se estima que se obtendrían 200 ventas de las 500 ventas totales.

"""

y_train.value_counts()


"""

Lift:
*****

"""

skplt.metrics.plot_lift_curve(
    y_train, X_train, figsize=(12, 8), title_fontsize=20, text_fontsize=18
)
plt.show()



# endregion Lift





# region KS


"""

Para entender el indicador KS leer la Tesis Maestría Euge. Explica sencillo, rápido y muy bueno.

No encontré en Python cómo hacer graficos de KS y obtener un análisis tan detallado.

"""

# endregion KS




# endregion Logistic Regression in Python



# region Decision Trees


# region Construcción del Arbol de Decisión

"""

1) Para más información leer conceptos con mi tesis de maestría

2) Leer libro:"Practical Statistics for Data Scientists 50+ Essential Concepts Using R and Python" de O’Reilly

3) Páginas de donde obtuve los códigos:

https://www.datacamp.com/community/tutorials/decision-tree-classification-python

https://towardsdatascience.com/decision-tree-in-python-b433ae57fb93

"""

credit7.columns

X = credit7[[
        'ACCOUNT_BALANCE_1', 
        'ACCOUNT_BALANCE_2',
        'ACCOUNT_BALANCE_3',
        
        'cred_duration_bin_optimal_(0, 15]',
        'cred_duration_bin_optimal_(15, 36]',
       
        'age_bin_optimal_(0, 25]',
    
       'GUARANTORS_1', 
       'GUARANTORS_2',
       
       'AGRUP_REPAYMENT_HIST_0', 
       'AGRUP_REPAYMENT_HIST_1',
       'AGRUP_REPAYMENT_HIST_2', 
       
       'AGRUP_SAVINGS_1',
       'AGRUP_SAVINGS_3', 
       
       
       'AGRUP_PURPOSE_0',
       'AGRUP_PURPOSE_1', 
       'AGRUP_PURPOSE_2', 
       'AGRUP_PURPOSE_3',
       'AGRUP_PURPOSE_8', 
       
       'AGRUP_ASSET_1', 
       'AGRUP_ASSET_2', 
       
       
       'AGRUP_LENGTH_EMPLOY_1', 
       'AGRUP_LENGTH_EMPLOY_3',
       'AGRUP_LENGTH_EMPLOY_4', 
       
       
       'AGRUP_TYPE_APART_1',
       
       
       'AGRUP_OTHER_CREDITS_1', 
       
       
       'AGRUP_MARITAL_STATUS_1', 
       'AGRUP_MARITAL_STATUS_2',
       ]]
    
y = credit7['CREDITABILITY']



"""

With train_test_split(), you need to provide the sequences that you want to split as well as any optional arguments.

Estructura basica:

sklearn.model_selection.train_test_split(*arrays, **options) -> list


arrays: is the sequence of lists, NumPy arrays or pandas DataFrames.

options: are the optional keyword arguments that you can use to get desired behavior:


-train_size: is the number that defines the size of the training set. If you provide a float, then it must be between 0.0 and 1.0 and will 
define the share of the dataset used for testing. If you provide an int, then it will represent the total number of the training samples. 
The default value is None.

-test_size: is the number that defines the size of the test set. It’s very similar to train_size. You should provide either train_size or 
test_size. If neither is given, then the default share of the dataset that will be used for testing is 0.25, or 25 percent.

-random_state: Para generar siempre los mismos datasets de entrenamiento y testeo y obtener los mismos resultados.Is the object that controls randomization 
during splitting. It can be either an int or an instance of RandomState. The default value is None. Ejemplo si se explicita el parametro: random_state=4 o random_state=123 
siempre se genererán las mismas muestrar, independientemente que corramos una y otra vez la función train_test_split. Si no se explicita este parámetro, cada vez que volvamos a
correr la función train_test_split se generarán muestras distintas.

-shuffle: is the Boolean object (True by default) that determines whether to shuffle the dataset before applying the split.

-stratify: Para generar muestra de testeo estratificada, donde se respeten las proporciones de casos de una variable determinada, ejemplo de la variable y: stratify=y


https://realpython.com/train-test-split-python-data/



Then, apply train_test_split. For example, you can set the test size to 0.25, and therefore the 
model testing will be based on 25% of the dataset, while the model training will be based on 75% of the dataset:

https://datatofish.com/logistic-regression-python/


"""

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)


"""

Para generar siempre los mismos datasets de entrenamiento y testeo: random_state

Sometimes, to make your tests reproducible, you need a random split with the same output for each function call. 
You can do that with the parameter random_state. The value of random_state isn’t important, it can be any non-negative integer.

Para generar muestra de testeo estratificada, donde se respeten las proporciones de casos de una variable determinada, ejemplo de la variable y: stratify=y

If you want to (approximately) keep the proportion of y values through the training and test sets, then pass stratify=y

Ejemplo:

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4, stratify=y)

"""



"""

Building Decision Tree Model: con criterio de partición de nodo "entropía" y máxima profundidad de 3 niveles.

"""

# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

# Create Decision Tree classifer object

"""

Supported criteria are “gini” for the Gini index and “entropy” for the information gain.

https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

"""


clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))



"""

Visualizing Decision Trees

You can use Scikit-learn's export_graphviz function for display the 
tree within a Jupyter notebook. For plotting tree, you also need to 
install graphviz and pydotplus.

"""

"""

Instalar UNO a la vez, si ejecuto los dos pip en simultaneo no se instalan:
    
pip install graphviz

pip install pydotplus

"""


from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus
from pydot import graph_from_dot_data

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = X.columns,class_names=['0','1'])
(graph, ) = graph_from_dot_data(dot_data.getvalue())
graph.write_png('C:/Users/rodri/Google Drive/Aplicaciones_herramientas_Libros/Python/Credit_Score_Python/CREDITABILITY.png')
Image(graph.create_png())

# endregion Construcción del Arbol de Decisión


# region Confusion Matrix con el modelo de scikit-learn



"""

Generar  Tabla de Contingencia / Confusion Matrix a partir del modelo de scikit-learn

https://realpython.com/logistic-regression-python/#logistic-regression-in-python-with-statsmodels-example
https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
https://datatofish.com/logistic-regression-python/


scikit-learn:

Aplicamos el modelo estimado con el conjunto de datos de entrenamiento sobre el conjunto de datos de testeo.



"""




# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)




X_train.shape

X_test.shape
 





"""

En scikit-learn, el punto de corte de la funcion .predict() por default es 0.5 (default threshold=0.5).

"""


y_pred_train = clf.predict(X_train)


y_pred_train




#Get the confusion matrix

from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(y_train, y_pred_train)
print(cf_matrix)


"""

Obtener las probabilidades estimadas sobre el conjunto de Testeo:

Each row corresponds to a single observation. The first column is 
the probability of the predicted output being zero, that is 1 - 𝑝(𝑥). 
The second column is the probability that the output is one, or 𝑝(𝑥). 

https://realpython.com/logistic-regression-python/#logistic-regression-python-packages   
    

"""

y_Predicted_Proba_test = clf.predict_proba(X_test)


print(y_Predicted_Proba_test)


"""

Obtener los valores predecidos sobre el conjunto de Testeo segun punto de corte: 

"""

threshold = 0.5


y_Predicted_test = (y_Predicted_Proba_test [:,1] >= threshold).astype('int')

y_Predicted_test


"""

Tabla de Confusión sobre el conjunto de Testeo: contrastamos valores predecidos 
sobre el conjunto de testeo Vs. valores reales del conjunto de testeo:

"""


from sklearn.metrics import confusion_matrix
scikit_cm_test = confusion_matrix(y_test, y_Predicted_test)

scikit_cm_test




"""

Obtener la Tabla de Confusión con sklearn sobre el conjunto de Entrenamiento, 
según el punto de corte que se establezca:
    
Ejemplo: threshold = 0.1


"""

y_Predicted_Proba_train = clf.predict_proba(X_train)

y_Predicted_Proba_train



threshold = 0.1


y_Predicted_train = (y_Predicted_Proba_train [:,1] >= threshold).astype('int')

y_Predicted_train


from sklearn.metrics import confusion_matrix
cm_train = confusion_matrix(y_train, y_Predicted_train)

cm_train





"""

Distintas maneras de visualizar tabla de Confusion:

"""


"""

Ejemplo 1: con matplotlib

"""

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(scikit_cm_test)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, scikit_cm_test[i, j], ha='center', va='center', color='red')
plt.show()




"""

Ejemplo 2: seaborn heatmap
    
"""

import seaborn as sns
sns.heatmap(scikit_cm_test, annot=True, fmt='g')

# fix for mpl bug that cuts off top/bottom of seaborn viz
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values
plt.show() # ta-da!




"""

Ejemplo 3: seaborn heatmap
    
"""


sns.heatmap(scikit_cm_test/np.sum(scikit_cm_test), annot=True, 
            fmt='.2%', cmap='Blues')
# fix for mpl bug that cuts off top/bottom of seaborn viz
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values
plt.show() # ta-da!




"""

Ejemplo 4: seaborn heatmap
    
"""



labels = ['True Neg','False Pos','False Neg','True Pos']
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(scikit_cm_test, annot=labels, fmt='', cmap='Blues')

group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                scikit_cm_test.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     scikit_cm_test.flatten()/np.sum(scikit_cm_test)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(scikit_cm_test, annot=labels, fmt='', cmap='Blues')
# fix for mpl bug that cuts off top/bottom of seaborn viz
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values
plt.show() # ta-da!






# endregion Confusion Matrix con el modelo de scikit-learn



# region Precision, recall, fscore, support, Specificity




from sklearn.metrics import confusion_matrix
scikit_cm_test = confusion_matrix(y_test, y_Predicted_test)

"""

Graficamos la tabla de Contigencia:

"""

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(scikit_cm_test)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, scikit_cm_test[i, j], ha='center', va='center', color='red')
plt.show()




"""
Precisión, recall, fscore, support:  de Y=0
"""
print('Precision', scikit_cm_test[0, 0] / sum(scikit_cm_test[:, 0]))
print('Recall', scikit_cm_test[0, 0] / sum(scikit_cm_test[0, :]))
print('Specificity', scikit_cm_test[1, 1] / sum(scikit_cm_test[1, :]))


"""
Precisión, recall, fscore, support:  de Y=1
"""
print('Precision', scikit_cm_test[1, 1] / sum(scikit_cm_test[:, 1]))
print('Recall', scikit_cm_test[1, 1] / sum(scikit_cm_test[1, :]))
print('Specificity', scikit_cm_test[0, 0] / sum(scikit_cm_test[0, :]))





"""

Otra manera de calcular: precision_recall_fscore_support

precision
recall
fscore
support


"""

from sklearn.metrics import precision_recall_fscore_support

"""
precision, recall, fscore, support:  de Y=0 
"""

precision,recall,fscore,support = precision_recall_fscore_support(y_test, y_Predicted_test,labels=[0])


precision

recall

fscore

support

"""
precision, recall, fscore, support:  de Y=0 
"""

precision,recall,fscore,support = precision_recall_fscore_support(y_test, y_Predicted_test,labels=[1])


precision

recall

fscore

support


"""
precision, recall, fscore, support:  de Y=1 y Y=0 en simultaneo
"""

precision,recall,fscore,support = precision_recall_fscore_support(y_test, y_Predicted_test,labels=[0,1])


precision

recall

fscore

support





"""

https://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/modules/generated/sklearn.metrics.precision_recall_fscore_support.html

macro:
Average over classes (does not take imbalance into account).

micro:
Average over instances (takes imbalance into account). This implies that precision == recall == f1

weighted:
Average weighted by support (takes imbalance into account). 
Can result in f1 score that is not between precision and recall.

"""

print(precision_recall_fscore_support(y_test, y_Predicted_test, average='macro'))
print(precision_recall_fscore_support(y_test, y_Predicted_test, average='micro'))
print(precision_recall_fscore_support(y_test, y_Predicted_test, average='weighted'))






# endregion Precision, recall, fscore, support, Specificity



# region Curva de ROC y AUC


"""

https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python

"""

"""

Curva de ROC aplicada al conjunto de datos de Entrenamiento:
    
OJO: con los parametros [:,0] y [:,1] elegimos distintos tipos de probabilidades.

Recordar cuando veíamos Tabla de Contingencia:
    

Each row corresponds to a single observation. The first column is 
the probability of the predicted output being zero, that is 1 - 𝑝(𝑥). 
The second column is the probability that the output is one, or 𝑝(𝑥). 

https://realpython.com/logistic-regression-python/#logistic-regression-python-packages   
    

y_Predicted_Proba_test = clf.predict_proba(X_test)

Elegimos las probabilidades de la primera columna:
clf.predict_proba(X_train)[:,0]

Elegimos las probabilidades de la segunda columna:
clf.predict_proba(X_train)[:,1]

"""

# Recordar distribucion de 'CREDITABILITY' entre 0 y 1: 0 son los malos y 1 los buenos clientes

credit7['CREDITABILITY'] .value_counts()



from sklearn.metrics import roc_curve, auc, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_train, clf.predict_proba(X_train)[:,1])

roc_df = pd.DataFrame({'recall': tpr, 'specificity': 1 - fpr})
ax = roc_df.plot(x='specificity', y='recall', figsize=(4, 4), legend=False)
ax.set_ylim(0, 1)
ax.set_xlim(1, 0)
ax.plot((1, 0), (0, 1))
ax.set_xlabel('specificity')
ax.set_ylabel('recall')

"""

AUC sobre el conjunto de datos de Entrenamiento:

"""

roc_auc = auc(fpr, tpr)
roc_auc


"""

Otra manera de calcular AUC:

"""

print(np.sum(roc_df.recall[:-1] * np.diff(1 - roc_df.specificity)))

print(roc_auc_score(y_train, clf.predict_proba(X_train)[:,1]))



"""

Conjunto de datos de Testeo:

"""


from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_test, scikit_log_reg.predict_proba(X_test)[:,1])

roc_df = pd.DataFrame({'recall': tpr, 'specificity': 1 - fpr})
ax = roc_df.plot(x='specificity', y='recall', figsize=(4, 4), legend=False)
ax.set_ylim(0, 1)
ax.set_xlim(1, 0)
ax.plot((1, 0), (0, 1))
ax.set_xlabel('specificity')
ax.set_ylabel('recall')


"""

AUC sobre el conjunto de datos de Testeo:

"""

roc_auc = auc(fpr, tpr)
roc_auc


"""

Otra manera de calcular AUC:

"""

print(np.sum(roc_df.recall[:-1] * np.diff(1 - roc_df.specificity)))

print(roc_auc_score(y_test, scikit_log_reg.predict_proba(X_test)[:,1]))






from sklearn.metrics import roc_curve, auc

roc_auc = auc(fpr, tpr)

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# endregion Curva de ROC y AUC



# region Lift y Cumulative Gains

"""


El Lift es una de las métricas más utilizadas en el Marketing Directo para medir el desempeño de un modelo de clasificación.

La tasa de respuesta de una campaña es el porcentaje de personas que efectivamente "responden" (por ejemplo: compran el producto) luego 
de haberse ejecutado dicha acción comercial.

El Lift es un ratio entre la tasa de respuesta de un grupo determinado (dentro de un conjunto de datos) dividida la tasa de respuesta promedio de 
todo el conjunto de datos. La tasa de respuesta promedio representa la tasa de respuesta que se obtendría si se seleccionara aleatoriamente a un 
grupo del conjunto sin tener un modelo predictivo. Usualmente el conjunto de datos utilizado y recomendado para calcular el Lift es el de Testeo.



DIFERENCIA ENTRE ROC/AUC y LIFT:
********************************

En el contexto de exponer resultados de un modelo a gente de negocios, Lift y Cumulative Gains son conceptos más sencillos de explicar y entender.

Por otro lado, hay que entender el objetivo de negocio / investigación: si nuestra investigación le da la misma importancia y valor a cometer Error de Tipo I y II, entonces
la métrica de ROC/AUC es más adecuada e integral.

Los problemas de negocio frecuentemente le dan mayor importancia a predecir una determinada clase, antes que a otra. Es decir, que cometer un determinado Tipo de Error
tiene un costo mayor que otro.

Cuando existen restricciones presupuestarias en una campaña comercial, y sólo se permite contactar, por ejemplo, a 5000 de los 25000 clientes,
entonces se busca generalmente identificar los 5000 clientes más propensos a comprar o responder la oferta. Lift y Cumulative Gains son métricas útiles
tanto para explicar cuánto mejor es el modelo predectivo, que simplemente seleccionar al azar clientes, e incluso permiten estimar cuántas ventas se obtendrían
de contactar a los 5000 mejores clientes.


Para más información leer mi tesis de maestría y las siguientes fuentes:
************************************************************************

https://towardsdatascience.com/evaluate-model-performance-with-cumulative-gains-and-lift-curves-1f3f8f79da01

https://towardsdatascience.com/model-lift-the-missing-link-70eb37460e67



"""


"""

Cumulative Gains:
*****************

"""



# pip install scikit-plot

from sklearn import linear_model
import matplotlib.pyplot as plt
import scikitplot as skplt

# get what the predicted probabilities are to use creating cumulative gains chart
predictions = clf.predict_proba(X_train)

skplt.metrics.plot_cumulative_gain(
    y_train, X_train, figsize=(12, 8), title_fontsize=20, text_fontsize=18
)
plt.show()



"""

Para contar la cantidad total de ventas obtenidas del conjunto de entrenamiento y luego poder estimar la cantidad de ventas que se obtendrían
de contactar, por ejemplo, al 20% de los clientes más propensos.

Ejemplo: si hay 500 ventas totales, y Cumulative Gains indica que contactar el 20% de los clientes más propensos captura el 40% de las ventas, entonces
se estima que se obtendrían 200 ventas de las 500 ventas totales.

"""

y_train.value_counts()


"""

Lift:
*****

"""

skplt.metrics.plot_lift_curve(
    y_train, X_train, figsize=(12, 8), title_fontsize=20, text_fontsize=18
)
plt.show()



# endregion Lift



# endregion Decision Trees




# region Intento de obtener mismos resultados que Tuffery

"""

******************************************************************************
******************************************************************************

Intento de obtener mismos resultados que Tuffery: traté pero no pude coincidir resultados.
Por un lado Tuffery utiliza un set de entrenamiento especifico.
Por otro lado, no estoy seguro si es la variable target debería estar invertida resoecto el dataset original:  convertir los 0 en 1 y los 1 en 0.


******************************************************************************
******************************************************************************

Fuentes:
    
https://datatofish.com/logistic-regression-python/

https://realpython.com/logistic-regression-python/


******************************************************************************
******************************************************************************

Selección de variables Dummy y de categorías base:
**************************************************
    
Tener en cuenta que SPSS Modeler selecciona como "categoría base" aquella 
categoría que es representada con el mayor valor númerico. 

Ejemplo: si la variable "Partido Politico" tiene 3 categorías: 
 "Democratas", "Republicanos" y "Verdes" y cada una de ellas es 
 representada respectivamente con los valores respectivos de 1, 2 y 3, 
 entonces la categoría "Verdes" será la "categoría base"

 
Esto se observa de la salida del nodo de regresión logistica en la 
parte "Codificación de los parámetros".

******************************************************************************
******************************************************************************

Para coincidir en resultados con las estimaciones de SPSS Modeler, descartamos 
las variables dummys correspondientes a las categorías con el mayor valor númerico.

Ejemplo: descartamos 'ACCOUNT_BALANCE_4', 'cred_duration_bin_optimal_(36, 100]','GUARANTORS_3',etc.

Set the independent variables (represented as X) and the dependent variable (represented as y):

"""


credit5.columns

credit5['ACCOUNT_BALANCE'].value_counts()

credit5['AGRUP_REPAYMENT_HIST'].value_counts()

credit5['cred_duration_bin_optimal'].value_counts()

credit5['AGRUP_SAVINGS'].value_counts()

credit5['age_bin_optimal'].value_counts()

credit5['AGRUP_PURPOSE'].value_counts()

credit5['GUARANTORS'].value_counts()

credit5['AGRUP_OTHER_CREDITS'].value_counts()


X = credit7[[
        'ACCOUNT_BALANCE_1', 
        'ACCOUNT_BALANCE_2',
        'ACCOUNT_BALANCE_3',
		
        # 'ACCOUNT_BALANCE_4'="No checking account" es la categoría base o de referencia en libro Tuffery
		
	   'AGRUP_REPAYMENT_HIST_0', 
       'AGRUP_REPAYMENT_HIST_2',
	   'AGRUP_REPAYMENT_HIST_4',
	   
	   # 'AGRUP_REPAYMENT_HIST_1'="Previous non-payments"="all credits at this bank paid back duly" es la categoría base o de referencia en libro Tuffery
		
		'cred_duration_bin_optimal_(36, 100]',
        'cred_duration_bin_optimal_(15, 36]',
		
       # 'cred_duration_bin_optimal_(0, 15]' es la categoría base o de referencia en libro Tuffery
	   
       'AGRUP_SAVINGS_1',
       'AGRUP_SAVINGS_3', 
       
       # 'AGRUP_SAVINGS_5="No savings" es la categoría base o de referencia en libro Tuffery
       
       
        'age_bin_optimal_(0, 25]',
    
	   # 'age_bin_optimal_(25, 100]' es la categoría base o de referencia en libro Tuffery
	   
       'AGRUP_PURPOSE_0',
       'AGRUP_PURPOSE_1', 
       'AGRUP_PURPOSE_2', 
       'AGRUP_PURPOSE_8', 
       'AGRUP_PURPOSE_9',

	   
	   # 'AGRUP_PURPOSE_3'="Video HIFI"="radio/television" es la categoría base o de referencia en libro Tuffery
	   
       'GUARANTORS_2',
	   'GUARANTORS_3',
	   
       # 'GUARANTORS_1'="No guarantee" es la categoría base o de referencia en libro Tuffery

       'AGRUP_OTHER_CREDITS_3'
	   
	   # 'AGRUP_OTHER_CREDITS_1'="Other banks or institutions" es la categoría base o de referencia en libro Tuffery

       ]]
    
"""
Resto de las variables son descartadas por el método de stepwise selection:
       

       'AGRUP_ASSET_1', 
       'AGRUP_ASSET_2', 
       
       
       'AGRUP_LENGTH_EMPLOY_1', 
       'AGRUP_LENGTH_EMPLOY_3',
       'AGRUP_LENGTH_EMPLOY_4', 
       
       
       'AGRUP_TYPE_APART_1',
       
       

       
       
       'AGRUP_MARITAL_STATUS_1', 
       'AGRUP_MARITAL_STATUS_2',
"""    
    


def f(row):
    if row['CREDITABILITY'] == 1:
        val = 0
    elif row['CREDITABILITY'] == 0:
        val = 1
    else:
        3
    return val


credit7['CREDITABILITY2'] = credi54.apply(f, axis=1)

credit7['CREDITABILITY2'] 

credit7['CREDITABILITY'] 





y = credit7['CREDITABILITY2']


"""
Para paquete StatsModels:
Se agrega una columna con todos  los valores igual a 1 y así agregar 
intercepto en el modelo.

You can get the inputs and output the same way as you did 
with scikit-learn. However, StatsModels doesn’t take the 
intercept 𝑏₀ into account, and you need to include the 
additional column of ones in x. You do that with add_constant():
    
"""
import statsmodels.api as sm

X = sm.add_constant(X)


X



"""

Logistic Regression in Python With StatsModels




"""



model = sm.Logit(y, X)





"""

Now, you’ve created your model and you should fit it with the 
existing data. You do that with .fit() or, if you want to apply
L1 regularization, with .fit_regularized():

"""

result = model.fit(method='newton')


result.summary()

result.summary2()


# endregion Intento de obtener mismos resultados que Tuffery



# region Otros analisis del Arbol CHAID

"""
**************************************************************************
Otros analisis del Arbol CHAID
**************************************************************************
"""




""""

Otros analisis del Arbol CHAID:

""""






credit['age_bin_deciles2'] 



from CHAID import Tree


import numpy as np


credit.columns

credit


"""
credit3 = credit.set_index(['CREDITABILITY','AGE_YEARS'])

credit3 



credit3.count(level="CREDITABILITY")

"""

independent_variable_columns = ['AGE_YEARS']
dep_variable = 'CREDITABILITY'



"""

parametro "nominal": 

"""
"""
funcionoooooooooo: llamar al conunto credit a secas
https://github.com/Rambatino/CHAID

"""
tree = Tree.from_pandas_df(credit, dict(zip(independent_variable_columns, ['nominal'] * 1)), dep_variable)

tree


"""

parametro "ordinal": 

"""

tree = Tree.from_pandas_df(credit, dict(zip(independent_variable_columns, ['ordinal'] * 1)), dep_variable)

tree




tree.print_tree()


"""
to get a LibTree object:

"""
    
tree.to_tree()


"""

 the different nodes of the tree can be accessed like

"""

first_node = tree.tree_store[0]

first_node


second_node = tree.tree_store[1]

second_node


third_node = tree.tree_store[2]

third_node

fourth_node = tree.tree_store[3]

fourth_node

fith_node = tree.tree_store[4]

fith_node

sixth_node = tree.tree_store[5]

sixth_node

"""
the properties of the node can be access like
"""

first_node.members

"""
the properties of split can be accessed like

"""

first_node.split.p

first_node.split.score


"""

Generating Splitting Rules

"""

tree.classification_rules(first_node)

tree.classification_rules(second_node)

tree.classification_rules(third_node)

tree.classification_rules(fourth_node)



"""

Cambiar parametro modelo de Arbol:
min_parent_node_size=2
min_child_node_size=7

split_threshold: Float (default = 0): The split threshold when bucketing root node surrogate splits

"""



tree2 = Tree.from_pandas_df(credit, dict(zip(independent_variable_columns, ['ordinal'] * 1)), dep_variable , min_parent_node_size=30, min_child_node_size=100)

tree2


tree3 = Tree.from_pandas_df(credit, dict(zip(independent_variable_columns, ['ordinal'] * 1)), dep_variable , split_threshold=30)

tree3





"""

Utilizamos algoritmo CHAID (disponible sólo para variables nomniales y oridinales)
sobre la variable ordinal en deciles generada a partir de la variable AGE


"""


credit['age_bin_deciles'] = pd.qcut(credit['AGE_YEARS'], 10, labels=False)

"""
credit['age_bin_deciles2'] = pd.qcut(credit['AGE_YEARS'], 10)

"""

credit['age_bin_deciles'] 






from CHAID import Tree


import numpy as np


credit.columns

credit



independent_variable_columns = ['age_bin_deciles']
dep_variable = 'CREDITABILITY'




tree = Tree.from_pandas_df(credit, dict(zip(independent_variable_columns, ['ordinal'] * 1)), dep_variable)

tree








"""

Otras variables explicativas:
****************************

"""


credit.columns

credit

independent_variable_columns = ['DURATION_OF_CREDIT_MONTH']
dep_variable = 'CREDITABILITY'


tree = Tree.from_pandas_df(credit, dict(zip(independent_variable_columns, ['ordinal'] * 1)), dep_variable)

tree



credit.columns

credit

independent_variable_columns = ['CREDIT_AMOUNT']
dep_variable = 'CREDITABILITY'


tree = Tree.from_pandas_df(credit, dict(zip(independent_variable_columns, ['ordinal'] * 1)), dep_variable)

tree



"""
******************************************************************************
******************************************************************************

Ejercicio del tutorial de CHAID en Python

******************************************************************************
******************************************************************************

"""



"""
************************************************************************************
************************************************************************************
************************************************************************************

Chi-Squared Automatic Inference Detection (CHAID) decision tree

https://github.com/Rambatino/CHAID

https://libraries.io/pypi/CHAID


************************************************************************************
************************************************************************************
************************************************************************************


"""


pip install CHAID



from CHAID import Tree


import numpy as np


"""
Generamos números aleatorios y los pasamos a un dataframe de pandas.

Generamos las variables dataframe son 'a', 'b', 'c'
y luego la variable 'd'.

"""

"""
create the data
"""

ndarr = np.array(([1, 2, 3] * 5) + ([2, 2, 3] * 5)).reshape(10, 3)

ndarr

import pandas as pd
df = pd.DataFrame(ndarr)

df

df.columns = ['a', 'b', 'c']

df

arr = np.array(([1] * 5) + ([2] * 5))

arr

df['d'] = arr

df


"""

set the CHAID input parameters
Establecemos variables independientes son 'a', 'b', 'c'
y luego la variable 'd' es dependiente.

"""


independent_variable_columns = ['a', 'b', 'c']
dep_variable = 'd'


var_dict=dict(zip(independent_variable_columns, ['nominal'] * 3))

var_dict

var_dict2=dict(zip(independent_variable_columns, ['nominal'] * 2))

var_dict2

var_dict1=dict(zip(independent_variable_columns, ['nominal'] * 1))

var_dict1

"""
Investigar que es zip function y dict function:

https://realpython.com/python-zip-function/
"""

create the Tree via pandas
tree = Tree.from_pandas_df(df, dict(zip(independent_variable_columns, ['nominal'] * 3)), dep_variable)

tree

"""
create the same tree, but without pandas helper

"""
tree = Tree.from_numpy(ndarr, arr, split_titles=['a', 'b', 'c'], min_child_node_size=5)

tree


tree.print_tree()

tree.to_tree()

first_node = tree.tree_store[0]

first_node


first_node.members




NominalColumn(ndarr[:,0], name='a')

"""
 create the same tree using the tree constructor
"""
cols = [NominalColumn(ndarr[:,0], name='a')
        NominalColumn(ndarr[:,1], name='b')
        NominalColumn(ndarr[:,2], name='c')
        ]


tree = Tree(cols, NominalColumn(arr, name='d'), {'min_child_node_size': 5})





tree.to_tree()

tree.render(path=None, view=False)







# endregion Otros analisis del Arbol CHAID





# region Matriz de Correlación

"""
**************************************************************************
Matriz de Correlación
**************************************************************************
"""





"""

import seaborn as sns
iris = sns.load_dataset
iris.head()
#escribiur rows dentro de In de Terminal

iris = sns.load_dataset('rows')

print(iris)



iris = sns.load_dataset(rows)
iris.head()


"""

"""
El siguiente código genera Matriz de Correlación, demora mucho y se obtienen muchos cuadros.

Es muy pesado el Output. Conviene guardarlo como imagen y abrila desde un Paint.

"""
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.pairplot(pd.DataFrame(rows, columns=[x[0] for x in cursor1.description]), hue='Account Balance', size=5);


import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.pairplot(pd.DataFrame(rows, columns=[x[0] for x in cursor1.description]), hue='Creditability', size=5);


# endregion Matriz de Correlación





# region Optimal Binning

"""
**************************************************************************
Optimal Binning: ojo invertí tiempo y pregunté a Diego, parece que 
este algoritmo está roto y no funciona. Salta error
**************************************************************************
"""




"""

**************************************************************************

Optimal Binning

https://pypi.org/project/optbinning/

http://gnpalencia.org/optbinning/


http://gnpalencia.org/optbinning/tutorials/tutorial_binary.html



**************************************************************************


"""


pip install optbinning==0.2.0


pip install optbinning==0.4.0


pip install optbinning==0.1.0

pip install optbinning

pip install optbinning==0.5.0

"""
conocer todos los archivos que tiene un paquete que hayamos 
instalado lo hacemos mediante la sintaxis:

"""

pip show --files optbinning


pip show --files numpy


pip show --files math


pip install wxPython

pip show --files wxPython

import wx


from random import randint



"""

Para conocer todos los paquetes instalados en 
nuestro entorno de Python debemos utilizar el comando 'list':

"""

pip list



"""

Para actualizar un paquete ya instalado 
debemos pasar el parámetro 'upgrade':
   
"""

pip install --upgrade optbinning



"""
To install from source, download or clone the git repository

git clone https://github.com/guillermo-navas-palencia/optbinning.git

cd optbinning
python setup.py install

"""


git clone https://github.com/guillermo-navas-palencia/optbinning.git
cd optbinning
python setup.py install



git clone git://github.com/pudo/dataset.git
pip install ./dataset


import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer


data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)

df

"""

We choose a variable to discretize and the binary target.

"""


variable = "mean radius"
x = df[variable].values
y = data.target

x
y


"""

No estoy seguro si es un problema de sistema operativo que funciona con 
linux, pero no logro resolver el problema:
    
"ImportError: DLL load failed while importing _pywrapsat:"
 from . import _pywrapsat
 
 ver si tal vez aprendiendo a instalar de otra manera puedo:
     
     https://pypi.org/project/optbinning/#files

https://packaging.python.org/tutorials/installing-packages/



Import and instantiate an OptimalBinning object class. 
We pass the variable name, its data type, and a solver, in this case, 
we choose the constraint programming solver.

ojo a partir de aca salta problema:

"""

import OptimalBinning as opt

import optbinning as opt



cd optbinning
python setup.py install


from optbinning import OptimalBinning

optb = OptimalBinning(name=variable, dtype="numerical", solver="cp")




conda install spyder=4.1.3



# endregion Optimal Binning




# region intento fallido de graficar arbol CHAID

"""
**************************************************************************
intento fallido de graficar arbol CHAID
**************************************************************************
"""

"""

Aprender a graficar arbol CHAID

https://github.com/Rambatino/CHAID
"""


tree.to_tree()

tree.render(path=None, view=False)



def render(self, path=None, view=True):
    Graph(self).render(path, view)
    
Graph(self)

def render(self, path=None, view=False):
    Graph(self).render(path, view)
    
    
    
    
    
    
    
    
    
    def render(self, path, view):
        if path is None:
            path = os.path.join("trees", "{:%Y-%m-%d %H:%M:%S}.gv".format(datetime.now()))
        with TemporaryDirectory() as self.tempdir:
            g = Digraph(
                format="png",
                graph_attr={"splines": "ortho"},
                node_attr={"shape": "plaintext", "labelloc": "b"},
            )
            for node in self.tree:
                image = self.bar_chart(node)
                g.node(str(node.node_id), image=image)
                if node.parent is not None:
                    edge_label = "     ({})     \n ".format(', '.join(map(str, node.choices)))
                    g.edge(str(node.parent), str(node.node_id), xlabel=edge_label)
            g.render(path, view=view)



    

from tree.Tree import export_graphviz

export_graphviz(
        tree_clf,
        out_file="chaid.dot",
        feature_names=iris.feature_names[2:],
        class_names=iris.target_names,
        rounded=True,
        filled=True
)



tree.render(path=None, view=False)

"""
In order to use visually graph the CHAID tree, you'll need to install two more libraries that aren't distributed via pypi:

1) graphviz - see here for platform specific installations
2) orca - see the README.md for platform specific installations

"""


"""
Para instalar python-graphviz:

https://anaconda.org/anaconda/python-graphviz


Fuente:
    
https://stackoverflow.com/questions/33433274/anaconda-graphviz-cant-import-after-installation

pdate: There exists now a python-graphviz package at Anaconda.org which contains the Python interface for the graphviz tool. Simply install it with conda install python-graphviz.
(Thanks to wedran and g-kaklam for posting this solution and to endolith for notifying me).


"""

conda install -c anaconda python-graphviz


import graphviz

with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)



"""

Para instalar orca:

Fuente: https://github.com/plotly/orca

"""

conda install -c plotly plotly-orca

import plotly-orca



prueba_dict=dict(zip(independent_variable_columns, ['nominal'] * 1))

prueba_dict

"""
tree = Tree.from_pandas_df(credit3, dict(zip(independent_variable_columns, ['ordinal'] * 3)), dep_variable)


"""


from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


# endregion intento fallido de graficar arbol CHAID


