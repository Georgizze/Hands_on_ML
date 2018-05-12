import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.tools.plotting import scatter_matrix

HOUSING_PATH = "datasets/housing"

csv_path = os.path.join(HOUSING_PATH, "housing.csv")
housing = pd.read_csv(csv_path)

# IMPRIME PRIMERAS 5 FILAS
#print(housing.head())

# INFO, TIPO DE DATO DE CADA ATRIBUTO
#print(housing.info())

# AGRUPAMIENTO DE ATRIBUTOS CATEGORICOS
#print(housing['ocean_proximity'].value_counts())

# RESUMEN DE CAMPOS NUMERICOS
#print(housing.describe())

# IMPRIMO HISTOGRAMAS PARA CADA ATRIBUTO NUMERICO
#housing.hist(bins=50, figsize=(20,15))
#plt.show()


###### ARMAR LOS SETS DE ENTRENAMIENTO Y PRUEBA

# YA QUE EL INGRESO MEDIO ES IMPORTANTE PARA PREDECIR VALORES DE CASA (SUPOSICION DE UN EXPERTO)
# AGREGAMOS UNA COLUMNA PARA DIVIDIR POR RANGOS LOS INGRESOS Y EN BASE A ELLOS HACER LA SEPARACION DEL LOS SETS

housing["income_cat"] = np.ceil(housing["median_income"]/1.5)
housing["income_cat"].where(housing["income_cat"]<5, 5.0, inplace=True)

#print(housing)

# SEPARO EN DOS EL SET COMPLETO, 80% PARA ENTRENAMIENTO Y 20% PARA PRUEBA
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
	strat_train_set = housing.loc[train_index]
	strat_test_set = housing.loc[test_index]

#print(len(strat_train_set))
#print(len(strat_test_set))

# AL SER UN ATRIBUTO IMPORTANTE, SEPARO POR EL PARA TENER LA MISMA PROPORCION EN LOS DATOS DE ENTRENAMIENTO Y PRUEBA SOBRE EL TOTAL
#print(housing["income_cat"].value_counts() / len(housing))
#print(strat_train_set["income_cat"].value_counts() / len(strat_train_set))
#print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

# BORRO LA COLUMNA GENERADA PARA DEJAR EL SET ORIGINAL COMO ESTABA
for set_ in(strat_test_set, strat_train_set):
	set_.drop("income_cat", axis=1, inplace=True)

#print(strat_train_set)
#print(strat_test_set)

# HAGO UNA COPIA PARA EXPLORAR EL SET SIN DAÑAR EL ORIGINAL
housing = strat_train_set.copy()

# PLOTEO LONGITUD Y LATITUD
#housing.plot(kind='scatter', x='longitude', y='latitude')

# PARA VER LA CONCENTRACION EN LOS PUNTOS (DENSIDAD)
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)

# AHORA LA DENSIDAD VA A ESTAR DADA POR EL VALOR MEDIO DE LA PROPIEDAD
# Y EL RADIO DE CADA CIRCULO REPRESANTA LA POBLACION DEL DISTRITO
#housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1,
#	s=housing["population"]/100, label="Population", figsize=(10,7),
#	c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True
#	)

#plt.legend()

#plt.show()

# CORRELACIONES ENTRE PARES DE ATRIBUTOS
corr_matrix = housing.corr()

#print(corr_matrix)

#print(corr_matrix["median_house_value"].sort_values(ascending=False))

# USO PANDAS PARA IMPRIMIR LAS CORRELACIONES ENTRE ALGUNOS ATRIBUTOS, PORQUE SERIAN 121 GRAFICOS PARA TODOS (11 x 11)
#attributes = ["median_house_value", "median_income","total_rooms", "housing_median_age"]

#scatter_matrix(housing[attributes],figsize=(12,8))

#housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)

#plt.show()

# HAY ATRIBUTOS QUE POR SI SOLOS NO PORVEEN MUCHA INFORMACION
# VAMOS A CREAR ALGUNOS QUE PUEDEN SER UTILIES

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]

# VEAMOS NUEVAMENTE LAS CORRELACIONES ENTRE PARES DE ATRIBUTOS
corr_matrix = housing.corr()

#print(corr_matrix["median_house_value"].sort_values(ascending=False))


##########################################################
# PREPARAR LOS DATOS PARA EL ALGORITMO DE MACHINE LEARNING
##########################################################

# SEPARO LAS FEATURES DE LOS LABELS

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# UTILIZO LA CLASE IMPUTER DE SKLEARN QUE COMPLETA LOS VALORES FALTANTES CON UNO SELECCIONADO
from sklearn.preprocessing import Imputer

# EN ESTE CASO UTILIZO LA MEDIA
imputer = Imputer(strategy="median")

# SOLO TIENE QUE HABER VALORES NUMERICOS
housing_num = housing.drop("ocean_proximity", axis=1)

imputer.fit(housing_num)

#print(imputer.statistics_)

X = imputer.transform(housing_num)

housing_tr = pd.DataFrame(X, columns=housing_num.columns)

#print(housing_tr)


# AHORA EL MANEJO DE ATRIBUTOS DE TEXTO

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

housing_cat = housing["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat)
#print(housing_cat_encoded)

#print(encoder.classes_)

# HAY UN PROBLEMA CON ESTE METODO, POR PROXIMIDAD LA CLASE 0 Y 4 SON CERCANAS PERO NO SE REFLEJA CON EN NUMERO DE CATEGORIA
# PARA EVITAR ESTO SE USA LA CLASE ONEHOTENCODER QUE PERMITE TOMAR DE A PARES Y EVALUAR LA CONDICION

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()

housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
#print(type(housing_cat_1hot))


from sklearn.preprocessing import LabelBinarizer

encoder = LabelBinarizer()

housing_cat_1hot = encoder.fit_transform(housing_cat)

#print(housing_cat_1hot)


# PARA NO TENER QUE COPIAR TODOS ESTOS PASOS CADA VEZ QUE SE UTILIZA
# PODEMOS HACER PIPELINES QUE EJECUTAN LOS PASOS EN EL ORDEN INDICADO
# Y LA SALIDA DE CADA PASO ES LA ENTRADA DEL SIGUIENTE

# DEFINAMOS UNA CLASE QUE AGREGA LAS COLUMNAS CALCULADAS ANTERIORMENTE
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
	def __init__(self, add_bedrooms_per_room = True):
		self.add_bedrooms_per_room = add_bedrooms_per_room
	def fit(self, X, y=None):
		return self
	def transform(self, X, y=None):
		rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
		population_per_household = X[:, population_ix] / X[:, household_ix]
		if self.add_bedrooms_per_room:
			bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
			return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
		else:
			return no.c_[X, rooms_per_household, population_per_household]


# DEFINAMOS UNA CLASE QUE SELECCIONA LAS COLUMNAS DESEADAS DE UN DATAFRAME
class DataFrameSelector(BaseEstimator, TransformerMixin):
	def __init__(self, attribute_names):
		self.attribute_names = attribute_names
	def fit(self, X, y=None):
		return self
	def transform(self, X):
		return X[self.attribute_names].values

# ARMAMOS EL PIPELINE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
	('selector', DataFrameSelector(num_attribs)),
	('imputer', Imputer(strategy="median")),
	('attribs_adder', CombinedAttributesAdder()),
	('std_scaler', StandardScaler()),
	])

cat_pipeline = Pipeline([
	('selector', DataFrameSelector(cat_attribs)),
	('label_binarizer', LabelBinarizer()),
	])


# PODEMOS UNIR LOS PIPELINES PARA QUE SE CORRAN EN UNO SOLO
from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list=[
	("num_pipeline", num_pipeline),
	("cat_pipeline", cat_pipeline),
	])

housing_prepared = full_pipeline.fit_transform(housing)

#print(housing)
#print(housing_prepared)


##########################################
# SELECCIONAR Y ENTRENAR EL MODELO DE ML #
##########################################

print(housing_prepared)
print(housing_labels)



