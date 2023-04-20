from pyspark.ml.feature import StringIndexer, VectorAssembler, MinMaxScaler
from pyspark.ml.classification import LinearSVC
from pyspark.sql.types import DoubleType
from pyspark.sql import SparkSession
from pyspark.sql.functions import * 
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

spark = SparkSession.builder.appName("Bank Marketing").getOrCreate()

df = spark.read.csv("ML_hw_dataset.csv", header=True, inferSchema=True)

num_cols = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp_var_rate', 'cons_price_idx', 'cons_conf_idx', 'euribor3m', 'nr_employed']
str_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
categorical_cols = [ 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']

mean_values = df.select([mean(c).alias(c) for c in num_cols]).collect()[0].asDict()

# For null values in columns with int and double type, replace with mean
for column in num_cols:
  df = df.withColumn(column, when(col(column).isNull(), mean_values[column]).otherwise(col(column)))
# Handle categorical null values by replacing them with mode 
for column in categorical_cols:
  mode = df.groupBy(column).count().orderBy('count', ascending=False).limit(1).select(column).collect()[0][0]
  df = df.fillna(mode, subset=column)

# For null unkown valuse in columns with string type, if it was less than 100 remove and if it was more thanreplace with most common
for column in str_cols:
  count_unknown = df.filter(col(column) == 'unknown').count()
  if count_unknown > 0 and count_unknown < 100:
    df = df.filter(col(column) != 'unknown')
  elif count_unknown >= 100 :
    most_common = df.groupBy(column).count().orderBy(col("count").desc()).collect()[0][0]
    df = df.withColumn(column, when(col(column) == 'unknown', most_common).otherwise(col(column)))

# change outliers in num_cols with avg_val
for col_name in num_cols:
    avg_val = df.select(mean(col(col_name))).collect()[0][0]
    std_dev = df.select(stddev(col(col_name))).collect()[0][0]
    lower_bound = avg_val - 3*std_dev
    upper_bound = avg_val + 3*std_dev
    df = df.withColumn(col_name, \
                       when(col(col_name) < lower_bound, avg_val)\
                        .when(col(col_name) > upper_bound, avg_val)\
                        .otherwise(col(col_name)))


 # Change categortical values to numeric
for column in categorical_cols:
  indexer = StringIndexer(inputCol=column, outputCol=column+"_index")
  df = indexer.fit(df).transform(df)
   
 
# Create feature col with Selected columns and then normalize it
assembler = VectorAssembler(inputCols=["duration", "pdays", "previous", "emp_var_rate", "cons_price_idx", "euribor3m", "nr_employed"], outputCol="features")
df = assembler.transform(df)
scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features")
scaler_model = scaler.fit(df)
df = scaler_model.transform(df)


#Selecting relevant columns for logistic regression
df = df.select("scaled_features", "y")

#Splitting the data into training and testing sets
train_data, test_data = df.randomSplit([0.7, 0.3], seed=42)

#Running SVM algorithm
svm = LinearSVC(featuresCol="scaled_features", labelCol="y", maxIter=10, regParam=0.1)
svm_model = svm.fit(train_data)

predictions = svm_model.transform(test_data)
predictions.show(20)
evaluator = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction', labelCol='y')
auc = evaluator.evaluate(predictions)
print('AUC:', auc)


true_positive = predictions.filter(col("prediction") == 1.0).filter(col("y") == 1.0).count()
false_positive = predictions.filter(col("prediction") == 1.0).filter(col("y") == 0.0).count()
true_negative = predictions.filter(col("prediction") == 0.0).filter(col("y") == 0.0).count()
false_negative = predictions.filter(col("prediction") == 0.0).filter(col("y") == 1.0).count()

accuracy = (true_positive + true_negative) / float(predictions.count())
precision = true_positive / float(true_positive + false_positive)
recall = true_positive / float(true_positive + false_negative)
f1_score = 2 * ((precision * recall) / (precision + recall))

print("Accuracy: {}".format(accuracy))
print("Precision: {}".format(precision))
print("Recall: {}".format(recall))
print("F1 Score: {}".format(f1_score))