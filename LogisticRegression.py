from pyspark.ml.feature import StringIndexer, VectorAssembler, MinMaxScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.types import DoubleType
from pyspark.sql import SparkSession
from pyspark.sql.functions import * 
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
import matplotlib.pyplot as plt
import random

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
  print(f"item <{mode}> is most common in <{column}>Column")
  df = df.fillna(mode, subset=column)

# For null unkown valuse in columns with string type, if it was less than 100 remove and if it was more thanreplace with most common
for column in str_cols:
  count_unknown = df.filter(col(column) == 'unknown').count()
  print(f"Column {column} has {count_unknown} unkown items")
  if count_unknown > 0 and count_unknown < 100:
    df = df.filter(col(column) != 'unknown')
  elif count_unknown >= 100 :
    most_common = df.groupBy(column).count().orderBy(col("count").desc()).collect()[0][0]
    print(f"------> {most_common} is the most common in column {column}")
    df = df.withColumn(column, when(col(column) == 'unknown', most_common).otherwise(col(column)))

# BarChart for numeric cols
for item in num_cols:
  color = ["#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)])]
  df.toPandas().hist(column = item, grid=False, color = color)

#Show Each value count for categorical cols
for item in categorical_cols:
  df.groupBy(item).count().orderBy(col("count").desc()).show() 

# Change categortical values to numeric
for column in categorical_cols:
  indexer = StringIndexer(inputCol=column, outputCol=column+"_index")
  df = indexer.fit(df).transform(df)

#Checking correlation between features and label
corr_df = df.select(corr('age', 'y').alias('age_y'),
                    corr('marital_index', 'y').alias('marital_index_y'),
                    corr('education_index', 'y').alias('education_index_y'),
                    corr('housing_index', 'y').alias('housgin_index'),
                    corr('loan_index', 'y').alias('loan_index_y'),
                    corr('contact_index', 'y').alias('contact_index_y'),
                    corr('duration', 'y').alias('duration_y'),
                    corr('campaign', 'y').alias('campaign_y'),
                    corr('pdays', 'y').alias('pdays_y'),
                    corr('previous', 'y').alias('previous_y'),
                    corr('emp_var_rate', 'y').alias('emp_var_rate_y'),
                    corr('cons_price_idx', 'y').alias('cons_price_idx_y'),
                    corr('cons_conf_idx', 'y').alias('cons_conf_idx_y'),
                    corr('euribor3m', 'y').alias('euribor3m_y'),
                    corr('nr_employed', 'y').alias('nr_employed')
                    )
                    
corr_df.show()

# Visualize correlations
import seaborn as sns
import matplotlib.pyplot as plt

plt.subplots(figsize=(15,10))
sns.heatmap(corr_df.toPandas(), annot=True, cmap='coolwarm')
plt.title('Correlation between features and label')
plt.show()

#Checking correlation between features
corr_all = df.select("y", "age","marital_index", "education_index", "housing_index", "loan_index", "contact_index", "duration", "campaign", "pdays", "previous", "emp_var_rate", "cons_price_idx", "cons_conf_idx", "euribor3m", "nr_employed").toPandas()
corr = corr_all.corr()

plt.subplots(figsize=(15,10))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation between features')
plt.show()

# Create feature col with Selected columns and then normalize it
assembler = VectorAssembler(inputCols=["duration", "pdays", "previous", "emp_var_rate", "cons_price_idx", "euribor3m", "nr_employed"], outputCol="features")
df = assembler.transform(df)
scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features")
scaler_model = scaler.fit(df)
df = scaler_model.transform(df)

df.toPandas().to_csv('bank.csv')

#Selecting relevant columns for logistic regression
df = df.select("scaled_features", "duration", "campaign", "previous", "pdays", "emp_var_rate", "cons_price_idx", "euribor3m", "nr_employed", "y")

#Splitting the data into training and testing sets
train_data, test_data = df.randomSplit([0.7, 0.3], seed=42)

#Running logistic regression algorithm
lr = LogisticRegression(featuresCol="scaled_features", labelCol="y")
lr_model = lr.fit(train_data)
predictions = lr_model.transform(test_data)
predictions.limit(100).show()

#Evaluating the model
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="y")
accuracy = evaluator.evaluate(predictions)
print("Accuracy of logistic regression: ", accuracy)

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

evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="y")
auc = evaluator.evaluate(predictions)
print("AUC: {}".format(auc))