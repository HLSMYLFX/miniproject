import findspark
findspark.init()
from pyspark.sql.functions import month
from pyspark.sql.functions import regexp_replace, col, to_date, unix_timestamp, date_format, to_timestamp
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('lin_reg').getOrCreate()

# -----------------------------------------Read-----------------------------------------
import pandas as pd
df = spark.read.csv('access_log',inferSchema=True,header=False, sep=" ")

# -----------------------------------------Select-----------------------------------------
#model_df = df.select('_c0','_c3')
model_df = df.select('_c3')
model_df = model_df.withColumn("_c3",regexp_replace(col("_c3"),"[\\[]",""))
model_df = model_df.withColumn("date", to_date(unix_timestamp("_c3","dd/MMM/yyyy:HH:mm:ss").cast("timestamp")))
model_df = model_df.withColumn("month", month(to_timestamp("date","yyyy/MM/dd")))
model_df = model_df.groupBy("month").count()
model_df = model_df.withColumn("x",col("month")*0+1)
model_df = model_df.na.drop()  # 扔掉任何列包含na的行
model_df.show(5, False)

from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=['month','x'], outputCol="features")
output = assembler.transform(model_df)
label_features = output.select("features", "count").toDF('features','label')
label_features.show()


# ------------------------------------------LR--------------------------------------------
from pyspark.ml.regression import LinearRegression


print('-------------- LR ------------------')

lin_Reg = LinearRegression(featuresCol='features', labelCol='label')

lr_model = lin_Reg.fit(label_features)

#intercept
print('{}{}'.format('intercept:',lr_model.intercept))
training_predictions = lr_model.evaluate(label_features)

#MSE
print('{}{}'.format('MSE:',training_predictions.meanSquaredError))


#It is used to determine whether the built model can accurately predict. The larger the prediction, the higher the accuracy of the prediction.
print('{}{}'.format('coefficient：',training_predictions.r2))

