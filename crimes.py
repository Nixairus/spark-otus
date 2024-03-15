from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, concat_ws, mean, dense_rank, desc, count, collect_list
from pyspark.sql.window import Window

# Создание Spark сессии
spark = SparkSession.builder.appName("Boston crimes statistics").getOrCreate()

# Загрузка данных
data_path = "crime.csv"
offense_codes_path = "offense_codes.csv"

data = spark.read.format("csv").option("header", "true").load(data_path)
offense_codes = spark.read.format("csv").option("header", "true").load(offense_codes_path)

# Очистка данных
data = data.dropDuplicates()

# Определение широты и долготы района
aggregated_data = (
    data.groupBy("DISTRICT")
    .agg(
        count("INCIDENT_NUMBER").alias("crimes_total"),
        mean("Lat").alias("lat"),
        mean("Long").alias("lng")
    )
)

# Разбиение NAME из offense_codes на crime_type
offense_codes = (
    offense_codes.withColumn("crime_type", split(col("NAME"), " - ")[0])
    .select("CODE", "crime_type")
)

# Присоединение offense_codes к data для получения crime_type
crime_types_data = data.join(offense_codes, data.OFFENSE_CODE == offense_codes.CODE, "left")

# Расчет crimes_monthly
monthly_data = (
    data.groupBy("DISTRICT", "YEAR", "MONTH")
    .agg(count("INCIDENT_NUMBER").alias("crimes"))
    .groupBy("DISTRICT")
    .agg(
        mean("crimes").alias("crimes_monthly")
    )
)

# Расчет frequent_crime_types
w = Window.partitionBy("DISTRICT").orderBy(col("crime_count").desc())
frequent_crime_types = (
    crime_types_data.groupBy("DISTRICT", "crime_type")
    .agg(count("INCIDENT_NUMBER").alias("crime_count"))
    .withColumn("rank", dense_rank().over(w))
    .filter(col("rank") <= 3)
    .groupBy("DISTRICT")
    .agg(concat_ws(", ", collect_list("crime_type")).alias("frequent_crime_types"))
)

# Объединение всех данных
result = aggregated_data.join(monthly_data, "DISTRICT", "left")
result = result.join(frequent_crime_types, "DISTRICT", "left")

# Сохранение витрины в формате Parquet
output_folder_path = "output"
result.write.mode("overwrite").parquet(output_folder_path)
