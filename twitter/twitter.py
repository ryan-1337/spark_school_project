from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, lower, date_format, lit, concat, round

spark = SparkSession.builder.appName("TrumpTweetAnalysis").getOrCreate()

df_tweets = spark.read.option("header", "true").csv(
    "trump_insult_tweets_2014_to_2021.csv", encoding="latin1"
)

# Conversion de la colonne date en type date
df_tweets = df_tweets.withColumn("date", col("date").cast("date"))

# Mise en minuscule du texte des tweets pour une analyse uniforme
df_tweets = df_tweets.withColumn("tweet", lower(col("tweet")))

top_targets = (
    df_tweets.filter(col("target").isNotNull())
    .groupBy("target")
    .agg(count("*").alias("nb_insults"))
    .orderBy(col("nb_insults").desc())
    .limit(7)
)

total_targets = df_tweets.filter(col("target").isNotNull()).count()
target_percentages = top_targets.withColumn(
    "percent", round((col("nb_insults") / total_targets) * 100, 2)
)

print("Les 7 cibles les plus insultées par Donald Trump (nombre et pourcentage) :")
target_percentages.show()

frequent_insults = (
    df_tweets.filter(col("insult").isNotNull())
    .groupBy("insult")
    .agg(count("*").alias("nb_insults"))
    .orderBy(col("nb_insults").desc())
)

total_insults = df_tweets.filter(col("insult").isNotNull()).count()
insult_percentages = frequent_insults.withColumn(
    "percent", round((col("nb_insults") / total_insults) * 100, 2)
)

print("Les insultes les plus fréquentes de Donald Trump (nombre et pourcentage) :")
insult_percentages.show()

biden_insults = (
    df_tweets.filter((col("target") == "joe-biden") & col("insult").isNotNull())
    .groupBy("insult")
    .agg(count("*").alias("nb_insults"))
    .orderBy(col("nb_insults").desc())
    .limit(1)
)

print("L'insulte la plus utilisée contre Joe Biden :")
biden_insults.show()

# Compter les tweets contenant certains mots-clés
nb_tweets_mexico = df_tweets.filter(col("tweet").contains("mexico")).count()
nb_tweets_china = df_tweets.filter(col("tweet").contains("china")).count()
nb_tweets_covid = df_tweets.filter(col("tweet").contains("coronavirus")).count()

print("Nombre de tweets mentionnant 'Mexico' :", nb_tweets_mexico)
print("Nombre de tweets mentionnant 'China' :", nb_tweets_china)
print("Nombre de tweets mentionnant 'coronavirus' :", nb_tweets_covid)

df_tweets = df_tweets.filter(col("date").isNotNull())

# Créer une période mensuelle pour chaque tweet
df_tweets = df_tweets.withColumn(
    "month_period", concat(date_format(col("date"), "yyyy-MM"), lit("-01"))
)

# Compter le nombre de tweets par période mensuelle
tweets_per_period = (
    df_tweets.groupBy("month_period").agg(count("*").alias("nb_tweets")).orderBy("month_period")
)

print("Nombre de tweets par période mensuelle :")
tweets_per_period.show()

spark.stop()
