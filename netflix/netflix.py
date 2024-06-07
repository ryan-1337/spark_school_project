
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, split, explode, regexp_extract, avg, desc

spark = SparkSession.builder.appName("NetflixContentAnalysis").getOrCreate()

netflix_data = (
    spark.read.option("header", "true")
    .option("encoding", "UTF-8")
    .csv("netflix_titles.csv")
)

films_df = netflix_data.filter(col("type") == "Movie")

# Calcul du nombre de films par réalisateur
directors_movies_count = (
    films_df.filter(col("director").isNotNull())
    .groupBy("director")
    .agg(count("title").alias("total_movies"))
    .orderBy(col("total_movies").desc())
)

print("Les réalisateurs avec le plus grand nombre de films :")
directors_movies_count.show()

# Calcul du pourcentage de films/séries produits par pays
countries_df = netflix_data.filter(col("country").isNotNull()).withColumn(
    "country", explode(split(col("country"), ", "))
)

country_production_count = countries_df.groupBy("country").agg(count("show_id").alias("total_count"))

total_productions = netflix_data.filter(col("country").isNotNull()).count()
country_production_percentage = country_production_count.withColumn(
    "percentage", (col("total_count") / total_productions) * 100
).orderBy(col("percentage").desc())

print("Pourcentages des productions par pays :")
country_production_percentage.show()

# Calcul de la durée moyenne des films et identification du film le plus long et le plus court
film_duration_df = films_df.filter(col("duration").isNotNull()).withColumn(
    "duration_minutes", regexp_extract(col("duration"), r"(\d+)", 1).cast("int")
)

average_film_duration = film_duration_df.select(avg("duration_minutes")).collect()[0][0]

longest_film = (
    film_duration_df.orderBy(col("duration_minutes").desc())
    .select("title", "duration_minutes")
    .first()
)
shortest_film = (
    film_duration_df.orderBy(col("duration_minutes").asc())
    .select("title", "duration_minutes")
    .first()
)

print("Durée moyenne des films :", average_film_duration)
print("Film le plus long :", longest_film)
print("Film le plus court :", shortest_film)

# Durée moyenne des films par intervalles de 2 ans
films_by_year_df = film_duration_df.filter(col("release_year").isNotNull()).withColumn(
    "year_interval", (col("release_year") - (col("release_year") % 2)).cast("int")
)

avg_duration_by_interval = (
    films_by_year_df.groupBy("year_interval")
    .agg(avg("duration_minutes").alias("average_duration"))
    .orderBy(col("year_interval").desc())
)

print("Durée moyenne des films par intervalles de 2 ans :")
avg_duration_by_interval.show()

actors_df = films_df.filter(
    col("cast").isNotNull() & col("director").isNotNull()
).withColumn("actor", explode(split(col("cast"), ", ")))

director_actor_movies_count = (
    actors_df.groupBy("director", "actor")
    .agg(count("title").alias("total_movies"))
    .orderBy(col("total_movies").desc())
)

print("Le duo réalisateur-acteur le plus prolifique :")
director_actor_movies_count.show()

spark.stop()
