from pyspark.sql import SparkSession
from pyspark.sql.functions import col, format_number, concat, lit, avg, count, desc, sum as F_sum, round as spark_round, when, create_map, expr
from functools import reduce
from itertools import chain
from operator import itemgetter

spark = SparkSession.builder.appName("AirbnbDataAnalysis").getOrCreate()

listings_df = spark.read.csv("listings.csv", header=True, inferSchema=True, multiLine=True, escape="\"")

total_listings = listings_df.count()

room_type_distribution = listings_df.groupBy('room_type').count()
room_type_distribution = room_type_distribution.withColumn('percentage', (col('count') / total_listings) * 100)
room_type_distribution = room_type_distribution.withColumn('percentage', format_number('percentage', 1))
room_type_distribution = room_type_distribution.withColumn('percentage', concat(col('percentage'), lit('%')))
room_type_distribution.show()

listings_df = listings_df.withColumn("price", col("price").cast("string").substr(2, 100).cast("float"))

avg_nights_booked = listings_df.agg(spark_round(avg(col("reviews_per_month") * col("minimum_nights") * 12), 0)).first()[0]
avg_price_per_night = listings_df.agg(spark_round(avg("price"), 0)).first()[0]
avg_income = listings_df.withColumn("estimated_income", col("reviews_per_month") * col("minimum_nights") * col("price") * 12) \
                        .agg(spark_round(avg("estimated_income"), 0)).first()[0]

print("\n")
print(f"Average nights booked per year: {avg_nights_booked}")
print(f"Average price per night: £{avg_price_per_night}")
print(f"Average estimated annual income: £{avg_income}")
print("\n")

listings_df = listings_df.withColumn("total_booked_nights", col("reviews_per_month") * col("minimum_nights") * 12)

bins = [0, 30, 60, 90, 120, 150, 180, 210, 240, 255]
labels = ["0-30", "31-60", "61-90", "91-120", "121-150", "151-180", "181-210", "211-240", "241-255+"]

label_order = {label: index for index, label in enumerate(labels)}

binned_data = listings_df.withColumn("occupancy_bin", 
    when(col("total_booked_nights") <= 30, labels[0])
    .when((col("total_booked_nights") > 30) & (col("total_booked_nights") <= 60), labels[1])
    .when((col("total_booked_nights") > 60) & (col("total_booked_nights") <= 90), labels[2])
    .when((col("total_booked_nights") > 90) & (col("total_booked_nights") <= 120), labels[3])
    .when((col("total_booked_nights") > 120) & (col("total_booked_nights") <= 150), labels[4])
    .when((col("total_booked_nights") > 150) & (col("total_booked_nights") <= 180), labels[5])
    .when((col("total_booked_nights") > 180) & (col("total_booked_nights") <= 210), labels[6])
    .when((col("total_booked_nights") > 210) & (col("total_booked_nights") <= 240), labels[7])
    .when(col("total_booked_nights") > 240, labels[8])
    .otherwise("Unknown"))

mapping_expr = create_map([lit(x) for x in chain(*label_order.items())])
binned_data = binned_data.withColumn("sort_order", mapping_expr[col("occupancy_bin")])

binned_data = binned_data.sort("sort_order")

occupancy_counts = binned_data.groupBy("occupancy_bin").count()
occupancy_counts.show()

short_term_listings = listings_df.filter(col("minimum_nights") <= 30).count()
long_term_listings = listings_df.filter(col("minimum_nights") > 30).count()

percentage_short_term = round((short_term_listings / total_listings) * 100, 2)
percentage_long_term = round((long_term_listings / total_listings) * 100, 2)

print(f"Short-term rentals: {short_term_listings} ({percentage_short_term}%)")
print(f"Long-term rentals: {long_term_listings} ({percentage_long_term}%)")
print("\n")

total_hosts = listings_df.select("host_id").distinct().count()
single_listings = listings_df.filter(col("calculated_host_listings_count") == 1).select("host_id").distinct().count()
multi_listings = total_hosts - single_listings

percentage_single_listings = round((single_listings / total_hosts) * 100, 2)
percentage_multi_listings = round((multi_listings / total_hosts) * 100, 2)

print(f"Single listings: {single_listings} ({percentage_single_listings}%)")
print(f"Multi-listings: {multi_listings} ({percentage_multi_listings}%)")

labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10+"]
listing_counts = [listings_df.filter(col("calculated_host_listings_count") == i).count() for i in range(1, 10)]
listing_counts.append(listings_df.filter(col("calculated_host_listings_count") >= 10).count())

print("\n")

top_hosts = listings_df.groupBy("host_name", "room_type").agg(count("room_type").alias("count"))
top_hosts = top_hosts.groupBy("host_name").pivot("room_type").sum("count")
top_hosts = top_hosts.withColumnRenamed("Entire home/apt", "#Entire home/apts") \
                     .withColumnRenamed("Private room", "#Private rooms") \
                     .withColumnRenamed("Shared room", "#Shared rooms") \
                     .withColumnRenamed("Hotel room", "#Hotel Rooms")
top_hosts = top_hosts.fillna(0)

columns_to_sum = [col(c) for c in top_hosts.columns if c != "host_name"]
top_hosts = top_hosts.withColumn("#Listings", reduce(lambda x, y: x + y, columns_to_sum))
top_hosts = top_hosts.sort(desc("#Listings"))

top_hosts.show()
spark.stop()
