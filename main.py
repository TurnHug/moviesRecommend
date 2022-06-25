import sys
import os
import math
from pyspark.mllib.recommendation import ALS
from pyspark import SparkContext

# 读取数据
from test_helper.test_helper import Test

baseDir = os.path.join('')
inputPath = os.path.join('input')
ratingsFilename = os.path.join(baseDir, inputPath, 'ratings.txt')
moviesFilename = os.path.join(baseDir, inputPath, 'movies.txt')
sc = SparkContext.getOrCreate()
numPartitions = 2
rawRatings = sc.textFile(ratingsFilename).repartition(numPartitions)
rawMovies = sc.textFile(moviesFilename)


def get_ratings_tuple(entry):
    """ 评分数据集(ratings.dat)中的每一行格式如下：
    参数：
        entry （str）：评级数据集中的一行，格式为 UserID：：MovieID：：Rating：：Timestamp
    返回：
        元组：（用户 ID、影片 ID、评级）
    """
    items = entry.split('::')
    return int(items[0]), int(items[1]), float(items[2])


def get_movie_tuple(entry):
    """
        电影数据集(movies.dat)中的每一行格式为：
        参数：
        entry （str）：电影数据集中的一行，格式为 MovieID：：Title：：Genres(电影类型)
    返回：
        元组：（影片 ID、标题）
    """

    items = entry.split('::')
    return int(items[0]), items[1]


# 解析这两个文件生成两个RDD
ratingsRDD = rawRatings.map(get_ratings_tuple).cache()
moviesRDD = rawMovies.map(get_movie_tuple).cache()

ratingsCount = ratingsRDD.count()
moviesCount = moviesRDD.count()

print('数据集中有%d个电影评分和%d个电影' % (ratingsCount, moviesCount))
print('评分集的前三个元素为: %s' % ratingsRDD.take(3))
print('电影集的前三个元素为: %s' % moviesRDD.take(3))


# tmp1 = [(1, u'alpha'), (2, u'alpha'), (2, u'beta'), (3, u'alpha'), (1, u'epsilon'), (1, u'delta')]
# tmp2 = [(1, u'delta'), (2, u'alpha'), (2, u'beta'), (3, u'alpha'), (1, u'epsilon'), (1, u'alpha')]
#
# oneRDD = sc.parallelize(tmp1)
# twoRDD = sc.parallelize(tmp2)
# oneSorted = oneRDD.sortByKey(True).collect()
# twoSorted = twoRDD.sortByKey(True).collect()
# print(oneSorted)
# print(twoSorted)
# assert set(oneSorted) == set(twoSorted)  # Note that both lists have the same elements
# assert twoSorted[0][0] < twoSorted.pop()[0]  # Check that it is sorted by the keys
# assert oneSorted[0:2] != twoSorted[0:2]  # Note that the subset consisting of the first two elements does not match


def sortFunction(tuple):
    """ Construct the sort string (does not perform actual sorting)
    Args:
        tuple: (rating, MovieName)
    Returns:
        sortString: the value to sort with, 'rating MovieName'
    """
    key = str('%.3f' % tuple[0])
    value = tuple[1]
    return key + ' ' + value


#
#
# print(oneRDD.sortBy(sortFunction, True).collect())
# print(twoRDD.sortBy(sortFunction, True).collect())
# oneSorted1 = oneRDD.takeOrdered(oneRDD.count(), key=sortFunction)
# twoSorted1 = twoRDD.takeOrdered(twoRDD.count(), key=sortFunction)
# print('one is %s' % oneSorted1)
# print('two is %s' % twoSorted1)
# assert oneSorted1 == twoSorted1
#

# TODO: Replace <FILL IN> with appropriate code
# 得到电影平均评分
# First, implement a helper function `getCountsAndAverages` using only Python
def getCountsAndAverages(IDandRatingsTuple):
    """
        计算平均评分
    参数：
        IDandRatingsTuple(电影Id和电影评分)：一个元组（MovieID，（Rating1，Rating2，Rating3，...））
    返回：
        元组：（MovieID，（电影评分的个数，平均评分））的元组
    """
    movie = IDandRatingsTuple[0]
    ratings = IDandRatingsTuple[1]
    return movie, (len(ratings), float(sum(ratings)) / len(ratings))


# TODO: Replace <FILL IN> with appropriate code

# From ratingsRDD with tuples of (UserID, MovieID, Rating) create an RDD with tuples of
# the (MovieID, iterable of Ratings for that MovieID)
# 转换RDD “ratingsRDD”中包含（UserID、MovieID、Rating）的元组。基于“ratingsRDD”创建一个具有以下形式元组的RDD（MovieID，该MovieID评分的Python 迭代）。
movieIDsWithRatingsRDD = (ratingsRDD
                          .map(lambda x: (x[1], x[2]))
                          .groupByKey())
print('movieIDsWithRatingsRDD: %s\n' % movieIDsWithRatingsRDD.take(3))

movieIDsWithAvgRatingsRDD = movieIDsWithRatingsRDD.map(lambda rec: getCountsAndAverages(rec))
print('movieIDsWithAvgRatingsRDD: %s\n' % movieIDsWithAvgRatingsRDD.take(3))

# 对“moviesRDD”，使用“movieIDsWithAvgRatingsRDD”的RDD转换来获取“movieIDsWithAvgRatingsRDD”的电影名称，
# 生成以下形式的元组（平均评分、电影名称、评分数）。
movieNameWithAvgRatingsRDD = (moviesRDD
                              .join(movieIDsWithAvgRatingsRDD)
                              .map(lambda movie: (movie[1][1][1], movie[1][0], movie[1][1][0])))
print('movieNameWithAvgRatingsRDD: %s\n' % movieNameWithAvgRatingsRDD.take(3))

# 平均评分最高、评论超过500条的电影
print('以下是评分最高且评论超过500条的电影：')
movieLimitedAndSortedByRatingRDD = (movieNameWithAvgRatingsRDD
                                    .filter(lambda movie: movie[2] > 500)
                                    .sortBy(sortFunction, False))
print('电影评分最高的20个: %s' % movieLimitedAndSortedByRatingRDD.take(20))

# TODO: Replace <FILL IN> with appropriate code

# Apply an RDD transformation to `movieNameWithAvgRatingsRDD` to limit the results to movies with
# ratings from more than 500 people. We then use the `sortFunction()` helper function to sort by the
# average rating to get the movies in order of their rating (highest rating first)
# 训练集（RDD）我们将用它来训练模型
#
# 验证集（RDD）我们将使用它来选择最佳模型
#
# 测试集（RDD）我们将使用它来测试模型
trainingRDD, validationRDD, testRDD = ratingsRDD.randomSplit([6, 2, 2], seed=2)

print('Training:(训练集) %s, validation:(验证集) %s, test:(测试集) %s\n' % (trainingRDD.count(),
                                                                   validationRDD.count(),
                                                                   testRDD.count()))
print(trainingRDD.take(3))
print(validationRDD.take(3))
print(testRDD.take(3))
print(validationRDD.count())

print('\n')


# 均方根误差RSME

def computeError(predictedRDD, actualRDD):
    """
        计算预测值和实际值之间的均方根误差
    参数：
        predictedRDD：每部电影和每个用户的预测评级，其中每个条目都采用
                      （用户 ID、影片 ID、评级）
        actualRDD：实际评级，其中每个条目都采用以下形式（用户ID，MovieID，评级）
    返回：
        RSME（浮点数）：计算的 RSME 值
        将“predictedRDD”转换为以下形式的元组（（UserID，MovieID），Rating）。
        例如，[((1, 1), 5), ((1, 2), 3), ((1, 3), 4), ((2, 1), 3), ((2, 2), 2), ((2, 3), 4)]。
        将“actualRDD”转换为以下形式的元组（（UserID，MovieID），Rating）。
        例如，像[((1, 2), 3), ((1, 3), 5), ((2, 1), 5), ((2, 2), 1)]这样的元组。
    """
    predictedReformattedRDD = predictedRDD.map(lambda movie: ((movie[0], movie[1]), movie[2]))
    actualReformattedRDD = actualRDD.map(lambda movie: ((movie[0], movie[1]), movie[2]))
    squaredErrorsRDD = (predictedReformattedRDD
                        .join(actualReformattedRDD)
                        .map(lambda pre: (pre[1][0] - pre[1][1]) ** 2))
    totalError = squaredErrorsRDD.reduce(lambda a, b: a + b)
    numRatings = squaredErrorsRDD.count()
    return math.sqrt(float(totalError) / numRatings)


testPredicted = sc.parallelize([
    (1, 1, 5),
    (1, 2, 3),
    (1, 3, 4),
    (2, 1, 3),
    (2, 2, 2),
    (2, 3, 4)])
testActual = sc.parallelize([
    (1, 2, 3),
    (1, 3, 5),
    (2, 1, 5),
    (2, 2, 1)])
testPredicted2 = sc.parallelize([
    (2, 2, 5),
    (1, 2, 5)])

testError = computeError(testPredicted, testActual)
print('Error for test dataset (should be 1.22474487139): %s' % testError)
testError2 = computeError(testPredicted2, testActual)
print('Error for test dataset2 (should be 3.16227766017): %s' % testError2)
testError3 = computeError(testActual, testActual)
print('Error for testActual dataset (should be 0.0): %s' % testError3)

# 测试
Test.assertTrue(abs(testError - 1.22474487139) < 0.00000001,
                'incorrect testError (expected 1.22474487139)')
Test.assertTrue(abs(testError2 - 3.16227766017) < 0.00000001,
                'incorrect testError2 result (expected 3.16227766017)')
Test.assertTrue(abs(testError3 - 0.0) < 0.00000001,
                'incorrect testActual result (expected 0.0)')

# TODO: Replace <FILL IN> with appropriate code

# 建一个输入RDD“validationForPredictRDD”，它由从“validationRDD”提取的（UserID，MovieID）元组组成。
validationForPredictRDD = validationRDD.map(lambda movie: (movie[0], movie[1]))

seed = 2
iterations = 5
regularizationParameter = 0.1
ranks = [4, 8, 12]
errors = [0, 0, 0]
err = 0
tolerance = 0.1

minError = float('inf')
bestRank = -1
bestIteration = -1
# 使用ALS.train(trainingRDD，rank，seed=seed，iterations=iterations，lambda ium=regulationparameter)函数创建模型有三个参数：
# 由用于训练模型的元组（UserID、MovieID、rating）组成的RDD，一个整数秩（4、8或12），要执行的迭代次数（将“iterations”参数设置为5），
# 以及正则化系数（将“regulationparameter”设置为0.1）。
for rank in ranks:
    model = ALS.train(trainingRDD, rank, seed=seed, iterations=iterations,
                      lambda_=regularizationParameter)
    predictedRatingsRDD = model.predictAll(validationForPredictRDD)
    error = computeError(predictedRatingsRDD, validationRDD)
    errors[err] = error
    err += 1
    print('For rank %s the RMSE is %s' % (rank, error))
    if error < minError:
        minError = error
        bestRank = rank

print('The best model was trained with rank %s' % bestRank)

# 测试
Test.assertEquals(trainingRDD.getNumPartitions(), 2,
                  'incorrect number of partitions for trainingRDD (expected 2)')
Test.assertEquals(validationForPredictRDD.count(), 1189844,
                  'incorrect size for validationForPredictRDD (expected 1189844)')
Test.assertEquals(validationForPredictRDD.filter(lambda t: t == (1, 1907)).count(), 0,
                  'incorrect content for validationForPredictRDD')
Test.assertTrue(abs(errors[0] - 0.883710109497) < tolerance, 'incorrect errors[0]')
Test.assertTrue(abs(errors[1] - 0.878486305621) < tolerance, 'incorrect errors[1]')
Test.assertTrue(abs(errors[2] - 0.876832795659) < tolerance, 'incorrect errors[2]')

# TODO: Replace <FILL IN> with appropriate code
myModel = ALS.train(trainingRDD, bestRank, seed=seed, iterations=iterations, lambda_=regularizationParameter)
testForPredictingRDD = testRDD.map(lambda movie: (movie[0], movie[1]))
predictedTestRDD = myModel.predictAll(testForPredictingRDD)
testRMSE = computeError(testRDD, predictedTestRDD)
print('The model had a RMSE on the test set of %s' % testRMSE)
Test.assertTrue(abs(testRMSE - 0.87809838344) < tolerance, 'incorrect testRMSE')
# TODO: Replace <FILL IN> with appropriate code
# 使用“trainingRDD”计算训练数据集中所有电影的平均评分。
# 使用刚刚得出的平均评分和“testRDD”创建一个包含（userID、movieID、average rating）的RDD。
# 使用“computeError”函数计算刚创建的“testRDD”，验证RDD和“testForAvgRDD”之间的RMSE。
trainingAvgRating = trainingRDD.map(lambda x: x[2]).mean()
print('训练集中电影的平均评分为:%s' % trainingAvgRating)
testForAvgRDD = testRDD.map(lambda x: (x[0], x[1], trainingAvgRating))
testAvgRMSE = computeError(testRDD, testForAvgRDD)
print('平均集上的 RMSE 为 ：%s' % testAvgRMSE)

print('评分最高的电影：')
print('(平均评分,电影名称,评论数量)')
for ratingsTuple in movieLimitedAndSortedByRatingRDD.take(50):
    print(ratingsTuple)


myUserID = 0
myRatedMovies = [
    (myUserID, 1088, 5),
    (myUserID, 1195, 5),
    (myUserID, 1110, 4),
    (myUserID, 1250, 5),
    (myUserID, 1775, 4),
    (myUserID, 789, 5),
    (myUserID, 1039, 4),
    (myUserID, 811, 5),
    (myUserID, 1447, 4),
    (myUserID, 1438, 4)
]
myRatingsRDD = sc.parallelize(myRatedMovies)
print('我的电影评分: %s' % myRatingsRDD.take(10))

# Note that the movie IDs are the *last* number on each line. A common error was to use the number of ratings as the
# movie ID.
trainingWithMyRatingsRDD = trainingRDD.union(myRatingsRDD)

print('训练数据集现在比原始训练数据集具有多 %s 个条目' %
      (trainingWithMyRatingsRDD.count() - trainingRDD.count()))
assert (trainingWithMyRatingsRDD.count() - trainingRDD.count()) == myRatingsRDD.count()

# TODO: Replace <FILL IN> with appropriate code


myRatingsModel = ALS.train(trainingWithMyRatingsRDD, bestRank, seed=seed, iterations=iterations,
                           lambda_=regularizationParameter)

# TODO: Replace <FILL IN> with appropriate code
predictedTestMyRatingsRDD = myRatingsModel.predictAll(testForPredictingRDD)
testRMSEMyRatings = computeError(testRDD, predictedTestMyRatingsRDD)
print('该模型在测试集上的 RMSE为: %s' % testRMSEMyRatings)
# TODO: Replace <FILL IN> with appropriate code
myUnratedMoviesRDD = (moviesRDD.map(lambda movie_title: (myUserID, movie_title[0])).filter(
    lambda user_movie: (user_movie[0], user_movie[1]) not in [(u, m) for (u, m, r) in myRatedMovies]))
predictedRatingsRDD = myRatingsModel.predictAll(myUnratedMoviesRDD)

# TODO: Replace <FILL IN> with appropriate code

# Use the Python list myRatedMovies to transform the moviesRDD into an RDD with entries that are pairs of the form (
# myUserID, Movie ID) and that does not contain any movies that you have rated.
movieCountsRDD = movieIDsWithAvgRatingsRDD.map(lambda movie_num_avg: (movie_num_avg[0], movie_num_avg[1][0]))

predictedRDD = predictedRatingsRDD.map(lambda x: (x[1], x[2]))

predictedWithCountsRDD = predictedRDD.join(movieCountsRDD)
ratingsWithNamesRDD = (predictedWithCountsRDD
                       .join(moviesRDD)
                       .map(lambda movie_rating_num_name: (movie_rating_num_name[1][1][0], movie_rating_num_name[1][1],
                                                           movie_rating_num_name[1][0][1]))
                       .filter(lambda rating_name_num: rating_name_num[2] > 75))
print(ratingsWithNamesRDD)

predictedHighestRatedMovies = ratingsWithNamesRDD.takeOrdered(20, key=lambda x: x[0])
print('预测的我最高评分的电影 (对于评论超过75的电影):\n%s' %'\n'.join(map(str, predictedHighestRatedMovies)))

