##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if (!require(tidyverse))
  install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if (!require(caret))
  install.packages("caret", repos = "http://cran.us.r-project.org")
if (!require(data.table))
  install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)


# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip",
              dl)

ratings <-
  fread(
    text = gsub("::", "\t", readLines(unzip(
      dl, "ml-10M100K/ratings.dat"
    ))),
    col.names = c("userId", "movieId", "rating", "timestamp")
  )

movies <-
  str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
# movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                            title = as.character(title),
#                                            genres = as.character(genres))
# if using R 4.0 or later:
movies <-
  as.data.frame(movies) %>% mutate(
    movieId = as.numeric(movieId),
    title = as.character(title),
    genres = as.character(genres)
  )


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind = "Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <-
  createDataPartition(
    y = movielens$rating,
    times = 1,
    p = 0.1,
    list = FALSE
  )
edx <- movielens[-test_index, ]
temp <- movielens[test_index, ]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>%
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#########################################################
# From this point on, the codes are authorized by Yi Wang
#########################################################

#load("edx.rda")
#load("validation.rda")
#Temporarily suppress warnings
defaultW <- getOption("warn")
options(warn = -1)

# load the library lubridate to use the function as_datetime
library(lubridate)

################################################################################
#Data Wraggling: Add a column of "hour" when a rating is given/recommended into 
#the edx set and the validation set
################################################################################
train_set_0 <- edx %>% mutate(hour = hour(as_datetime(timestamp)))
validation <-
  validation %>% mutate(hour = hour(as_datetime(timestamp)))

#########################################################################
#This section explore the effects of genres and hours when a rating is given. 

#########################################################################

# Explore the effect of genres
mu <- mean(train_set_0$rating) 
genres_avg <- train_set_0 %>% 
  group_by(genres) %>% 
  summarize(b_g = mean(rating-mu))
qplot(b_g, data = genres_avg, bins = 10, color = I("black"))

# Explore the effect of hours
mu <- mean(train_set_0$rating) 
hours_avg <- train_set_0 %>% 
  group_by(hour) %>% 
  summarize(b_t = mean(rating-mu))
qplot(b_t, data = hours_avg, bins = 10, color = I("black"))

# Explore the effect of rate of ratings for movies
train_set_0 %>% mutate( date = as_datetime(timestamp)) %>%
  mutate(year=year(date)) %>% group_by(movieId) %>%
  summarize(n = n(), years = max(year) - min(year)+1,
            rating = mean(rating)) %>%
  mutate(rate = n/years) %>%
  ggplot(aes(rate, rating)) +
  geom_point() +
  geom_smooth()

#Define the RMSE function
RMSE <- function(true_ratings, predicted_ratings) {
  sqrt(mean((true_ratings - predicted_ratings) ^ 2))
}

# Define the out-of-range correction
OFR_corr <- function(pred){
  pred[pred>5] <- 5
  pred[pred<0.5] <- 0.5
  return(pred)
}

###############################################################
#Define the proposed training model  as a R function
# function arguments
#train_set: the training set
#test_set: the test set
#l: the regulation parameter, defaulted to be 5
# Genre, Hour, Rate, OFR: switches to determine if an effect will be considered. 
#For simplicity, the user and movie effects are always included. 
# Each additional effect will be assumed to be added incrementally only 
# for the purpose of this project  for the reason of simplicity. 
###############################################################

train_test_rmse <- function(train_set, test_set, l = 5, 
                            Genre=TRUE, Hour=TRUE, Rate=TRUE, OFR=TRUE) {
  # average rating of all movies by all users
  mu <- mean(train_set$rating)
  
  # Effect of  movies
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu) / (n() + l))
  #Effect of users
  b_u <- train_set %>%
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu) / (n() + l))
  
  predicted_ratings <-
    test_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
        mutate(pred = mu + b_i + b_u  ) %>%
    .$pred
 
  #Assess Effect of genres
  if (Genre == TRUE) {
  b_g <- train_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_i - b_u - mu) / (n() + l))
  
  predicted_ratings <-
    test_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
        mutate(pred = mu + b_i + b_u + b_g  ) %>%
    .$pred
}
  
  #Assess Effect of timing when a rating is given
  if (Genre == TRUE & Hour == TRUE) {
  b_t <- train_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    group_by(hour) %>%
    summarize(b_t = sum(rating - b_i - b_u - b_g - mu) / (n() + l))
  predicted_ratings <-
    test_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    left_join(b_t, by = "hour") %>%
    mutate(pred = mu + b_i + b_u + b_g + b_t ) %>%
    .$pred
  
  }
  
  # Assess the effect of the rate of ratings
  if ( Genre == TRUE & Hour == TRUE & Rate == TRUE) {
  b_r <-train_set %>% mutate( date = as_datetime(timestamp)) %>%
    mutate(year=year(date)) %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    left_join(b_t, by = "hour") %>%
    group_by(movieId) %>%
    summarize(n = n(), years = max(year) - min(year)+1,
              rating1 = sum(rating- mu - b_i-b_u-b_g-b_t)/(n+l) ) %>%
    mutate(rate = n/years) %>%
    mutate(rating_delta=predict(lm(rating1~rate))) %>% 
    select(movieId, rating_delta)
  
  
  # Make prediction and validate it on the test_set
  predicted_ratings <-
    test_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    left_join(b_t, by = "hour") %>%
    left_join(b_r, by = "movieId") %>%
    mutate(pred = mu + b_i + b_u + b_g + b_t+ rating_delta ) %>%
    .$pred
  }
  
 # Out-of-range-correction
  if (OFR==TRUE) {
  predicted_ratings <- OFR_corr(predicted_ratings)
  }
  
  #Compute RMSE 
  return(RMSE(predicted_ratings, test_set$rating))
  
}


################################################################
# Define Cross validation using bootstrap strategy for training as a R function
################################################################

# First define the training and validation function
train_val_rmses <-  function(train_set_0, lambdas=5,fold=1, pct=0.1){
  sapply(lambdas, function(l) {
    # set the regulation parameter l for cross-validation
    
    rmses_k <- sapply(1:fold, function(k) {
      # for each fold k
      set.seed(k, sample.kind = "Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
      # split the entire edx set randomly into train_set (90%) and test_set (10%)
      # Note the edx set is renamed as train_set_0 with added column "hour"
      test_index <-
        createDataPartition(
          y = train_set_0$rating,
          times = 1,
          p = pct,
          list = FALSE
        )
      train_set <- train_set_0[-test_index,]
      temp <- train_set_0[test_index,]
      
      # Make sure userId, movieId, genres, hour  in test_set are also in train_set
      test_set <- temp %>%
        semi_join(train_set, by = "movieId") %>%
        semi_join(train_set, by = "userId") %>%
        semi_join(train_set, by = "genres") %>%
        semi_join(train_set, by = "hour")
      
      # Add rows removed from test_set back into train_set
      removed <- anti_join(temp, test_set)
      train_set <- rbind(train_set, removed)
      
      #remove the unused objects from the workspace
      rm(test_index, temp,  removed)
      
      # Now start training
      train_test_rmse(train_set, test_set, l)
      
    }) # end of the k-loop
    
    # return the average rmse wit the regulaton parameter l of the k-th fold validaiton
    return(mean(rmses_k))
  })  # end of the l-loop
} # end of the train_val_rmses function


# regulation parameters grid
lambdas <-  seq(0, 10, 0.25)
# number of folds for the cross-validation
K <- 5

################################################################
# Perform cross validation using bootstrap strategy for training 
################################################################

rmses <- train_val_rmses(train_set_0, lambdas,K,pct=0.1)

#visualize the effect of regulation to the RMSE
qplot(lambdas, rmses)

best_lambda <- lambdas[which.min(rmses)]

sprintf("The best lambda = %f", best_lambda)

sprintf("The best RMSE of the predicion  on the training set by our model: %f", 
min(rmses) )

################################################################################### 
# Now use the best regulation parameter to train the entire edx set which is renamed
#as train_set_0 with the added column 'hour' as required by our model, and then test
# on the final validation set that is not seen by the training program. We also compare 
# the different models by adding each additional effects incrementally. 
###################################################################################

sprintf("Prediction RMSE on the validation set: user+movie: %f", 
        train_test_rmse(train_set_0, validation, best_lambda, 
                        Genre=FALSE, Hour=FALSE, Rate=FALSE, OFR=FALSE) )

sprintf("Prediction RMSE on the validation set: user+movie+genres: %f", 
        train_test_rmse(train_set_0, validation, best_lambda, 
                        Genre=TRUE, Hour=FALSE, Rate=FALSE, OFR=FALSE) )

sprintf("Prediction RMSE on the validation set: user+movie+genres+hour: %f", 
        train_test_rmse(train_set_0, validation, best_lambda, 
                        Genre=TRUE, Hour=TRUE, Rate=FALSE, OFR=FALSE) )

sprintf("Prediction RMSE on the validation set: user+movie+genres+hour+rate: %f", 
        train_test_rmse(train_set_0, validation, best_lambda, 
                        Genre=TRUE, Hour=TRUE, Rate=TRUE, OFR=FALSE) )

sprintf("Prediction RMSE on the validation set: user+movie+genres+hour+rate+OFR: %f", 
        train_test_rmse(train_set_0, validation, best_lambda, 
                        Genre=TRUE, Hour=TRUE, Rate=TRUE, OFR=TRUE) )

#Restore the original warning option
options(warn = defaultW)


