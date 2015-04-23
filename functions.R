# Data Mining
# ID: 1102930

# this function assigns tags to the documents and duplicates entries that have multiple tags
# assigning each new entry one of the tags
tagsfind <- function(data) {    
  data <- data[ rowSums(data[,(4:138)])!=0, ] #first remove all entries without any tags
  data <- data[ !(data$purpose == "not-used"), ]  #remove those in "not-used" category
  data <- data[ !(data$doc.text==""), ]       #remove any documents without any doc.text
  
  possTags <- apply(data[,4:138],1,function(x) which(x==1))	# creates a list of all possible tags per document
  test <- 1
  count <- 100
  
  for(n in 1:nrow(data) ){			# iterates through all rows in the dataset
    for (i in 1:length(possTags[[n]])){			# and through all possible tags
        datanew <- data[n, c(1,3,140)]			# for each possible tag a new row is created
        datanew$tags <- names(possTags[[n]][i])	# with that tag, and they are rbind() together
		
		if (test==1){			# in the first iteration, there is no need to rbind
			result <- datanew
			test <- 0
		}
		else{
			result <- rbind(result, datanew )	# in all other iterations, the duplicated row is added to the result
		}
	}
    if (n==count) {		# a short if statement to print progress to console
      cat("Row",n," out of ",nrow(data)," total completed\n")
      count <- count+ 100
    } 
  }
  result$tags <- gsub('^.{6}', '', result$tags)	# removed "topic." characters from the tag label
  cat("Tag allocation complete")
  return(result)		# returns a new tagged dataset
}


# this function performs Naive Bayes, SVM and Random Forest on the dataset in order to predict
# the tags label using the train/test split. This function returns the performance statistics of the classifiers
testtrain <- function(b){
	# first it creates the train and test datasets
  train <- b[(b$purpose=="train"),]
  test <- b[(b$purpose=="test"),]
  
  # the classifiers are modelled on train and tested on test
  # note the first column (purpose) is always removed, the second (tags)
  # is only removed for prediction
  NB <- naiveBayes(as.factor(tags)~.,train[, -1])
  NBP <- predict(NB, newdata=test[,-c(1,2)])
  cat("Naive Bayes complete\n")
  
  SVM <- svm(tags~.,train[,-1], kernel="linear",cost=1)
  SVMP <- predict(SVM, newdata= test[,-c(1,2)])
  cat("SVM complete\n")  
  
  RF <- randomForest(tags~., train[,-c(1)])
  RFP <- predict(RF,newdata= test[,-c(1,2)])
  cat("Random Forest complete\n")
  
  # statistics are generated for each classifier from the results of the models, using morestats()
  confusionNB <- confusionMatrix(NBP, test$tags)
  statsNB <- 	morestats(confusionNB)
  confusionSVM <- confusionMatrix(SVMP, test$tags)
  statsSVM <- 	morestats(confusionSVM)
  confusionRF <- confusionMatrix(RFP, test$tags)
  statsRF <- 	morestats(confusionRF)
  
  # the statistics are combined in a list and each element is named
  results <- list(confusionNB, statsNB, confusionSVM, statsSVM, confusionRF, statsRF)
  names(results) <- c("Naive Bayes","Naive Bayes2",
                       "SVM", "SVM2", "Random Forest", "Random Forest2")
					   
  # the list of statistics is returned
  return(results)
}

# this function normalises the input to between 0 and 1
normalise <- function(x) {
  (x - min(x, na.rm=TRUE))/(max(x,na.rm=TRUE) - min(x, na.rm=TRUE))
}


# this function performs Naive Bayes, SVM and Random Forest on the dataset in order to predict
# the tags label using k-fold cross validation, with k=10 here.
# This function returns the performance statistics of the classifiers
crossfold <- function(data) {
  data <- data[sample(1:nrow(data)), ] #randomises the data
  for (i in 2:ncol(data)){
    data[,i]<-normalise(data[,i])     #normalises each column
  }
  
  # this creates a sequence to be used for finding the folds
  c <- floor(nrow(data)/10)
  v <- seq(from=1, by=c, length=10)
  v <- append(v,nrow(data)+1)
  # the sequence is 11 elements long from 1 to nrow(data) +1 and the gaps between the numbers
  # corresponds to the rows in each fold.
  # note: the first number of each gap is also included in that fold
  # i.e. if the sequence was 1,5,9,13,...
  # the first fold is rows 1 to 4 inclusive, the second fold is row 5 to 8 inclusive
  
  # initialises vectors to hold the predicted values
  predictedNB <- vector()
  predictedSVM <- vector()
  predictedRF <- vector()
  
  #now iterates through each fold
  for (i in 1:10) {

	# each classifier is trained on the dataset without a fold, and then tested on that fold
    NB <- naiveBayes(tags ~ .,data[-(v[i]:(v[i+1]-1)),])
    NBP <- predict(NB, newdata=data[(v[i]:(v[i+1]-1)),-1])
    cat("Naive Bayes complete for fold",i,"\n")
	
    svm <- svm(tags~.,data[-(v[i]:(v[i+1]-1)),], kernel="linear", cost=1)
    SVMP <- predict(svm, newdata=data[(v[i]:(v[i+1]-1)),-1])
    cat("SVM complete for fold",i,"\n")
	
    RF <- randomForest(tags~.,data[-(v[i]:(v[i+1]-1)),])
    RFP <- predict(RF,newdata=data[(v[i]:(v[i+1]-1)),-1])
	cat("Random Forest complete for fold",i,"\n")
	
	# the predictions from each classifier are stored before the start of a new iteration
    predictedNB <- append(predictedNB,NBP)
    predictedNB <- factor(predictedNB, levels=1:nlevels(NBP), labels=levels(NBP))
    predictedSVM <- append(predictedSVM,SVMP)
    predictedSVM <- factor(predictedSVM, levels=1:nlevels(SVMP), labels=levels(SVMP))
    predictedRF <- append(predictedRF,RFP)
    predictedRF <- factor(predictedRF, levels=1:nlevels(RFP), labels=levels(RFP))
    cat("Cross-fold number",i,"completed\n\n")
  }
  
  # statistics are generated for each classifier from the results of the models, using morestats()
  confusionNB <- confusionMatrix(predictedNB, data[,1])
  statsNB <- 	morestats(confusionNB)
  confusionSVM <- confusionMatrix(predictedSVM, data[,1])
  statsSVM <- 	morestats(confusionSVM)
  confusionRF <- confusionMatrix(predictedRF, data[,1])
  statsRF <- 	morestats(confusionRF)
  
  # the statistics are combined in a list and each element is named  
  results <- list(confusionNB, statsNB, confusionSVM, statsSVM, confusionRF, statsRF)
  names(results) <- c("Naive Bayes","Naive Bayes2",
                       "SVM", "SVM2", "Random Forest", "Random Forest2")
  # the list of statistics is returned
  return(results)
}

# this function creates some additional statistics from a confusionMatrix input
morestats <- function(confus) {
  fp <- 1- confus$byClass[,1]			# calculates false positive
  fn <- 1- confus$byClass[,2]			# calculates false negative
  rec <- (confus$byClass[,1]/(confus$byClass[,1] + fn))		# calculates recall
  fmeasure <- (2*confus$byClass[,3]*rec) / (confus$byClass[,3] + rec)	# calculates fmeasure
  stats <- data.frame("Precision"=confus$byClass[,3], "True Positive" = confus$byClass[,1],		#creates a dataframe to hold stats
                       "True Negative" = confus$byClass[,2], "False Positive"=fp,
                       "False Negative"=fn,"Recall"=rec, "F-Measure"=fmeasure)
  macros <- as.vector(sapply(c(1:6), function(x) mean(stats[,x],na.rm=TRUE)))	# calculate macros (means) for each columns
  #c(mean(stats[,1]),mean(stats[,2]),mean(stats[,3]),mean(stats[,4]),mean(stats[,5],na.rm=TRUE),mean(stats[,6],na.rm=TRUE))  
  microprec <- sum(stats[,2])/(sum(stats[,2]) + sum(stats[,4]))		# calculates micro avg precision
  microrec <- sum(stats[,2])/(sum(stats[,2]) + sum(stats[,5]))		# calculates micro avg recall
  micros <- c(microprec, "", "", "","", microrec, (2*microprec*microrec)/(microrec + microprec))	# calculates micro avg fmeasure
  
  # tidies up the stats table and adds the macro and micro data
  names <- row.names(stats)
  names <- append(names, "Macro Avg")
  names <- append(names, "Micro Avg")
  stats<- rbind(stats,macros)
  stats<- rbind(stats,micros)
  row.names(stats) <- names
  
  # ensures all numbers are reduced to 4 significant figures
  for (i in 1:nrow(stats)){ for (j in 1:ncol(stats)) stats[i,j] <- signif(as.numeric(stats[i,j]),digits=4)}
  
  #returns the statistics
  return(stats)
}
