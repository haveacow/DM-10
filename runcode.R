# Data Mining
# ID: 1102930

# A list of lines of code to be run in the R terminal once functions.R has been loaded

# These are the librarys required for the following code
library(tm)
library(topicmodels)
library(caret)
library(randomForest)
library(e1071)

# we first load the .csv file
data <- read.csv(file='reutersCSV.csv',header=T,sep=",")

# then run the tagging function "tagsfind()"
data.tagged <- tagsfind(data)

# then create a reduced dataset with only the 10 most frequent terms
data.red <- data.tagged[data.tagged$tags %in% c("earn", "acq", "money.fx", "grain", "crude", "trade", "interest", "ship", "wheat", "corn"), ]

# we create a corpus from the doc.text
mycorpus1 <- Corpus(VectorSource(data.red$doc.text))

# then perform transformations to clean the corpus as part of pre-processing
# more details about each step are in the report
mycorpus1 <- tm_map(mycorpus1, tolower)
mycorpus1 <- tm_map(mycorpus1, removePunctuation)
mycorpus1 <- tm_map(mycorpus1, removeNumbers)
mycorpus1 <- tm_map(mycorpus1, removeWords, stopwords("english"))
mycorpus1 <- tm_map(mycorpus1, stemDocument)
mycorpus1 <- tm_map(mycorpus1, stripWhitespace)
mycorpus1 <- tm_map(mycorpus1, PlainTextDocument)

# we create a DocumentTermMatrix with term frequency weighting and words that are 4 letters or longer and appear in at least 5 documents
DTM.tf <- DocumentTermMatrix(mycorpus1, list(weighting = weightTf, wordLengths=c(4, Inf), bounds = list(global = c(5,Inf)) ))

# we remove those words that have a sparsity above 98%
DTM.tf <- removeSparseTerms(DTM.tf,0.98)

# we used LDA to find 500 terms to use for a new dictionary
LDA <- LDA(DTM.tf,10, method= "Gibbs")
ldaterms <- terms(LDA, 50)

# create the new DocumentTermMatrix using the LDA dictionary
DTM.lda <- DocumentTermMatrix(mycorpus1, list(weighting = weightTfIdf, wordLengths=c(4, Inf), dictionary=as.vector(ldaterms)))

# remove sparse words as before
DTM.lda2 <- removeSparseTerms(DTM.lda,0.93)

# we transform the DocumentTermMatrix into a dataframe, and add the purpose and tags columns from before
DTM.df <- as.data.frame(as.matrix(DTM.lda2))
DTM.df2 <- as.data.frame(inspect( DTM.tf))
b <- cbind(data.red[c("purpose","tags")],DTM.df)

# not tested in the following code, but here in case you wish to explore the term frequency feature set as I did
b2 <- cbind(data.red[c("purpose","tags")],DTM.df2) 

# we must transform the tags column to factors for the classifiers
b$tags <- as.factor(b$tags)

# this line performs classification on the test/train split, and outputs results to matrix2
results <- testtrain(b)

# this line performs classification on a 10 k-fold split, and outputs results to matrix
# necessary to remove the 1st column (purpose) in advance of running this crossfold function
results2 <- crossfold(b[,-1])

# this line writes out the dataframe to a .csv file to be used in Weka for clustering
write.csv(b, file = "ClusterTest.csv")
