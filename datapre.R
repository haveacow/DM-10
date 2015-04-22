setwd("C:/Users/James/Desktop/CS909")
library(tm)
library(topicmodels)

data <- read.csv(file='reutersCSV.csv',header=T,sep=",")
data.tagged <- tagsfind(data)
data.red <- data.tagged[data.tagged$tags %in% c("earn", "acq", "money.fx", "grain", "crude", "trade", "interest", "ship", "wheat", "corn"), ]

mycorpus1 <- Corpus(VectorSource(data.red$doc.text))
mycorpus1 <- tm_map(mycorpus1, tolower)
mycorpus1 <- tm_map(mycorpus1, removePunctuation)
mycorpus1 <- tm_map(mycorpus1, removeNumbers)
mycorpus1 <- tm_map(mycorpus1, removeWords, stopwords("english"))
mycorpus1 <- tm_map(mycorpus1, stemDocument)
mycorpus1 <- tm_map(mycorpus1, stripWhitespace)
mycorpus1 <- tm_map(mycorpus1, PlainTextDocument)
DTM.tf <- DocumentTermMatrix(mycorpus1, list(weighting = weightTf, wordLengths=c(4, Inf), bounds = list(global = c(5,Inf)) ))
DTM.tf <- removeSparseTerms(DTM.tf,0.98)

LDA <- LDA(DTM.tf,10, method= "Gibbs")
ldaterms <- terms(LDA, 50)

DTM.lda <- DocumentTermMatrix(mycorpus1, list(weighting = weightTfIdf, wordLengths=c(4, Inf), dictionary=as.vector(ldaterms)))

DTM.lda2 <- removeSparseTerms(DTM.lda,0.93)

DTM.df <- as.data.frame(as.matrix(DTM.lda2))
b<- cbind(data.red[c("purpose","tags")],DTM.df)

b$tags <- as.factor(b$tags)
matrix2 <- testtrain(b)
matrix <- crossfold(b[,-1])



write.csv(b, file = "ClusterTest.csv")
