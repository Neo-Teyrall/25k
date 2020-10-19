#Unix
#Mac
#setwd(dir = "/Users/MAEL/Documents/M2_BI/AIAO/AIAO_floobits/kaggle/projet_perso/stanford-covid-vaccine")

library(jsonlite)
library(reticulate)
#install.packages("RRNA")
library(RRNA)
library(stringr)

require(MASS)

np = import("numpy")

#Import des données
data_arn = list()
data_arn$test = lapply(readLines("../data/test.json"), fromJSON)
data_arn$train = lapply(readLines("../data/train.json"), fromJSON)


#Représentation graphique en mapping
data_graph1 = np$load(paste("bpps/", data_arn$test[[1]]$id, ".npy", sep = ""))
image(data_graph1)


#Représentation graphique RNA plot (visuel des boucles ARN)
str = data_arn$test[[6]]$structure
seq = data_arn$test[[6]]$sequence
ct = makeCt(str,seq)
coord = ct2coord(ct)
RNAPlot(coord, nt = TRUE, dp = 0.75)


#Exemple de print de ce qui existe dans chaque objet de test 
print(data_arn$test[[1]])


#Conversion de la list en dataframe pour traitement données
df_arn_test <- data.frame ( Reduce (rbind , data_arn$test) )
df_arn_train <- data.frame ( Reduce (rbind , data_arn$train) )
#Problème : les colonnes sont des listes et plus difficiles à traiter


#Résoudre problème : fonction unlist
for (num_col in 1:ncol(df_arn_test)) {
  df_arn_test[num_col] = unlist(df_arn_test[num_col])
  df_arn_train[num_col] = unlist(df_arn_train[num_col])
}


#Analyse sur les bases sur les données de train
#Comptage des bases
vec_bases = c("A", "U", "G", "C")
count_bases = matrix(data = NA, nrow = nrow(df_arn_train), ncol = 4,
       dimnames = list(df_arn_train$index , vec_bases) )
count_bases = as.data.frame(count_bases)

for (i in 1:length(df_arn_train$sequence)) {
  rang = str_locate(df_arn_train$sequence[i],"AAAAGAAACAACAACAACAAC")
  df_arn_train$sequence[i] = str_sub(df_arn_train$sequence[i],1, rang[1])
}

i = 1
for (seq in df_arn_train$sequence) {
  j = 1
  for (base in vec_bases) {
    count_bases[i,j] = str_count(seq, base)/str_length(seq)
    j = j+1
  }
  i = i+1
}

#Histogramme des proportions de bases

par(mfrow = c(2,2))
color_vec = c("#FEBFD2", "#FFFF6B", "#BEF574", "#77B5FE")
for (base in 1:ncol(count_bases)) {
  truehist(count_bases[,base], xlab = colnames(count_bases)[base],
           col = color_vec[base], xlim = c(0,0.8))
}

#Analyse sur les structures
vec_loop = c("E", "S", "H", "X", "I", "B","M")
count_struct = matrix(data = NA, nrow = nrow(df_arn_train), ncol = 7,
                     dimnames = list(df_arn_train$index ,vec_loop) )
count_struct = as.data.frame(count_struct)

i = 1
for (seq in df_arn_train$predicted_loop_type) {
  j = 1
  for (base in vec_loop) {
    count_struct[i,j] = str_count(seq, base)/str_length(seq)
    j = j+1
    
  }
  i = i+1
}
#Histogramme des proportions de structures
par(mfrow = c(3,3))
color_vec = c("#FEBFD2", "#FFFF6B", "#BEF574", "#77B5FE", "red", "magenta", "#CCCCFF")
for (base in 1:ncol(count_struct)) {
  truehist(count_struct[,base], xlab = colnames(count_struct)[base], col = color_vec[base])
}

#Chi test sur bases et struct
count_bases_loop = matrix(data = 0, nrow = 4, ncol = 7)
count_bases_loop = as.data.frame(count_bases_loop, row.names = vec_bases)
colnames(count_bases_loop) = vec_loop

for (row in 1:nrow(df_arn_train)) {
  for (loop in vec_loop) {
    tmp_loop = str_locate_all(df_arn_train[row, 5], loop)[[1]][,1]
    for (base in vec_bases) {
      tmp_base = str_locate_all(df_arn_train[row, 3], base)[[1]][,1]
      count_bases_loop[base,loop] = count_bases_loop[base,loop] + length(intersect(tmp_base, tmp_loop))
    }
  }
}

Xsq = chisq.test(count_bases_loop)
Xsq$p.value
Xsq$residuals

#Analyse sur structures (parenthèses)
vec_struct = c("(", ")", ".")
count_parenth = matrix(data = NA, nrow = nrow(df_arn_train), ncol = length(vec_struct))
count_parenth = as.data.frame(count_parenth)
colnames(count_parenth) = vec_struct
i = 1
for (seq in df_arn_train$structure) {
    j = 1
    for (struct in vec_struct) {
      count_parenth[i,j] = str_count(seq, fixed(struct))/str_length(seq)
      j = j+1
    }
    i = i+1
  }
count_parenth[,1] = count_parenth[,1] + count_parenth[,2]
count_parenth[,2] = count_parenth[,3]
count_parenth = count_parenth[,-3]
count_parenth

par(mfrow = c(1,2))
names_struct = c("()", ". point")
for (i in 1:ncol(count_parenth)) {
  truehist(count_parenth[,i],  xlab = names_struct[i],
           xlim = c(0,1), col = color_vec[i])
}

# Chi test sur bases et struct
count_bases_struct = matrix(data = 0, nrow = 4, ncol = 3)
count_bases_struct = as.data.frame(count_bases_struct, row.names = vec_bases)
colnames(count_bases_struct) = vec_struct

for (row in 1:nrow(df_arn_train)) {
  for (struct in vec_struct) {
    tmp_struct = str_locate_all(df_arn_train[row, 4], fixed(struct) )[[1]][,1]
    for (base in vec_bases) {
      tmp_base = str_locate_all(df_arn_train[row, 3], base)[[1]][,1]
      count_bases_struct[base,struct] = count_bases_struct[base,struct] + length(intersect(tmp_base, tmp_struct))
    }
  }
}

count_bases_struct[1] = count_bases_struct[1] + count_bases_struct[2]
count_bases_struct = count_bases_struct[,-2]

Xsq = chisq.test(count_bases_struct)
Xsq$p.value
Xsq$residuals


#Stats en vrac
summary(df_arn_test$seq_length)
summary(df_arn_test$seq_scored)
#df_arn_train$deg_error_pH10

write.table(df_arn_train$sequence[which(df_arn_train$SN_filter == 1)],
            row.names = FALSE,
            col.names = FALSE,
            file = "dataExS.csv")


