suppressPackageStartupMessages(library(limma))
suppressPackageStartupMessages(library(GEOquery))
suppressPackageStartupMessages(library(BiocGenerics))
gse = getGEO(filename = 'GSE26972_series_matrix.txt.gz', getGPL = FALSE)
gse$title
gse@experimentData
gse@annotation
gpl = getGEO(filename = 'GSE26972_family.soft.gz')  ## 得到注释信息, 用python解析
write.table(gpl@gpls$GPL5188@dataTable@table[, c(1, 13)], file = 'annoRaw.tsv', sep='\t', quote = FALSE, row.names = FALSE)
anno = read.table('anno.tsv', sep = '\t', header = TRUE)
exp  = as.data.frame(exprs(gse))
exp = exp[rownames(exp) %in% anno$ProbeID,]  ### 保留有Symbol的探针
exp$ProbeID = rownames(exp)
exp1 = merge(exp, anno, by='ProbeID')
exp1$ProbeID = NULL
exp1$Symbol = NULL
expSet <- aggregate(x = exp1[,1:6], by = list(exp1$Entrez), FUN = max) ###平局值
rownames(expSet) = expSet$Group.1
expSet$Group.1 = NULL
write.table(expSet, file='exp.tsv', sep = '\t', quote = FALSE, col.names = NA, row.names = TRUE)

