library(edgeR)
Args <- commandArgs(T)
treat = read.table(Args[1],sep = '\t',header = T, row.names = 1,check.names = F)
control = read.table(Args[2],sep = '\t',header = T,row.names = 1,check.names = F)


group = factor(c(replicate(ncol(control),1), replicate(ncol(treat),2)))
design = model.matrix(~group)
data = cbind(control, treat)
data = DGEList(counts = data, group = group)

keep = filterByExpr(data, group)
data = data[keep, ,keep.lib.sizes=FALSE]

y = calcNormFactors(data, method = 'TMM')   ### limma log
out_log = cpm(y, log = TRUE, prior.count = 1, normalized.lib.sizes = TRUE)
file_control_log = 'tmp_control.tsv'
file_treat_log = 'tmp_treat.tsv'

write.table(out_log[,1:ncol(control)],file=file_control_log ,quote = FALSE,sep='\t',
            row.names = TRUE,col.names = NA)

write.table(out_log[,-(1:ncol(control)),drop=FALSE],file=file_treat_log ,quote = FALSE,sep='\t',
            row.names = TRUE,col.names = NA)
