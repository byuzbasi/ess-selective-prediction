# Descriptive scientific figures from locally generated CSV files; base R only.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==1L)
folder <- args[1]
paths <- file.path(folder,c('figure1_country_gaps.pdf','figure2_turnover.pdf'))
stopifnot(!any(file.exists(paths)))
d <- read.csv(file.path(folder,'figure1_country_gaps.csv'))
stopifnot(nrow(d)==38L)
pdf(paths[1],width=8,height=6)
par(mar=c(5,4,1,1))
countries <- sort(unique(d$country))
colors <- c(catboost='#244A73',tabm='#B65D27')
plot(0,0,type='n',xlim=c(1,length(countries)),ylim=range(c(0,d$gap_pp)),xaxt='n',
     xlab='Target country',ylab='Low minus bachelor or higher coverage percentage points')
axis(1,at=seq_along(countries),labels=countries,las=2)
abline(h=0,col='gray60')
for (family in names(colors)) {
 z <- d[d$family==family,]; lines(match(z$country,countries),z$gap_pp,type='b',pch=16,col=colors[family])
}
legend('bottomleft',legend=c('CatBoost','TabM'),col=colors,pch=16,lty=1,bty='n')
dev.off()
d <- read.csv(file.path(folder,'figure2_turnover.csv'))
stopifnot(nrow(d)==4L)
pdf(paths[2],width=8,height=4)
par(mfrow=c(1,2),mar=c(4,5,2,1))
for (rule in c('primary80','source_tau')) {
 z <- d[d$rule==rule,]
 barplot(z$turnover_percent,names.arg=c('CatBoost','TabM')[match(z$family,c('catboost','tabm'))],
         col=unname(colors[z$family]),ylab='Selection turnover percent',
         main=if (rule=='primary80') 'Fixed 80 percent capacity' else 'Fixed source threshold')
}
dev.off()
