#!/usr/bin/Rscript
# -*- coding: utf-8 -*-

#####################
## R Command Index ##
#####################

## Exit
q()

## Workspace leeren
rm(list=ls())

## Arbeitsverzeichnis abfragen, setzen
getwd()
setwd("/home/user/rfolder")

## Die im Arbeitsverzeichnis befindliche Datei so.R ausführen (Linux/Unix)
source('so.R')

## Vom Terminal aus das R-Skript so.R ausführen (Linux/Unix) - nicht in R
R CMD BATCH so.R

## Paket <paketname> installieren, einbinden, ent-/ausladen, Datensatz <datensatz> laden und 
## Beschreibung für bmi anzeigen lassen
install.packages()
install.packages(c("fGarch","fBasics","stabledist"), repos="https://cran.uni-muenster.de/")
install.packages("/home/tobias/Downloads/UsingR_2.0-1.tar.gz", repos=NULL)
library(<paketname>)#bzw. require(<paketname>)
detach("package:<paketname>")
data(<datensatz>)
?bmi

## Repos umdefinieren (z.B. zu https)
local({
   r <- getOption("repos");
   r["CRAN"] <- "https://cran.rstudio.com/"
   options(repos=r)
})

## Datensatz vorauswählen mit attach
attach(airquality)
Oznoe#statt airquality$Ozone
detach(airquality)

## Variable x kommentieren
x <- NA; comment(x) <- "Kommentar"; comment(x)

## Befehlshistorie ausgeben
history()

##
## Grundlegendes
##
##

## Text ausgeben
print("Text")

## Konkatenieren bzw. zusammenführen
i <- 1; paste("Konkatenieren von", i, "und", i)
i <- 1; cat("Konkatenieren von", i, "und", i)

## Objekte nicht ausgeben (z.B. statt return())
invisible(x)

## Vergleichsoperatoren: equal, not equal, greater/less than, greater/less than or equal
==
!=
> <
>= <=

## Logische Operatoren: and, or, not
&
|
!

## Grundrechenarten
2 + 3
2 - 3
2 * 3
2 / 3
2 ** 3
2 ^ 3
1i ** 2
sqrt(9)
log(0)

## Matrizenoperationen
c(1, 1) %*% c(1, 2)
outer(c(1, 1), c(1, 2))

## Eigenwerte
eigen(outer(c(1, 1), c(1, 2)))

## Vektoren
c(1:4)
rep(NA, 4)
seq(1.575, 5.125, by = 0.05)

##
sort()

##
order()

## Länge eines Vektors
length(airquality$Ozone)

##Transposition einer Matrix
t(matrix)

## Dimension einer Matrix
dim(airquality)#fixme

## Zeilenanzahl
nrow(airquality)
ncol(airquality)

## Summe
sum()

## Mittelwert über eingegebenen Vektor
mean(vektor)
## Median über eingegebenen Vektor
median(vektor)

## Funktionen zum Kürzen etc
max()#Maximaler Wert eines Vektors
min()#Minimaler Wert eines Vektors
abs()#Absoluter Wert (Betrag)
ceiling()#Nächster größerer ganzzahliger Wert
floor()#Nächster kleinerer ganzzahliger Wert
trunc()#Nächster ganzzahliger Wert in Richtung der Null

## Wichtige Funktionen
sqrt()#Wurzel-Funktion
exp()#Exponential-Funktion
log()#Natürlicher Logarithmus (zur Basis e)
log10()#Logarithmus zur Basis 10
gamma()#Gamma-Funktion
lgamma()#natürlicher Logarithmus der Gamma-Funktion

### Trigonometrische Funktionen
cos()#Cosinus bzw. Kosinus
sin()#Sinus
tan()#Tangens
### Hyperbolische Trigonometrische Funktionen
cosh()#Cosinus Hyperbolicus
sinh()#Sinus Hyperbolicus
tanh()#Tangens Hyperbolicus
### Inverse Trigonometrische Funktionen
acos()#Arcuscosinus bzw. Arkuskosinus
asin()#Arcussinus bzw. Arkussinus
atan()#Arcustangens bzw. Arkustangens
### Inverse Hyperbolische Trigonometrische Funktionen
acosh()
asinh()
atanh()



atan2(y, x)

cospi(x)
sinpi(x)
tanpi(x)

## Weitere Funktionen
log()

##
## weitere algebraische Funktionen
##
##

## Primfaktorzerlegung
prime.factor(24)

## Primzahlen innerhalb eines Intervalls ausgeben
primes(1,9)#  gibt immer bis 17 aus??!?

##
## Datensätze
##
##

## Spalten beschriften und Zeilennamen abrufen
colnames(airquality)[length(airquality)] <- "Windstaerke"
rownames(airquality)

## Vektoren x,y,z zu Matrix zusammenfügen
cbind(x, y, z)
rbind(x, y, z)

## Teilmenge von Datensatz bilden
subset(airquality, subset=(airquality$Wind<10))
airquality[airquality$Wind<10,]

## Elemente, die in spalte vorkommen (keine Mehrfachnennung)
unique(datensatz$spalte)

##
factor(letters[1:20], labels="letter")

## Datensatz "airquality" nach allen in Spalte "Month" vorkommenden Merkmalen aufteilen/gruppieren
split(airquality,airquality$Month)

## Datensatz chickwts ohne Zeilen mit NA Einträgen
na.omit(chickwts)

## Position von Einträgen in Spalte feed, die Bed. erfüllen
which(chickwts$feed=="horsebean")

## Eintrag auswählen
chickwts$feed[1]

## Zufalls-Lottozahlen generieren
sample(49, 6, replace=FALSE)

## Unfairer Würfel
sample(6, 1, replace=TRUE, prob=c(rep(0.1,5),0.5))

## Datensatz von URL holen
read.table("http://www.informatik.uni-bremen.de/~farhad/gewicht.txt", 
header=TRUE)


##
## Datentypen und verschiedene Objekte
##
##

## Datentypen String, ???, reelle Zahl, komplexe Zahl und Fehlerwert überprüfen
is.character()
is.factor()
is.numeric()
is.complex()
is.na()
mode()#character, numeric, logical

## Analog die Umwandung in andere Datentypen
as.character()

## Ein leeres Objekt erstellen
x <- NULL

## Strings: Anzahl von Buchstaben von x ausgeben
nchar(x)

##
## Wichtige Programmierkonstrukte
##
##

## if, elseif, else
if (variable==TRUE) {
  DoThis(variable)
} else if (variable!=FALSE) {
  return(DoThat(variable))
} else if (DoSomeThingWrong(variable)) {
  stop("Da ist was schief gelaufen.")
} else {
  break("Jetzt ist aber Schluss!") # Dieser Text wird nicht ausgegeben
}

## Wenn x==NA, dann gebe 999 zurück, sonst x
ifelse(is.na(x), 999, x)

## Switch-Funktion: zwei Bsple
switch(2, 'some value', 'something else', 'some more') # gibt 'something else'
for(i in c(-1:3, 9))
  print(switch(i, 1, 2 , 3, 4))

## for-Schleife
for(i in c(1:5)) {
  print(i)
}

## Funktionen
MeineFunktion <- function(n) {
  if (missing(n))
    stop("Bitte n angeben.")
  print(n)
}
stop(<logical-expression>)#gibt bei Fehlerausgabe auch Funktionsnamen an
stopifnot(<logical-expression>)
return(<something>)


##
## Plotten/Ausgaben erzeugen
##
##

## Einfachster Plotbefehl
plot(cos)

## Eine Kurve plotten (einfachster Weg)
curve(4*x - 4*x^2, 0, 0.5)

pie()
barplot()
hist()
density()
boxplot()
abline()

## Device-Funktionen
x11()#Fenster-Device für Linux
windows()#Fenster-Device für Windows
win.metafile()#Windows Metafile
bmp()
postscript()
pdf()
jpeg()
png()

## Plot Beispiele
plot(x=cars$speed, y=cars$dist, pch=4, lwd=2, col="blue",
    xlab="speed", ylab="distance", main="cars")
plot(iris[, 1], iris[, 2], col="red", pch=2, main="Kelchblatt", cex=1, 
    cex.axis=0.8, cex.main=2, col.axis=2, xlab="Länge", ylab="Breite")
plot(iris)
pairs(iris, panel=panel.smooth)

## Plot direkt ausgeben
graphics.off()#löscht alle devices
par(mfrow = c(1, 2))
pie(table(bak$resistenz),main="Bakterienresistenz")
barplot(table(bak$resistenz),main="Bakterienfarbe")

## Eine Linie von (x[1],y[1]) nach (x[2],y[2]) plotten
x <- c(0, 1); y <- c(0, 1); plot(x, y, type="l")

## Plot in postscript-Datei
postscript(file="my_new_graphic.ps")
barplot(table(bak$farbe))
pie(table(bak$farbe))
dev.off()

## Plot in .pdf
pdf(width=14, height=7, file="tab2.pdf")
pie(tab1[5, 1:4 , 3], col=col2, main="Augenfarbe-Alle")
dev.off()

## Plot in Fenster Beispiel
x11()
op <- par(mfrow = c(1, 2))
hist(table(airquality$Ozone), freq=F)
boxplot(airquality$Ozone, main=colnames(airquality$Ozone))
par(op)
dev.off()

## Gerade mit Steigung 0 und y-Achsen-Abschnitt 1 einzeichnen
abline(1, 0)

## Konsolenoutput in Datei umleiten
sink("output1.txt")
sink()#stoppt Umleitung

## cat-Befehl (wie in der Linux Shell)
x <- 1; cat("x == 0")
print("x == 0")

##
## Tabellen
##
##

## Kontingenztafeln mit mehr als 2 Merkmalen
table(t1)
prop.table(t1)
ftable(t1)#erfordert das Package "vcd"
structable(formula=Hair~Sex+Eye, data=t1)#erfordert das Package "vcd"

## Tabellenspalten permutieren
t1 <- HairEyeColor
t1[, , c(2,1)]
t1[c(4, 3, 2, 1), c(3, 2, 1, 4), ]
t2 <- aperm(t1, perm=c(2, 1, 3))

##
## Data frames
##
##
neuerDataFrame <- data.frame(c(2,9,5),c("bli","bla","blub"),c(TRUE,FALSE,TRUE))
col_headings <- c('Zahlen','Spam','Boolsch')
names(neuerDataFrame) <- col_headings

##
## Statistische Funktionen
##
##

## Chi-Quadrat Funktion und Test
chisq()
chisq.test()

## Varianz, Kovarianz
var(iris[1:4]); cov(iris[1:4])

## Korrelation
cor(iris[1:4]) # method ="pearson"
cor(iris[1:4], method="spearman")
cor(iris[1:4], method="kendall")

## Verteilungen, Zufallszahlen
# statt d- (Dichte) Anfangsbuchstaben sind auch p (Verteilungsfunktion), 
# q (Quantilsfunktion) und r (Zufallszahl) möglich. Also z.B. rnorm()
dnorm #Normal-
dlnorm #Lognormal-
dt #Student-t-
dchisq #chi^2
df #f-
dexp #Exponential-
dunif #Gleich-
dbeta #Beta-
dcauchy #Cauchy-
dbinom #Binomial-
dmultinom #Multinomial-
dpois #Poisson-
dgeom #Geometrische-
dweibull #Weibull-
dwilcoxon #Wilcoxon- #fixme

## Anwendungsbeispiel für p-Quantile von 0.01 bis 0.99
qnorm(seq(0.01,0.99,0.01),0,1)

##
## Zusätzlich installieren: Skew-normal distribution
install.packages(c("fGarch","fBasics","stabledist"), repos="https://cran.uni-muenster.de/")
library(fGarch)
library(fBasics)
library(stabledist)
hist(rsnorm(100, mean = 0, sd = 1, xi = 1.5))


##
## Testen
##
##

## T-Test: Mittelwertanalyse
t.test(normtemp[,1]~normtemp[,2], 
  alternative=c("less", "greater", "two.sided"), #"less" bei Gleichheit, "greater" wenn Aternative größer als Nullhyp. sein soll
  conf.level=1-alpha, 
  var.equal=F)
## Beispiel für T-Test
data1 <- c(1, 2, 3, 4); data2 <- c(2, 3, 4, 6)
t.test(data1, data2, paired=F, var.equal=F)$p.value
t.test(data1, data2, paired=T)$p.value

## Ermittlung von Fallzahlen bei t-test
n.ttest(power = 0.8, alpha = 0.05, mean.diff = 0.8, sd1 = 0.83, sd2 = 2.65, k = 2, design = "unpaired",fraction = "unbalanced")

## Varianztests/-analyse
var.test() #(vergleicht die Quotienten zweier Varianzen beruhend auf F-test)


##
## Power, Fallzahl,
##
##

## n, delta, sd, sig.level oder power berechnen. Zu berechnendes als NULL übergeben.
power.t.test(
  n = c(
        NULL, # Falls Fallzahl berechnet werden soll
        length(Gruppe1) # Für gleich große Gruppen: length(Gruppe1) == length(Gruppe2)
        2*length(Gruppe1)*length(Gruppe2)/(length(Gruppe1)+length(Gruppe2)) # Für unterschiedlich große Gruppen: length(Gruppe1) != length(Gruppe2)
      )
  delta = c(
        NULL, # Falls Mittelwert von Abweichung berechnet werden soll
        mean(Gruppe1-Gruppe2) # Für Zweistichproben-t-Tests (gepaart)
      )
  sd = c(
         sd(Gruppe1-Gruppe2), #Bei gepaartem Zweistichproben-t-Test
         sqrt(((n1-1)*s1**2+(n2-1)*s2**2)/((n1-1)+(n2-1))) #Bei ungepaartem Zweistichproben-t-Test die pooled sd, wobei n1=length(Gruppe1), n2=length(Gruppe2), s1=sd(Gruppe1), s2=sd(Gruppe2)
       )
  sig.level = alpha, 
  power = NULL, 
  type = c("two.sample", "one.sample", "paired"), 
  alternative = c("two.sided", "one.sided")
)

## Power bei fixer Fallzahl etc berechnen
# Ziel: neue Therapie zeigt größere Wirkung mit Signifikanzniveau alpha
# Gegeben: Zwei Stichproben Standard und Neu mit gleicher Größe, ungepaart
power.t.test(n = length(Standard), 
  delta = mean(Standard)-mean(Neu), 
  sd = sd(Standard-Neu), 
  sig.level = alpha,
  power = NULL, 
  type = "two.sample",
  alternative = "two.sided"
)

## Fallzahl bei fixer Power etc berechnen
# Ziel: neue Therapie zeigt größere Wirkung mit Signifikanzniveau alpha
# Gegeben: Zwei Stichproben Standard und Neu mit gleicher Größe, ungepaart
power.t.test(n = NULL, 
  delta = mean(Standard)-mean(Neu), 
  sd = sd(Standard-Neu), 
  sig.level = alpha,
  power = 0.9, 
  type = "two.sample", 
  alternative = "two.sided"
)


##
## Vorinstallierte Datensätze
##
##
airquality
Titanic
cars
iris
mtcars


##
##
## Verschiedene Beispiele
##
# Random walk
plot(cumsum(rnorm(100)),type="l")




##
##  Funktionen zur Benutzerinteraktion
##
##

## Variablen aus Benutzereingabe lesen
cat("\n","Enter x","\n") # prompt
y<-scan(n=1)

## Skriptausführung pausieren
Sys.sleep(1)

## Warten bis Benutzer Enter drückt
cat ("Press [enter] to continue"); line <- readline()
print ("Press [enter] to continue"); number <- scan(n=1)

##
##
## Weiterführendes:
##
## Ein Styleguide
## https://google-styleguide.googlecode.com/svn/trunk/Rguide.xml
##
## Writing R-Scripts
## http://cran.r-project.org/doc/contrib/Lemon-kickstart/kr_scrpt.html
##
## R-Befehel-Index
## https://de.wikibooks.org/wiki/GNU_R:_Befehle-Index
##
## CRAN-Spiegelserver
## http://cran.r-project.org/mirrors.html
##

## Code für das bessere Verständnis kommentieren
cat("\nMean risk-taking scores for all groups\n\n")
brkdn(risk~group,adolrisk03.df)
