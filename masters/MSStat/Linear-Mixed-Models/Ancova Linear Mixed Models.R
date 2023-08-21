
#              4. Ancova - Linear Mixed Models 

library(haven)
require(nlme)
summary(RatPupWeight)

ratpup <-as.data.frame( read.table("rat_pup.txt", h = T))
head(ratpup, n = 10)  # επαναλαμβανόμενες μετρήσεις σε μορφή "long format"
                      # γραμμές που δεν είναι ανεξάρτητες
tail(ratpup, n = 10)
summary(ratpup)

attach(ratpup)
names(ratpup)   

apply(ratpup,MARGIN=2,"class")
is.factor(sex)


# Περιγραφική στατιστική για κάθε συνδυασμό των επιπέδων των παραγόντων
# με τη συνάρτηση summary
with(ratpup, tapply( weight, interaction(treatment,sex),summary))

round(with(ratpup, tapply( weight, interaction(treatment,sex),sd)),3)

# Ένας ακόμη τρόπος να πάρουμε περιγραφική στατιστική της κύριας μεταβλητής
# λαμβάνοντας υπόψη τους παράγοντες που εξετάζονται
# Η εντολή llist λειτουργεί όπως η list με τη διαφορά ότι διατηρεί
# τις εικέτες στις μεταβλητές που είναι παράγοντες (variables label attribute).

require(Hmisc)
g <- function(x)c(N=length(x),MIN=min(x,na.rm=TRUE),MAX=max(x,na.rm=TRUE),
                 MEDIAN=median(x,na.rm=TRUE), MEAN=mean(x,na.rm=TRUE),
                 SD=sd(x,na.rm=TRUE))
s1 <- summarize(weight,by=llist(treatment,sex),g) ;s1 
class(s1)
s1.1 <- within(s1, {
  MEAN <- round(MEAN, 3) 
  SD <- round(SD, 3)})
s1.1
s1.2 <- s1.1[order(-s1.1$MEAN), ]
s1.2



# Διάγγραμμα αλληλεπιδράσεων
with(ratpup, interaction.plot( treatment,sex,weight))

# Οπτικός έλεγχος ισότητας διασπορών για τις ομάδες που δημιουργούνται
# από τους συνδυασμούς φύλου και αγωγής

# Η συνάρτηση που χρησιμοποιείται για την κατασκευή των boxplots,
# αρχικά οργανώνει τα δεδομένα ως προς τους παράγοντες TREAT και SEX έτσι
# ώστε τα επίπεδα του TREAT αποτελούν Ομάδες που μέσα σ' αυτές (within)
# παρουσιάζονται τα επίπεδα του παράγοντα SEX

library(lattice)  # trellis graphics
library(grid)

bwplot(weight ~ sex|treatment, data=ratpup,aspect = 2, ylab="Birth Weights", 
       xlab="SEX",main = "Boxplots of birth weights for levels of treatment 
       by sex")

# Στην προηγούμενη γραφική παράσταση δε λάβαμε υπόψη τις επαναλαμβανόμενες
# μετρήσεις ανά μητέρα-γέννα (litter), κάτι που θα κάνουμε με τα επόμενα 
# θηκογράμματα. Κάθε παράθυρο θα περιέχει την κατανομή των βαρών για κάθε
# γέννα ανά συνδυασμό φύλου και αγωγής. Για το σκοπό αυτό δημιουργείται η
# μεταβλητή ranklit, ο μικρότερος αριθμός νεογέννητων ήταν 2 (litsize=1)
# και ο μεγαλύτερος 18, (litsize=27).

ranklit <- litsize+0.01*litter
sort(ranklit)
ranklit
ranklit <- factor(ranklit)

levels(ranklit) <- c( "1","2", "3","4","5","6","7","8","9","10", "11","12", 
                      "13","14","15","16","17","18","19","20", "21","22", 
                      "23","24","25","26","27")

ranklit

bwplot(weight ~ ranklit | treatment*sex, data=ratpup, aspect = 0.5, 
       ylab="Birth Weights", xlab="" , groups=litter,
       scales = list(x = list(at=c(4,8,12,16,20,24))))


#------------------------------------------------------------------------
#                      Model 3.1.
#     πλήρες μοντέλο με τυχαίους και σταθερούς παράγοντες
#------------------------------------------------------------------------

library(nlme)
library(lme4)

# Δημιουργούμε την δείκτρια μεταβλητή sex1 με τιμές 1 (για θηλυκό) 
# και 0 (αρσενικό). Οι άνδρες μετατρέπονται σε ομάδα αναφοράς
ratpup$sex1[sex == "Female"] <- 1
ratpup$sex1[sex == "Male"] <- 0
attach(ratpup)

# Πως χρησιμοποιείται ο παράγοντας treatment ?
str(treatment)

# Το πλήρες μοντέλο
model3.1.fit <- lme(weight ~ treatment + sex1 + litsize + treatment*sex1,
                    random = ~1 | litter, ratpup, method = "REML")

mat1 <- as.data.frame(unique(model.matrix(model3.1.fit)))
names(mat1)
mat2 <- unique(mat1[ , c(-5)])
mat2   # O ΠΙΝΑΚΑΣ ΣΧΕΔΙΑΣΜΟΥ ΧΩΡΙΣ ΕΠΑΝΑΛΗΨΕΙΣ ΚΑΙ ΧΩΡΙΣ ΤΗΝ "litsize"

# Πίνακας σχεδιασμού σταθερών επιδράσεων
# Μετατροπή των παραγόντων σε 0-1 διανύσματα  (αντιθέσεις-contrasts)
contrasts(ratpup$treatment)
contrasts(ratpup$sex)

# Ποιος είναι ο πίνακας σχεδιασμού με την litsize, χωρίς τις επαναλήψεις
# των γραμμών που περιέχουν τους ίδιους συνδυασμούς τιμών ?
model3.1.matr <- as.matrix(unique(model.matrix(model3.1.fit)))
View(model3.1.matr)
# Τον μεταφέρουμε πχ σε excel αφού τον αποθηκεύσουμε ως αρχείο txt 
write.table(model3.1.matr, file = "design,matrix.rutpap.txt", row.names = F)
# Ο πίνακας σχεδιασμού για την 3η μητέρα
model.matrix( weight ~ treatment + sex1 + litsize + treatment*sex1, 
              ratpup[ ratpup$litter == "3", ])
# Ο πίνακας σχεδιασμού για την 25η μητέρα
model.matrix( weight ~ treatment + sex1 + litsize + treatment*sex1, 
              ratpup[ ratpup$litter == "25", ])

# Αποτελέσματα του μοντέλου
ranef(model3.1.fit)
summary(model3.1.fit)
anova(model3.1.fit)

# Έλεγχος υπόθεσης σε likelihood ratio test 
library(RLRsim)
exactRLRT(model3.1.fit)


# Display the random effects (EBLUPs) from the model.
# Η τυχαία επίδραση που οφείλεται στη μητέρα (27 συνολικά)
random.effects(model3.1.fit)


#------------------------------------------------------------------------
#                      Model 3.1A.
#         πλήρες μοντέλο χωρίς τυχαίους παράγοντες
#------------------------------------------------------------------------

# Model 3.1A.
model3.1a.fit <- gls(weight ~ treatment + sex1 + litsize + treatment*sex1, 
                     data = ratpup)
summary(model3.1a.fit)

# Έλεγχος της υπόθεσης ότι η επίδραση του τυχαίου παράγοντα είναι μηδενική
# Διαιρούμαι δια 2 την p-τιμή του ελέγχου διότι ο έλεγχος χ-τετράγωνο για την 
# τιμή 0 είναι στο όριο τιμών της κατανομής επομενως το σωστό είναι να 
# παρουσιαστεί ως μονόπλευρος έλεγχος.
anova(model3.1.fit, model3.1a.fit)  

# Ο έλεγχος με το χέρι
-200.5522-(-245.255) # = 44.7028
A1pval<-0.5*(1-pchisq(44.7028,1))+0.5*(1-pchisq(44.7028,2)) # = 1.096142e-10
A1pval


#------------------------------------------------------------------------
#                      Model 3.2.
#         Έλεγχος για την ισότητα των διασπορών
#------------------------------------------------------------------------


# Model 3.2A. 
# Στα θηκογράμματα φάνηκε ότι στην αγωγή Control η διασπορά ήταν αυξημένη
# τόσο στα αρσενικά όσο και στα θηλυκά επομένως ο παράγοντας treatment
# καθορίζει τον πίνακα διασπορών-συνδιασπορών των τυχαίων παραγόντων. 

model3.2a.fit <- lme(weight ~ treatment + sex1 + litsize + treatment*sex1, 
                     random = ~1 | litter, ratpup, method = "REML", 
                     weights = varIdent(form = ~1 | treatment))
summary(model3.2a.fit)

# Έλεγχος για την υπόθεση ότι η διαφορά στις διασπορές μεταξύ των 
# αγωγής είναι μηδενική (συνολικός έλεγχος).
anova(model3.1.fit, model3.2a.fit)  

# απορρίπτεται επομένως υπάρχει στατιστικά σημαντική διαφορά
# μεταξύ των αγωγών ως προς τις διασπορές και αυτό θα ληφθεί
# υπόψη.



#------------------------------------------------------------------------
#                      Model 3.3.
#         Έλεγχος για την ισότητα των διασπορών στις 
#           αγωγές high και low (παράγοντας treatment)
#------------------------------------------------------------------------

# Προχωρώντας την προηγούμενη υπόθεση, θα ελεγχθεί αν οι αγωγές 
# high και low έχουν ίσες διασπορές. Αυτό θα γίνει ορίζοντας μια νέα μεταβλητή
# στην οποία οι δυο αγωγές γίνονται μια κατηγορία. Στη συνέχεια
# αυτή η νέα μεταβλητή θα καθορίσει τον πίνακα διασπορών-συνδιασπορών
# (pooled variance)

ratpup$trtgrp[treatment == "Control"] <- 1
ratpup$trtgrp[treatment == "Low" | treatment == "High"] <- 2

# Η νέα μεταβλητή "trtgrp" θα εισαχθεί στο όρισμα "weights" καθορίζοντας
# με ακρίβεια τη σχέση των διασπορών συνδιασπορών του τυχαίου παράγοντα,

model3.2b.fit <- lme(weight ~ treatment + sex1 + litsize +
                      treatment*sex1, random = ~1 | litter, ratpup, 
                     method = "REML",
                     weights = varIdent(form = ~1 | trtgrp))

# Εφαρμόζουμε και πάλι τον έλεγχο (likelihood ratio test) για να ελέγξουμε
# αν το νέο μοντέλο είναι καλύτερο ή ισοδύναμο με το προηγούμενο

anova(model3.2a.fit, model3.2b.fit) 

# Το αποτέλεσμα δεν είναι στατιστικά σημαντικό επομένως μπορούμε να κρατήσουμε
# το μοντέλο με την ανομοιογενή διασπορά (pooled για τις αγωγές high και low 2Β) 
# ενώ επιπλέον θα ελέγξουμε αν αυτό το μοντέλο είναι προτιμότερο από αυτό με 
# την ομοιογενή διασπορά (homogeneous  error variance model, Model 3.1)

anova(model3.1.fit, model3.2b.fit)  
# το αποτέλεσμα είναι στατιστικά σημαντικό επομένως θα κρατήσουμε το μοντέλο
# με την ανομοιογενή διασπορά .
summary(model3.2b.fit)

# Στο τελικό βήμα μένει να κρατήσουμε στο μοντέλο τις σημαντικές σταθερές κύριες
# επιδράσεις και αλληλεπιδράσεις. ΠΡΟΣΟΧΗ χρειάζεται τώρα όμως γιατί
# ο έλεγχος αυτός γίνεται με χρήση των ML εκτιμητών και όχι των REML

#Fixed effects: weight ~ treatment + sex1 + litsize + treatment * sex1 
#                      Value   Std.Error  DF   t-value p-value
#(Intercept)         8.350351 0.27567833 292 30.290196  0.0000
#treatmentHigh      -0.901844 0.19140146  23 -4.711793  0.0001
#treatmentLow       -0.466596 0.15999337  23 -2.916347  0.0078
#sex1               -0.408195 0.09303540 292 -4.387529  0.0000
#litsize            -0.130383 0.01856367  23 -7.023574  0.0000
#treatmentHigh:sex1  0.092026 0.12461723 292  0.738473  0.4608
#treatmentLow:sex1   0.076397 0.10939797 292  0.698337  0.4855

# Παρατηρούμε ότι  αλληλεπίδραση "treatment*sex1" δεν είναι 
# στατιστικά σημαντική για το μοντέλο, επομένως θα δημιουργήσουμε 
# το περιορισμένο μοντέλο και θα τα συγκρίνουμε στη συνέχεια.
# Επιπλέον θα δοκιμάσουμε και το μοντέλο χωρίς τον παράγοντα 
# treatment μιας και στον προηγούμενο πίνακα φαίνεται ότι ίσως 
# είναι η μικρότερη επίδραση.


#------------------------------------------------------------------------
#                      Model 3.4 - 5.
#         Έλεγχος για για μη σημαντικές κύριες επιδράσεις και 
#           αλληλεπιδράσεις των σταθερών παραγόντων
#                με χρήση εκτιμητών ML
#------------------------------------------------------------------------

# Test Hypothesis 3.5.
anova(model3.2b.fit)

#Model 3.3, χωρίς την αλληλεπίδραση με χρήση εκτιμητών ML (method = "ML") 
model3.3.ml.fit <- lme(weight ~ treatment + sex1 + litsize,
                         random = ~1 | litter, ratpup, method = "ML", weights =
                           varIdent(form = ~1 | trtgrp))
summary(model3.3.ml.fit)


#------------------------------------------------------------------------
#               Τελευταίο βήμα για το Model 3.2 Β
#         Υπολογισμός των παραμέτρων με εκτίμηση REML 
#           για σωστό υπολογισμό των συνιστωσών διαπσοράς
#------------------------------------------------------------------------
 
#Υπολογίζουμε με χρήση εκτιμητών REML το μεντέλο (προσοχή: η σειρά
#εισαγωγής των παραγόντων στην εξισωση παίζει ρόλο, τα αθροίσματα τετραγώνων
# είναι Type I, δηλαδή ιεραρχικά, το επόμενο λαμβάνει υπόψη το προηγούμενο 
# αλλά αυτό δεν ισχύει αντίστροφα)
# Model 3.3: Final Model.
  model3.3.reml.fit <- lme(weight ~  litsize + sex1 + treatment,
                             random = ~1 | litter, ratpup, method = "REML",
                             weights = varIdent(form = ~1 | trtgrp))
 summary(model3.3.reml.fit)
 intervals(model3.3.reml.fit)
 anova(model3.3.reml.fit)
 
#Μπορούμε να πάρουμε αποτελέσματα για τις διασπορές συνδιασπορές 
#και με τη συνάρτηση getVarCov
   getVarCov(model3.3.reml.fit, individual="27", type="marginal")

#Αποτελέσματα (για βιβλιογραφία)
# Οι συνιστώσες της διασποράς για τους τυχαίους παράγοντες
# υπολογίζονται όπως παρακάτω
# Δίνεται το μέρος των αποτελεσμάτων
#  Random effects:
   #   Formula: ~1 | litter
   #(Intercept)  Residual
   #StdDev:   0.3146374 0.5144324
   #
   #Variance function:
   #   Structure: Different standard deviations per stratum
   #Formula: ~1 | trtgrp 
   #Parameter estimates:
   #   1         2 
   #1.0000000 0.5889108 
# Var(litter) = 0.3146374^2 =0.10
# Var(high/low) = (0.5889108*0.5144324)^2 = 0.09
# Var(Control) = (0.5144324*1.000)^2 = 0.26

# Παράδειγμα αναφοράς αποτελεσμάτων για βιβλιογραφία δίνεται σε ένα φύλλο word



#------------------------------------------------------------------------
#               Διαγνωστικοί έλεγχοι
#              για το βέλτιστο μοντέλοL 
#------------------------------------------------------------------------

   
library(lattice)
trellis.device(color=F)

res <- resid(model3.3.reml.fit)
ratred <- data.frame(ratpup, res) #απλός τρόπος συνένωσης μεταβλητής με 
                                  # πλαίσιο δεδομένων
View(ratred)

## Προσαρμογή των υπολοίπων στην κανονική κατανομή (λαμβάνοντας υπόψη
# τον παράγοντα ανομοιογένειας των διασπορών)
histogram(~res | factor(trtgrp), data=ratred, layout=c(2,1), 
          aspect = 2 , xlab = "Residual") 
   
qqnorm(model3.3.reml.fit, ~resid(.) | factor(trtgrp), layout=c(2,1), 
       aspect = 2, id = 0.05)

by(res,factor(ratpup$trtgrp),shapiro.test)

# Όπως φαίνεται θα χρειαστεί μελέτη για κάποιο μετασχηματισμό ή 
# κάποιες παρατηρήσεις που θα πρέπει να εξαιρεθούν

# Ετεροσκεδαστικότητα
plot(model3.3.reml.fit, resid(.) ~ fitted(.) | factor(trtgrp), 
     layout=c(2,1), aspect=2, abline=0)
# Πράγματι υπάρχουν παρατηρήσεις που είναι παράτυπα σημεία
   
# Υπό συννθήκη υπόλοιπα
   
attach(ratpup)
   
   bwplot(resid(model3.3.reml.fit) ~ ranklit, data=model3.3.reml.fit, 
          ylab="residual", xlab="litter")
# Η ανάγκη για μελέτη των παρατηρήσεων φαίνεται και στο θηκόγραμμα 
# (πιθανός μετασχηματισμός)
   
plot(model3.3.reml.fit)


#------------------------------------------------------------------------



#----------------------------------------------------------------------
#                     Ανάλυση Συνδιακύμανσης
#                         Ancova
#----------------------------------------------------------------------

#Πραγματοποιήθηκε μελέτη για την επίδραση της γήρανσης στην ελαστικότητα δυο
#υλικών S, B. Στο αρχείο «Ancova» συμπεριλαμβάνονται : η αρχική και τελική 
#μέτρηση καθώς και ο παράγοντας υλικό. Να ελεγχθούν οι υποθέσεις:
#  
#  Τα υλικά δε διαφέρουν ως  προς τη ελαστικότητα πριν την επίδραση της 
#γήρανσης (baseline μέτρηση).

#Για κάθε υλικό δεν υπάρχει διαφορά μεταξύ αρχικής και τελικής μέτρησης 
#(Η γήρανση δεν μεταβάλλει την ελαστικότητα των υλικών).
#
#Η διαδικασία της γήρανσης δεν έχει διαφορετική επίδραση σε κάθε υλικό 
#(Η μεταβολή δεν εξαρτάται από το υλικό).
#
#Οι υποθέσεις να εξεταστούν σε στάθμη σημαντικότητας p<0.05

library(haven)

library(Hmisc)
Ancova <- spss.get("Ancova.sav", use.value.labels=TRUE)

names(Ancova)
levels(Ancova$group)
class(Ancova)
#View(Ancova)      # ενολές για να δούμε το πλαίσιο δεδομένων
head(Ancova)
tail(Ancova)
Ancova

#----------------------------------------------------------------------
#                     Περιγραφική στατιστική
#                      Έλεγχοι εγκυρότητας
#----------------------------------------------------------------------
attach(Ancova)
names(Ancova)
# require(Hmisc) # σε περίπτωση που δεν είναι ενεργή
g <- function(x)c(N=length(x),MIN=min(x,na.rm=TRUE),MAX=max(x,na.rm=TRUE),
                  MEDIAN=median(x,na.rm=TRUE), MEAN=mean(x,na.rm=TRUE),
                  SD=sd(x,na.rm=TRUE))
summarize(preer,by=llist(group),g)
summarize(poster,by=llist(group),g)

library(lattice)  # trellis graphics
library(grid)

par(mfrow=c(1,2))
boxplot(preer ~ group, data=Ancova,
        cex.axis=0.7, cex.lab=1.5,
        pch = 1,
        xlab="Group", ylab="Er",
        id=list(labels=rownames(Ancova)))
boxplot(poster ~ group, data=Ancova,
        cex.axis=0.7, cex.lab=1.5,
        pch = 1,
        xlab="Group", ylab="Er",
        id=list(labels=rownames(Ancova)))
par(mfrow=c(1,1))

# Έλεγχος κανονικής κατανομης
by(preer,group,shapiro.test) 
by(poster,group,shapiro.test) 

library(car)
anc1 <- lm(cbind(preer, poster) ~  group, 
           data=Ancova)
# MANOVA ως παράδειγμα, μας ενδιαφέρει η μέθοδος univariate
Manova(anc1)

summary(Anova(anc1), univariate=FALSE, multivariate=TRUE,
        p.adjust.method=TRUE)
# Προετοιμασία για repeated measures (μέθοδος:univariate)
time<-factor(rep(c("pre", "post"), c(1,1)))
idata<-data.frame(time)
ancovaN <- Anova(anc1, idata=idata, idesign= ~ time)
summary(ancovaN,multivariate = FALSE)


# Συμπεραίνουμε ότι παρατηρήθηκε στατιστικά σημαντική διαφοτρά
# μεταξύ των υλικών τόσο στην αρχική στιγμή μέτρησης όσο και μετά το
# τέλος του πειράματος. Επίσης παρατηρήθηκε στατιστικά σημαντική διαφορά 
# μεταξύ αρχικής και τελικής μέτρησης και γαι τα δυο υλικά

# Πολλες φορές συμβαίνει είτε λόγω διαδικασίας τυχαιοποίησης είτε λόγω υλικών
# να υπάρχει διαφορά μεταξύ των ομάδων κατά τη μέτρηση στην αρχή της μελέτης.
# Στην περίπτωση αυτή θα πρέπει αυτό να το λάβουμε υπόψη κατά την ανάλυση.
# Κατάλληλο μοντέλο είναι αυτό της Ανάλυσης Συνδιακύμανσης ANCOVA


#----------------------------------------------------------------------
#                     Ανάλυση Συνδιακύμανσης
#                        Προυποθέσεις
#----------------------------------------------------------------------

#1#Οι μετρήσεις ακολουθούν κανονική κατανομή (για κάθε συνδιασμό των παραγόντων)
# Ελέγχθηκε

#2# Δεν υπάρχει διαφορά μεταξύ των ομάδων ως προς τη διασπορά.

leveneTest(unclass(preer)  ~ group, data=Ancova, center=mean)
leveneTest(unclass(poster) ~ group, data=Ancova, center=mean)

#3# Ομοιογένεια των κλίσεων: θα πρέπει να υπάρχει γραμμική σχέση μεταξύ
# των μετρήσεων πριν και μετά και οι κλίσεις των ευθειών που σχηματίζονται 
# από τα επίπεδα του παράγοντα να ορίζουν δυο σχεδόν παράλληλες ευθείες
# ΔΗΛΑΔΗ να μην υπάρχει αλληλεπίδραση χρόνου και ομάδας


nfd<-Ancova
detach(Ancova)
names(nfd)
attach(nfd)

# Παρατηρούμε σχεδόν παράλληλες ευθείες που διαφερουν στο σταθερό όρο
# Ένδειξη για ομοιγένεια κλίσεων και ύπαρξη διαφοράς στην αρχική μέτρηση
plot(preer,poster,
     pch=16+as.numeric(group),col=c("blue","red")[as.numeric(group)])
abline(lm(poster[group=="S"]~preer[group=="S"]),lty=2,col="blue")
abline(lm(poster[group=="B"]~preer[group=="B"]),lty=2,col="red")
legend(x = "topleft", lty = 2, col = c("blue","red"),
       legend = c("Group S", "Group B"))

# Ο στατιστικός έλεγχος για την υπόθεση της ομοιογένειας των διασπορών
# γίνεται με το πλήρες μοντέλο γραμμικής παλινδρόμησης. Δηλαδή αυτό που ως
# ανεξάρτητες έχει την αρχική μέτρηση (preer), την ομάδα (group) καθώς και την
# αλληλεπίδραση preer*group. Αν η αλληλεπίδραση δεν είναι στατιστικά΄σημαντική
# τότε συμπεραίνουμε ότι ιδχύει η υπόθεση της ομοιγένειας των κλίσεων.

a0<-lm(poster ~ preer*group, data=nfd) 
summary(a0)
anova(a0)
# Πράγματι παρατηρούμε ότι για την αλληλεπίδραση preer:group p=0.5595 > 0.05
# επομένως ισχεύει και η τρίτη προυπόθεση. ’ρα το μοντέλο της Ancova είναι 
# κατάλληλο για να ελέγξει την υπόθεση ότι δεν υπάρχει διαφορά μεταξύ των 
# μέσων όρων των ομάδων κατά τη δεύτερη μέτρηση.
# Στο μοντέλο της Ancova δεν περιλαμβάνεται ο όρος της αλληλεπίδρασης (δεν
# πλήρες γραμμικό μοντέλο)
# Να σημειωθεί ότι στη γραφή "poster ~ 1+preer+group1" που ακολουθεί
# ο όρος 1 αντιστοιχεί στη σταθερά και δεν είναι απαραίτητο να γραφεί, 
# μόνο στην περίπτωση που δε θέλουμε σταθερό όρο γράφουμε -1.

a1<-lm(poster ~ 1+preer+group, data=nfd)
summary(a1)
anova(a1)

# Σϋμφωνα με το αποτέλεσμα της ανάλυσης α1, παρατηρούμε ότι δεν υπάρχει 
# στατιστική σημαντική διαφορά μεταξύ των δυο υλικών μετά τη
# διαδικασία γήρανσης.

# Ερωτήσεις α, β)
t.test(preer ~ group, data=nfd, var.equal = TRUE)
t.test(poster ~ group, data=nfd, var.equal = TRUE)
with(nfd, t.test(preer[group=='S'], poster[group=='S'], paired = TRUE,
                 var.equal = TRUE))
with(nfd, t.test(preer[group=='B'], poster[group=='B'], paired = TRUE,
                 var.equal = TRUE))
###########################################################################

