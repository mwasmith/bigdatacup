# Inspiration from Asmae Toumi's Big Data Bowl submission.

library(skimr)
library(BART)
library(coda)

df_direct_passes <- df_all %>%
  mutate(
    Player = as.factor(Player),
    Player.2 = as.factor(Player.2),
    Team = as.factor(Team),
    passer_zone = as.factor(passer_zone),
    passer_side = as.factor(passer_side),
    receiver_zone = as.factor(receiver_zone),
    receiver_side = as.factor(receiver_side),
    transition_zone = as.factor(transition_zone),
    transition_side = as.factor(transition_side),
    p = as.factor(p),
    position.x = as.factor(position.x),
    position.y = as.factor(position.y),
    shoots.x = as.factor(shoots.x),
    shoots.y = as.factor(shoots.y),
    pass_same_side = case_when(shoots.x == shoots.y ~ 1, TRUE ~ 0)
    ) %>%
  drop_na()

# prepping data for BART
drop_cols <- c("game_date", "Home.Team", "Away.Team", "Period", "Clock", 
               "Home.Team.Skaters", "Away.Team.Skaters", "Home.Team.Goals", 
               "Away.Team.Goals", "Team", "Detail.1", "Detail.2", "Detail.3", 
               "Detail.4",  "Event","pass_success_binary","x", "y","passer_zone",
               "passer_side", "receiver_zone", "receiver_side","transition_zone",
               "transition_side","shoots.x","shoots.y","age.x", "age.y", "p", "n", 
               "Player", "Player.2", "sequence", "distance", "BirthYear")

y <- data.matrix(df_direct_passes$pass_success_binary)
x <- data.matrix(df_direct_passes[,!colnames(df_direct_passes) %in% drop_cols])

# BART
bart_fit1 <- lbart(x.train = x, 
                   y.train = y, 
                   sparse = TRUE, 
                   ndpost = 500, 
                   nskip = 2500, 
                   keepevery = 5, 
                   printevery = 500)

saveRDS(bart_fit1, file = "bigdatacup/data/BART_pass/bart_fit1.RDS")

bart_fit2 <- lbart(x.train = x, 
                   y.train = y, 
                   sparse = TRUE, 
                   ndpost = 500, 
                   nskip = 2500, 
                   keepevery = 5, 
                   printevery = 500)

saveRDS(bart_fit2, file = "bigdatacup/data/BART_pass/bart_fit2.RDS")

prob_train1 <- bart_fit1$prob.train
prob_train2 <- bart_fit2$prob.train
prob_train <- rbind(prob_train1, prob_train2)

# posterior means
prob_means <- apply(prob_train, mean, MAR=2)

# 95% CI
ci.fun <- function(a){
  c(quantile(a,.025),quantile(a,.975))
}

prob.ci = apply(prob_train, 2, ci.fun)

low_bound = prob.ci[1,] 
upp_bound = prob.ci[2,]

trained_pass <- cbind(df_direct_passes, prob_means, low_bound, upp_bound)
trained_pass <- trained_pass %>% mutate(Position = position.x, Age = age.x, BirthYear = BirthYear.x)

save(trained_pass, file = "bigdatacup/data/BART_pass/trained_pass.RData")

# Variable selection
varcount <- rbind(bart_fit1$varcount, bart_fit2$varcount)
varprob <- rbind(bart_fit1$varprob, bart_fit2$varprob)

varcount_mean <- colMeans(varcount)
varcount_sd <- apply(varcount, FUN = sd, MARGIN = 2)

comp_prob <- trained_pass %>%
   mutate(Position = position.x) %>%
   group_by(Team, Player, Age, Position) %>%
   summarise(npasses = n(),
             pass_pct = mean(pass_success_binary),
             pass_pred = mean(prob_means),
             cpoe = (pass_pct - pass_pred)) %>% 
   filter(npasses >= 20) %>%
   arrange(-cpoe)

write.csv(comp_prob,"bigdatacup/data/comp_prob.csv")

comp_var <- trained_pass %>%
  mutate(Position = position.x, prob_int = cut(prob_means*100,breaks =c(0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100))) %>% 
  group_by(Team, Player, Age, Position, prob_int) %>%
  summarise(npasses = n(),
            pass_pct = mean(pass_success_binary),
            pass_pred = mean(prob_means),
            cpoe = sum(pass_success_binary - prob_means)) %>% 
  filter(npasses >= 5) %>%
  arrange(prob_int, -cpoe)

write.csv(comp_var,"bigdatacup/data/comp_var.csv")
