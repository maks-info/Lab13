# renv::init() # инициализация виртуального окружения
# renv::install("devtools") # установка библиотеки из CRAN
# renv::snapshot() # делаем снимок версий библиотек в нашем виртуальном окружении
# фиксируем этот список в .lock-файле для возможности восстановления
# renv::restore() # команда отктиться к предыдушему удачному обновления библиотек

# ------------------- 
# Лабораторная работа №13:
# Однослойные нейронные сети. Nnet, Neuralnet.


#  –аспознавание сорта оливкового масла

floor <- read.csv(file = "iris.csv", header = TRUE, sep=",")

#  Для стандартизации переменных для каждого столбца 
#  находим минимальное значение и размах
a <- sapply(floor[ , 1:4], min)
b <- sapply(floor[ , 1:4], max) - a
a
#  —обственно стандартизаци€ входных переменных
floor.x <- scale(floor[, 1:4], center=a, scale=b)

#  Преобразование выходной переменной в три столбца, в три индикаторные переменные.
y1 <- rep(0, nrow(floor))
y1[floor[ , 5]=="Setosa"] <-1

y2 <- rep(0, nrow(floor))
y2[floor[ , 5]=="Versicolor"] <-1

y3 <- rep(0, nrow(floor))
y3[floor[ , 5]=="Virginica"] <-1

z.1 <- as.data.frame(cbind(floor.x, y1, y2, y3))
z.1


# создание тестовой и обучающей выборки
set.seed(12345)
index <- sample(1:nrow(z.1), round(nrow(z.1)*2/3), replace=F)   # пропорция 2/3 на 1/3
z.train <- z.1[index,]
z.test <- z.1[-index,]


#  Подключаем библиотеку neuralnet.
install.packages("neuralnet")
library(neuralnet)

n <- names(z.1)
n


# хотим хранить лучшую сеть и значит датчик зерна
# зерно датчика случайных чисел
# дл€ лушей сети надо охранить значение критериz качества, еcли процент ошибок, то не только лучшую сеть, но и процент ошибок у этой лучшей сети
# вначале проент ошибок на обучающем множестве
# почти наверн€ка это не здорово, нас волнует на тестовом множестве
# у нас тут сумма трех Y

#задаем число сетей
num.nets <- 10

# начальный датчик cлуч чисел
seed.start <- 12345

# переменная которая хранит лучшую сеть
# nnet.nest

# нам нужно лучшее зерно датчика случ чисел, которое соответсвует наилучшей сети
# seed.best

# мне нужна одна переменная , хранящая процент ошибок по наилучшей ЌC (нулева€ Ќ— ошибаетс€ всегда, так как единица)
error.best <- 1

# но лучше накопить статистику по всем ошибкам по всем Ќ—
error.vector <- rep(-9999, num.nets)
seed.current <- seed.start


# когда строим одну сеть, то будут ошибки при одном нейроне
for (i in 1: num.nets){ # строю текущую сеть
  
  seed.current <- seed.current + 1
  set.seed(seed.current)
  nn.temp <- neuralnet( y1+y2+y3 ~ sepal.length + sepal.width + petal.length + petal.width ,
                        data=z.train, hidden = c(3,2), linear.output=F)
  # считаем процент ошибок длz текущей i—
  res.z <- compute(nn.temp, z.train[, 1:6] )  # это матрица веро€тностей предсказани€ по Ќ— те знач, которые получаютс€ после Ќ—
  res.z2 <- apply(res.z$net.result, 1, which.max )
  error.temp <- sum(res.z2 != floor[index,1] )/length(index) # без lenght(index) было бы просто это количесвто ошибок, а нам нужен процент ошибок по всем видам, кол-во раз когда не попало
  error.vector[i] <- error.temp
  if (error.temp < error.best)
  {
    nn.best <- nn.temp
    error.best <- error.temp
    seed.best <- seed.current
  }
}

plot(nn.best)

error.vector
#  [1] 0.454068241470 0.000000000000 0.005249343832 0.000000000000 0.000000000000
#  [6] 0.000000000000 0.000000000000 0.000000000000 0.000000000000 0.000000000000
error.best
# [1] 0
seed.best
# [1] 12347  - 3 i

# таблица сопряженности для лучшей i—
res.3 <- compute(nn.best, z.train[, 1:6] )    # обучающая выборка
res.z3 <- apply(res.3$net.result, 1, which.max)
table(res.z3, floor[index,1])
#  res.z3   1   2   3
#  1   215      0   0
#  2    0      70   0
#  3   0        0   96
res.4 <- compute(nn.best, z.test[, 1:6] )     # тестовая выборка
res.z4 <- apply(res.4$net.result, 1, which.max )
table(res.z4, floor[-index,1])
#  res.z4   1   2   3
#  1       108  0   0
#  2        0  28   0
#  3        0   0  55
sum(diag(table(floor[-index,1], res.z4)))/length(floor[-index,1])*100
#  [1] 97.38219895
100 - (sum(diag(table(floor[-index,1], res.z4)))/length(floor[-index,1])*100)
#  [1] 2.617801047