> boxplot(WHO$Over60 ~ WHO$Region)
> boxplot(WHO$Over60 ~ WHO$Region,xlba = "",ylab = "qqq",mina = "aaa")
> tapply(WHO$Over60, WHO$Region,mean)
Africa              Americas 
5.220652             10.943714 
Eastern Mediterranean                Europe 
5.620000             19.774906 
South-East Asia       Western Pacific 
8.769091             10.162963 
> tapply(WHO$LiteracyRate, WHO$Region,min, na.rm = TRUE)
Africa              Americas 
31.1                  75.2 
Eastern Mediterranean                Europe 
63.9                  95.2 
South-East Asia       Western Pacific 
56.8                  60.6 
> tapply(WHO$ChildMortality, WHO$Region, mean  )