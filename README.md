## CLAMIs

* CLAMI is for an unsupercised defect prediction. **CLAMIs** has six versions of CLAMI. (CLA, CLA+, CLAMI, CLAMI+, CLABI, CLABI+) 



## How to run 

Refer to 'help' option. 

- On Linux/Mac, see Help
  - ./CLAMI -h

- On Windows, see Help
  - CLAMI.bat -h



## Data Analysis Tool 

You can find *Tools* class fom [here](https://github.com/ISEL-HGU/CLAMIs/blob/CLAMI_ALL/src/main/java/net/lifove/clami/Tools.java). The final purpose of data analysis is to determine whether the dataset is suited to our approach such as CLA/CLAMI etc. 

* **selectPercentileCutoff** 
  * This is to find the best percentile cutoff of data. It iterates each clustering process of our approach by increasing 5 for cutoff (cutoff range: from 10 to 100). 



* **ksTest** 
  * ksTest is Kolmogorov Smirnov Test which tries to select the best metrics. We want to find the best metric group in which the relationship between metrics is stable and the tendency of metric and class label is reliable through ksTest. 
  * For all attributes of data, we made every combination (ex. m1-m1, m1-m2, ... mn-mn). And compute the *p-value* with two metrics. If *p-value* is larger than the critical value, select the corresponding metric combination. Otherwise, remove the metric, and finally generate a new instance. Then we can get the individual group up to the number of metrics. By running the 'clustering' of CLA for all generated instances, we can get the predicted label for all instances. Then count the number of the predicted label as buggy for each attribute.    

## 

## Data Feature 

[Data feature (factor)](https://github.com/ISEL-HGU/CLAMIs/tree/CLAMI_ALL/src/main/java/net/lifove/clami/factor) is data selection approach to determine whether the data is acceptable to CLA or other versions through data factor value. You can find GIR which is *number of groups / number of instances*. 

