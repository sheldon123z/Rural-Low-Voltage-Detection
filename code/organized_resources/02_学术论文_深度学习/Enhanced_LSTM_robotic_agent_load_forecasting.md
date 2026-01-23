---
title: Frontiers | Enhanced LSTM-based robotic agent for load forecasting in low-voltage distributed photovoltaic power distribution network
source: https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1431643/full
scraped_date: 2026-01-20 10:28:16
---

Your new experience awaits. Try the new design now and help us make it even better

Switch to the new experience




ORIGINAL RESEARCH article

Front. Neurorobot., 11 July 2024

Volume 18 - 2024 | <https://doi.org/10.3389/fnbot.2024.1431643>

This article is part of the Research TopicAdvancing Neural Network-Based Intelligent Algorithms in Robotics: Challenges, Solutions, and Future Perspectives[View all 17 articles](https://www.frontiersin.org/research-topics/62762/advancing-neural-network-based-intelligent-algorithms-in-robotics-challenges-solutions-and-future-perspectives/magazine)

# Enhanced LSTM-based robotic agent for load forecasting in low-voltage distributed photovoltaic power distribution network

[![
Xudong Zhang](https://loop.frontiersin.org/images/profile/2672927/74)Xudong Zhang](https://loop.frontiersin.org/people/2672927)1![Junlong Wang](https://loop.frontiersin.org/cdn/images/profile/default_32.jpg)Junlong Wang1![Jun Wang
](https://loop.frontiersin.org/cdn/images/profile/default_32.jpg)Jun Wang2*![Hao Wang](https://loop.frontiersin.org/cdn/images/profile/default_32.jpg)Hao Wang2![Lijun Lu](https://loop.frontiersin.org/cdn/images/profile/default_32.jpg)Lijun Lu2

  * 1State Grid Hebei Electric Power Company, Shijiazhuang, China
  * 2Henan XJ Metering Co., Ltd, Xuchang, China



To ensure the safe operation and dispatching control of a low-voltage distributed photovoltaic (PV) power distribution network (PDN), the load forecasting problem of the PDN is studied in this study. Based on deep learning technology, this paper proposes a robot-assisted load forecasting method for low-voltage distributed photovoltaic power distribution networks using enhanced long short-term memory (LSTM). This method employs the frequency domain decomposition (FDD) to obtain boundary points and incorporates a dense layer following the LSTM layer to better extract data features. The LSTM is used to predict low-frequency and high-frequency components separately, enabling the model to precisely capture the voltage variation patterns across different frequency components, thereby achieving high-precision voltage prediction. By verifying the historical operation data set of a low-voltage distributed PV-PDN in Guangdong Province, experimental results demonstrate that the proposed “FDD+LSTM” model outperforms both recurrent neural network and support vector machine models in terms of prediction accuracy on both time scales of 1 h and 4 h. Precisely forecast the voltage in different seasons and time scales, which has a certain value in promoting the development of the PDN and related technology industry chain.

## 1 Introduction

Load forecasting of the power distribution network (PDN) is an important link in safe operation and dispatching control. With the popularization and application of energy storage technology and the addition of new dispatchable resources such as electric vehicles, a large number of interruptible and bidirectional loads appear on the load side (Dairi et al., 2020; Razavi et al., 2020; Markovics and Mayer, 2022). These load's randomness and distributed access characteristics affect the power system regulation of the PDN. Active distribution network (ADN) uses the core technology of demand response to dynamically adjust the price of electricity and incentive policies and flexibly manage and control the original load demand of users. Furthermore, it actively guides users to participate in the optimization of power dispatching to enhance the synergy and complementarity of multiple loads. It not only considers users' satisfaction with electricity consumption but also improves the consumption ratio of distributed renewable energy (Hafiz et al., 2020; Mellit et al., 2021).

Proper planning and useful applications of load forecasting of the PDN require specific “predicting intervals”. According to the delivery cycle, load forecasting can be divided into ultra-short-term, short-term, medium-term, and long-term (Eom et al., 2020). Ultra-short-term forecasting is employed for real-time control, enabling rapid adjustments to generation and load to ensure the safe and stable operation of the power grid. Short-term forecasting is widely employed in the daily operations of the utility industry, facilitating dispatch of generation and transmission, optimizing grid resource allocation and enhancing grid operational efficiency. Medium-term forecasting is primarily utilized to forecast load variations over the next few months to a year, providing valuable insights for fuel procurement, maintenance planning and grid investment decisions. Long-term forecasting focuses on load growth trends over the next 1 to 20 years, employed to forecast the need for new power plants, grid planning and providing strategic guidance for power system development.

Load forecasting of the PDN is complex for engineers and academics, and remains an ongoing area of research. Moreover, the thorough exploration of load-side controllable resources to achieve optimal dispatch of the power system by the grid has emerged as a critical research priority for contemporary power utilities. Nowadays, it is more and more common for low-voltage PDNs to adopt distributed photovoltaic (PV) access. On this basis, considering the regularity of PV power generation, the problem of voltage fluctuation can be solved by predicting the voltage variation trend.

Accurate load forecasting plays a crucial role in optimizing the scheduling and management of power resources, effectively reducing operational costs and enhancing the overall efficiency of the power system. With the rapid development of deep learning-based robotic agent technology (Ma et al., 2023, 2024a), the application of deep learning in load forecasting has gained significant attention, particularly for approaches based on recurrent neural networks (RNN). Furthermore, deep learning techniques can handle complex nonlinear relationships and massive datasets, thereby improving the accuracy and reliability of predictions, which are paramount for the stable operation of the power grid. Deep learning models, however, demand substantial data and computational resources, while their hyperparameter tuning and training process necessitate specialized knowledge and expertise. The nonlinearity and time dependence of load data increase the complexity of predictions. As the data dimensions increase, deep learning models need to possess enhanced learning capabilities, thereby avoiding overfitting and performance degradation. While significant progress has been made in load forecasting techniques, several challenges remain that require further attention to enhance the accuracy and efficiency.

To address the specific scenario of load forecasting in low-voltage distributed photovoltaic power distribution networks, we customized a load forecasting model and employed a long short-term memory (LSTM) network architecture for forecasting. To enhance feature extraction, we placed a fully connected layer, denoted as dense layer, after the LSTM layer. Additionally, we integrated the frequency domain decomposition (FDD) method to obtain the amplitude and phase of each frequency component, and utilized LSTM to individually forecast low-frequency and high-frequency components, ultimately improving the model's accuracy. This study is expected to offer a new idea for the low-voltage distributed PV-PDN to meet the forecast. The contributions of this paper can be summarized as follows:

1) FDD-enhanced LSTM for load forecasting in PV-PDN: to address the load forecasting of low-voltage distributed PV-PDN, we propose a novel FDD-enhanced LSTM model. The proposed model outperforms conventional support vector machine (SVM) and RNN models, particularly in long-term forecasting scenarios. This method represents a significant advancement in the application of deep learning techniques in the distribution network domain, providing a novel approach to enhance grid reliability and operational efficiency.

2) A new benchmark for load forecasting in PV systems: the integration of FDD and LSTM networks has revolutionized load forecasting in low-voltage distributed PV systems, establishing a new benchmark for forecasting methodologies in distributed PV systems.

3) Comparative analysis of FDD-enhanced LSTM for load forecasting: to objectively evaluate the performance of the enhanced LSTM model in complex low-voltage distributed PV forecasting scenarios, we conducted a comprehensive comparative analysis of the mean absolute error (MAE) across different time scales. The results demonstrate the model's superior performance and reliability in complex voltage forecasting environments.

The rest of the paper is organized as follows: Section 2 reviews the related work of load forecasting and scene image monitoring analysis. Section 3 describes the proposed methods in detail. Section 4 reports the experimental result and analysis. Section 5 represents the conclusion and future work.

## 2 Related work

Low-voltage load forecasting is an intelligent technique that utilizes historical load data, weather information and socioeconomic factors to forecast future low-voltage load levels. This technique possesses extensive application value in power grid scheduling, grid planning and electricity pricing.

Statistical and time series methods are widely employed techniques for short-term load forecasting, with linear models being the most prevalent approach. Linear models typically employ linear parameters. Litjens et al. (2018) have utilized some of the simplest linear models, including seasonal persistence models and simple average models, often in conjunction with meteorological data. Borges et al. employed linear models with varying feature subsets for short-term load forecasting and missing data imputation in substation data (Borges et al., 2020). Their model utilized historical load data, meteorological data and neighboring substation data. While standard linear regression has proven successful in demand forecasting for all levels of low-voltage networks, nonlinear regression models have also gained attention due to their inherent flexibility. Hayes et al. employed a nonlinear autoregressive exogenous (NARX) model for smart meter load forecasting and demonstrated its superior performance compared to traditional NARX models and neural network models (Hayes et al., 2014). Tsekouras et al. (2007) employed nonlinear multiple regression, selecting a model based on testing various combinations of nonlinear functions for mid-term load forecasting. Nonlinear models, despite their wide applicability, are susceptible to overfitting issues.

Among time series forecasting models, ARIMA stands out due to its exceptional performance and has been widely adopted across various applications (Marinescu et al., 2013). Researchers have successfully integrated online ARIMA models into short-term forecasting of electricity systems in public school buildings (Lee et al., 2013). Leveraging historical load and temperature data, this model effectively captures energy efficiency, forecasts energy consumption and detects anomalies in energy usage. Furthermore, Espinoza et al. proposed a unified modeling framework based on periodic autoregressive models, enabling the effective integration of data from multiple entities to achieve load curve forecasting and clustering analysis (Espinoza et al., 2005).

With the continuous advancement of deep learning (Ma et al., 2021, 2024b; Li et al., 2023; Jin et al., 2024; Liufu et al., 2024), deep learning-based load forecasting has also gained widespread attention from researchers. Deep learning-based load forecasting methods, with their ability to capture complex data patterns and extract deep-level features, have gradually become a research hotspot in the field of power load forecasting and have achieved remarkable results. Shivam et al. (2021) discuss a predictive energy management strategy for residential PV-battery systems using RNN model, it has a deep inner hidden layer, which imitates the neural network inside humans to think like the human brain. Luo et al. (2021) enhance photovoltaic power generation forecasting by incorporating domain knowledge into deep learning models (Kim et al., 2020). The limitation of machine learning (ML) lies in the need for more learning ability for high-dimensional data. The purpose of representative learning is to simplify complex original data, remove invalid or redundant information from original data, and refine effective information to form features. The purpose of representative learning is to simplify complex raw data, remove redundant or invalid information from the data, and extract effective information to form features. In addition, SVM and LSTM have been widely used in load forecasting. Kabilan et al. (2021) and Feng et al. (2020) both employ machine learning models for short-term power prediction and quantifying daily global solar radiation, respectively, highlighting the potential of computational methods in optimizing and accurately forecasting solar energy production. Kim et al. (2020) focus on very-short-term photovoltaic forecasting for smart city energy management through multiscale LSTM-based deep learning.

In the realm of load forecasting, traditional methods have often faced limitations in capturing the intricate patterns and underlying relationships within complex electricity consumption data. To address these shortcomings, we propose a novel deep learning-based load forecasting framework that leverages the powerful capabilities of RNN and LSTM cells to effectively capture temporal dependencies.

## 3 Methods

### 3.1 Features of distributed PV-PDN

PV power generation is essentially a power technology that uses the photoelectric or photochemical effect of PV modules (semiconductor materials) to convert light energy directly into electric energy. Distributed PV power station usually refers to a power generation system with a small installed scale that uses distributed resources and is located near the user. Ordinarily, the power grid with a voltage level of <35 kV or lower is connected. The heart of a PV facility is solar panels. The semiconductor materials adopted for power generation principally cover polysilicon, monocrystalline silicon, amorphous silicon and cadmium telluride (Lopes et al., 2022). Solar panels are the core and most valuable part of a solar power system. Its role is to convert the radiant power of the sun into electrical energy, feed it into a storage battery or promote load operation. The function of the solar controller is to control the working state of the entire system and protect the battery from overcharge and discharge (Alipour et al., 2020; Korkmaz, 2021; Qadir et al., 2021). Qualified controllers should also have a temperature compensation function in places with large temperature differences.

The PV cell's equivalent circuit (EC) is shown in Figure 1. _I_ _ph_ and _I_ _d_ refer to the photo-generated and diode junction currents; _C_ _j_ means the junction capacitance (negligible); _R_ _s_ and _R_ _sh_ stand for series and parallel resistors. Typically, distributed PV projects have a capacity of within a few kilowatts. Unlike centralized plants, the scale of PV plants has little effect on power generation efficiency. Therefore, its influence on the economy is also tiny, and the return on investment of small PV systems will not be lower than that of large ones.

Figure 1

[ ![www.frontiersin.org](https://www.frontiersin.org/files/Articles/1431643/fnbot-18-1431643-HTML-r1/image_m/fnbot-18-1431643-g001.jpg) ](https://www.frontiersin.org/files/Articles/1431643/fnbot-18-1431643-HTML-r1/image_m/fnbot-18-1431643-g001.jpg)

**Figure 1**. EC of PV cell.

Solar energy's direct output is generally 48 VDC, 24 VDC and 12 VDC. To power an appliance at 220 VAC, direct current (DC) generated by a solar power system needs to be converted into alternating current (AC). To avoid power backflow, it is necessary to configure an anti-flow device for alarm, and the inverter then adjusts its capacity according to the received signal. To connect the distributed PV system to the PDN, it first needs to output the PV cells through the DC/DC converter, then connected to the DC/AC inverter, and next connected to the external PDN. Taking a household small distributed PV system as an example, the typical grid-connected PV structure is displayed in Figure 2. The grid-connected access information acquisition system of small distributed PV power stations is applied to transmit the collected information to the monitoring platform and display it to users or power grid enterprises intuitively and clearly. This can provide grid enterprises with grid-connected data of PV power stations, eliminate the “blind adjustment” phenomenon of PV power generation, assist power grid operation analysis and decision-making, and promote the operation of the power grid safe and stable (Karimi et al., 2020; Ding et al., 2021; Khan et al., 2022).

Figure 2

[ ![www.frontiersin.org](https://www.frontiersin.org/files/Articles/1431643/fnbot-18-1431643-HTML-r1/image_m/fnbot-18-1431643-g002.jpg) ](https://www.frontiersin.org/files/Articles/1431643/fnbot-18-1431643-HTML-r1/image_m/fnbot-18-1431643-g002.jpg)

**Figure 2**. Grid-connected structure of household small distributed PV system.

The overall diagram of the method in this study is shown as Figure 3. The method involves meticulous data collection and preprocessing to ensure high-quality inputs, followed by strategic feature selection via the XGBoost algorithm to optimize data relevancy. Then an advanced LSTM model is designed and refined, augmented with FDD, for enhanced predictive accuracy.

Figure 3

[ ![www.frontiersin.org](https://www.frontiersin.org/files/Articles/1431643/fnbot-18-1431643-HTML-r1/image_m/fnbot-18-1431643-g003.jpg) ](https://www.frontiersin.org/files/Articles/1431643/fnbot-18-1431643-HTML-r1/image_m/fnbot-18-1431643-g003.jpg)

**Figure 3**. The overall diagram of the method in this study.

### 3.2 Voltage data preprocessing and feature selection

As a kind of clean energy, the high proportion of PV connected to a low-voltage PDN will bring huge power generation benefits. However, due to its own uncertainty, it may bring a series of problems to the stable and safe operation of the PDN, such as voltage over the limit, line overload and power quality reduction. Thus, it is essential to accurately evaluate the acceptance capacity of PV in a low-voltage PDN. More importantly, to further improve the benefits of PV power generation, it is urgent to improve the acceptance capacity of distributed PV based on accurate assessment (Rana and Rahman, 2020). Before voltage prediction of distributed PV-PDN, data mining and preprocessing should be carried out, ensuring that it is in a suitable form for analysis. This step involves removing outliers, handling missing values, and normalizing data, which helps reduce variability and improve the model's accuracy. It also includes feature selection and transformation to identify and utilize the most relevant information for forecasting, thereby enhancing the prediction model's effectiveness and efficiency.

The missing data is filled using the cubic spline interpolation fitting function _f_ θ(_x_), and the equation for filling the value is Equation (1):

D(tmiss)=fθ(tmiss) (1)

_t_ _miss_ indicates the time point at which load data is missing.

For data satisfying normal distribution, standardized methods are used for dimensionless processing, with the specific equation as follows Equation (2):

x*=x−X¯S (2)

_x_ and _x_ * refer to the original and the processed feature data, respectively; X¯ and _S_ represent the mean and standard deviation of the feature, respectively.

DL model has advantages in capturing power voltage fluctuation in distributed PV voltage prediction due to their ability to model complex, nonlinear relationships within large datasets. They excel in identifying patterns and dependencies in temporal data, such as those found in voltage series, by leveraging multiple layers of processing. This capability allows DL models to provide more accurate and reliable forecasts of voltage fluctuations, which is essential for maintaining grid stability and optimizing energy distribution in distributed PV-PDN. At this time, dimensionless standardization of different power characteristics can significantly accelerate the optimization speed of the gradient descent algorithm. The maximum and minimum rescaling method of voltage and power is illustrated in Equations (3) and (4):

v*=v−vmaxvmax−vmin (3)

p*=p−pmaxpmax−pmin (4)

_v_ & _p_ and _v_ * & _p_ * represent time-series raw data and dimensionless data for voltage and power; _v_ max & _v_ min and _p_ max & _p_ min refer to the voltage data's and power data's maximum and minimum values, respectively.

Generally speaking, the power load is filled with data of similar size. Because the power load has a certain periodicity, it can be filled and replaced with similar load data of the same cycle. The power load has a regular periodicity, that is, the data of different periods at the same time should be very different. If the difference between the two data exceeds the threshold, the vertical method can be used for processing. For the PV system, the light intensity in winter is lower than that in summer, and the maximum light intensity is usually at noon, so the voltage will rise in this period. It can be seen that the time feature vector is very vital in the voltage prediction process, and it is a key factor in improving the prediction accuracy.

Considering various types of features in the voltage prediction process, this study will adopt the feature selection method based on the Extreme Gradient Boosting (XGBoost) algorithm (Bae et al., 2021), a method chosen for its efficiency and effectiveness in handling high-dimensional data. XGBoost is renowned for its ability to improve model performance by selecting the most relevant features, reducing noise and preventing overfitting. This approach aids in identifying the key predictors of voltage fluctuations in distributed PV systems, thus enhancing the predictive accuracy of the deep learning model. In the course of multiple iterations, the probability distribution (PD) of the training data used in the current iteration will be regulated based on the results of the previous iteration. That is to say, each sample of training data has a weight, which itself will be adjusted with iteration. As suggested in Figure 4, Dm is the training dataset's PD. In the first iteration, the classification error of basic classifier C1 is employed to adjust D2; In the second iteration, the base classifier C2 is used for iteration D3, and so on.

Figure 4

[ ![www.frontiersin.org](https://www.frontiersin.org/files/Articles/1431643/fnbot-18-1431643-HTML-r1/image_m/fnbot-18-1431643-g004.jpg) ](https://www.frontiersin.org/files/Articles/1431643/fnbot-18-1431643-HTML-r1/image_m/fnbot-18-1431643-g004.jpg)

**Figure 4**. Training flow of XGBoost algorithm.

XGBoost is the use of multiple base learning. Each base learning is relatively simple. To prevent overfitting, the next learning is the result of learning the previous base learning. The loss function of XGBoost algorithm reads Equations (5) and (6):

L=∑i=1nl(yi, yi^)+∑m=1MΩ(bm) (5)

Ω(bm)=γT+12λ‖w‖2 (6)

_n_ refers to the number of samples; _y_ _i_ and yi^ represent the label value of the i-th sample and output value predicted by the model, respectively; _l_ means the squared error function; Ω(_b_ _m_) expressed a regularized term for the tree model. _T_ displays the leaf nodes' quantities for a single tree model; _w_ signifies the output vector of the leaf node; γ are λ parameters that control the weights of regularized terms.

After the model is initialized, it needs to carry out M-round cycle calculation, so the objective function _Obj_(_t_) should be minimized during the t-round calculation Equations (7) and (8):

Obj(t)=∑i=1nl(yi, yi^(t−1)+bt(xi))+Ω(bt)+C (7)

C=∑i=1t−1(bi) (8)

_b_ _t_ represents tree model in the t-round training; yi^(t-1) denotes the predicted output value of the model obtained from the previous round; Ω(_b_ _t_) indicates the complexity of the tree model obtained in t-round; _C_ is a constant.

When solving the objective function of a binary tree, it is necessary to know the first-order and second-order derivatives of the loss function, and on which leaf node the sample is located. It is also necessary to find the first and second derivatives of the sample at each leaf node to find the objective function. In this way, it is possible to decide whether to split the node and according to the characteristic values of which node to split.

The voltage, power of key nodes and time characteristics of prediction points in PDN are selected and taken as input feature vector _x_ after series. When forecasting, the higher the prediction accuracy of 1 h ago, the higher the multiple time scales' prediction accuracy. The dimensionless node voltage and net power data of the complete PDN are obtained through data preprocessing. The voltage eigenvector _V_ _i_ of the node, the net power eigenvector _P_ _i_ and the corresponding label _y_ _i_ are obtained as follows Equations (9–11):

Vi=[vi+t-H,...,vi+t-2,vi+t-1] (9)

Vi=[pi+t-H,...,pi+t-2,pi+t-1] (10)

yi=vi+t (11)

_H_ represents the length of the sliding window, _V_ _i_ and _V_ _i_ are eigenvectors with dimension _H_.

The time variable of discretization is processed by unique thermal coding. Time eigenvector _T_ _i_ is constructed to predict time points. Finally, the input feature vector _x_ _i_ of the i-th sample can be expressed as Equation (12):

xi=[Vi,Pi,Ti] (12)

_x_ _i_ and _y_ _i_ together constitute the training sample set of the XGBoost algorithm, which can be written as Equation (13):

{(xi,yi)}i=1n (13)

Additionally, XGBoost suggests two ways to avoid overfitting. The first is Shrinkage, namely, the learning rate. In each tree iteration, each leaf node's weight is multiplied by a reduction coefficient. This way, the impact of each tree will not be too large, leaving more space for optimization for the trees below (Wang et al., 2017; Liu et al., 2022). Another way is Column Subsampling, which is similar to random forest selection for tree construction. There are two methods: (1) Random sampling by layer. Before splitting nodes of the same layer, some eigenvalues are randomly selected for traversal to calculate information gain (IG); (2) Some eigenvalues are randomly sampled before building a tree. Then the tree's all-node splits traverse these eigenvalues to compute IG.

The Mean Absolute Error (MAE) to validate the performance of prediction methods, which is an objective function used to measure the average absolute difference between predicted and true values in regression problems. It can measure the average error size between predicted values and true values, and has good robustness. The calculation formula for MAE is written as Equation (14):

MAE=1N∑i=1N|yi-yi^| (14)

_N_ represents the number of samples, _y_ _i_ is the true value, and yi^ is the predicted value.

The smaller the value of MAE, the smaller the average difference between the predicted value and the true value, indicating higher accuracy of the prediction.

### 3.3 Load forecasting of distributed PV system based on FDD + LSTM

In the context of distributed photovoltaic systems, load forecasting necessitates a multifaceted analytical approach. Key is the scrutiny of historical data to discern patterns and trends. Employing statistical methods, such as time series analysis, facilitates the understanding of complex data interrelations. Moreover, the application of machine learning algorithms, including neural networks, is essential for improving prediction accuracy given the nonlinear nature of load data. Selecting pertinent features, particularly those influenced by weather and temporal factors, is critical. Additionally, integrating renewable energy sources, notably solar power, introduces unpredictability, demanding innovative, adaptable forecasting techniques to ensure consistent power distribution.

FDD refers to taking the Fourier transform (FT) of the signal to analyze it. FT is a mathematical equation that relates a signal sampled in space or time to the same signal sampled at frequency (Polo et al., 2023). In signal processing, FT can reveal a signal's vital characteristics (i.e., its frequency component). For a vector x containing n uniform sampling points, FT is defined as Equation (15):

yk+1=∑j=0n-1ωjkxj+1 (15)

ω is one of the _n_ complex roots of unity; For _x_ and _y_ , indexes _j_ and _k_ range from 0 to _n_ − 1.

The Fourier analysis method is extended to aperiodic signals, and FT is introduced. When the period of a periodic signal increases infinitely, the frequency spectrum tends to become infinitely small and cannot be represented by the Fourier series. But from a physical point of view, the spectrum is still there. FT spectrum analysis divides PV power into load forecasting and high frequency components (Liu et al., 2020; Zang et al., 2020; Rai et al., 2021). The low-frequency component represents the conventional part of PV performance, which can be accurately predicted and indicates the trend characteristics. The high-frequency component exhibits the randomness of PV power and the fluctuation characteristics affected by weather and other factors, which is relatively difficult to predict. Figure 5 presents the correlation between low-frequency and high frequency components and PV power. When FDD is performed on PV power data, the more frequency is selected, the weaker the correlation between high frequency component and PV power is. However, the correlation between low-frequency component and photovoltaic power is stronger. Table 1 compares the predicted results of the two frequency components at different frequencies. The selection of frequency boundary points is based on frequency nodes with larger amplitude in the amplitude spectrum. It can be found that the core of frequency demarcation point selection is that the frequency selected by the low-frequency component should be as high as possible. Thereby, the low-frequency component accounts for more, and it is necessary to ensure that the frequency of the high frequency component is not too high, thus avoiding excessive difficulty in prediction.

Figure 5

[ ![www.frontiersin.org](https://www.frontiersin.org/files/Articles/1431643/fnbot-18-1431643-HTML-r1/image_m/fnbot-18-1431643-g005.jpg) ](https://www.frontiersin.org/files/Articles/1431643/fnbot-18-1431643-HTML-r1/image_m/fnbot-18-1431643-g005.jpg)

**Figure 5**. Correlation between high frequency and low-frequency components and PV power.

Table 1

[ ![www.frontiersin.org](https://www.frontiersin.org/files/Articles/1431643/fnbot-18-1431643-HTML-r1/image_m/fnbot-18-1431643-t001.jpg) ](https://www.frontiersin.org/files/Articles/1431643/fnbot-18-1431643-HTML-r1/image_m/fnbot-18-1431643-t001.jpg)

**Table 1**. Predicted results of low-frequency and high frequency components at different frequencies.

Convolutional networks can process images of different lengths and widths, and Recurrent Neural networks (RNN) have a recurrent function that can process data of different lengths and sequence types. However, due to the small range that RNN can utilize, it cannot handle the long sequence data well. The output that leads directly to a long sequence forgets the input that is farther away. LSTM is a special kind of RNN, a modified version of RNN, whose structure is plotted in Figure 6. The activation function is the sigmoid; tanh is the hyperbolic tangent function; ⊕ and ⊗ represent the addition and multiplication operations of vectors. The first layer of LSTM comprises a single-loop structure, which is determined by the dimensions and number of input data and loops, rather than the connection of multiple single-loop structures. LSTM cells contain input, forget, output and unit states (Akram et al., 2020; Zhang et al., 2020; Ahmad et al., 2022). The input gate determines how much network input data requires to be saved to the unit state at the current moment. The forget gate decides how many unit states need to be transferred from the last to the present moment. The output gate controls how much of the current unit state demands to be output to the present output value.

Figure 6

[ ![www.frontiersin.org](https://www.frontiersin.org/files/Articles/1431643/fnbot-18-1431643-HTML-r1/image_m/fnbot-18-1431643-g006.jpg) ](https://www.frontiersin.org/files/Articles/1431643/fnbot-18-1431643-HTML-r1/image_m/fnbot-18-1431643-g006.jpg)

**Figure 6**. Basic structural unit of LSTM.

In the discussed PV-PDN voltage prediction model based on “FDD+LSTM”, to better extract data characteristics, a fully connected layer, namely Dense layer, is placed behind the LSTM layer. The specific voltage prediction process is as follows. (1) The prediction methodology employs XGBoost for feature subset selection, focusing on crucial elements like voltage, power characteristics and temporal variables. This step is pivotal in distilling the most relevant features from a vast dataset, thereby improving model efficiency and focus. The resulting feature vector _x_ = [_V, P, T_] is a comprehensive aggregation of these elements, forming the LSTM input alongside the target training variable _y_ _i_. (2) The backpropagation algorithm is utilized for model training, optimizing the network to reduce prediction errors and heighten voltage forecasting accuracy. This phase ensures in-depth learning from historical data, a critical aspect of the model's predictive capability. (3) Finally, the trained LSTM model, equipped with learned patterns, processes the input dataset for voltage prediction. The inclusion of the dense layer at this point is significant. It acts as a refinement stage, aligning LSTM outputs with expected voltage levels and synthesizing complex relationships. This addition enhances the model's accuracy and robustness in diverse operational scenarios within PV-PDNs.

Detailed procedure for load forecasting of distributed PV system based on FDD+LSTM:

(1) Data selection and preprocessing: historical operation data is carefully selected and subjected to data mining and preprocessing techniques. This includes handling outliers, addressing missing values and normalizing the data to ensure its suitability for analysis.

(2) Feature selection: to identify the most influential variables contributing to the prediction task, we employ the XGBoost algorithm for feature selection. This approach enables us to pinpoint key predictive factors such as voltage, power characteristics and time variables that significantly impact the target variable.

(3) Model training: to achieve accurate and reliable voltage predictions, we employ the proposed “FDD+LSTM” neural network architecture and train it using the backpropagation algorithm.

(4) Load prediction: to harness the predictive ability of the trained proposed “FDD+LSTM” model, we utilize it to process the input dataset for accurate voltage forecasting. To further enhance the model's ability to extract meaningful features from the data and improve prediction accuracy, we incorporate a dense layer into the network architecture.

## 4 Results and discussion

### 4.1 Data selection and example analysis

This study selects the historical operation data of a low-voltage distributed PV-PDN in Guangdong Province as the research object. The time range is operation data from March 2020 to March 2022. The data sampling interval of the meter under test is 1 h, and rolling prediction is adopted. The constructed input feature vectors _x_ _i_ are: the vectors of voltage, power and time characteristics are 12, 12 and 35 dimensions, respectively, and a total of 16,275 data samples are constructed with _x_ _i_ and label _y_ _i_. An example analysis of the load forecasting model uses TensorFlow 14.0. The dropout layer is incorporated to prevent overfitting, followed by the connection to the output layer. The specific parameters of the “FDD+LSTM” prediction model are outlined in Table 2. The selected comparison algorithms are RNN, SVM, and LSTM to verify the validity of the prediction method proposed here.

Table 2

[ ![www.frontiersin.org](https://www.frontiersin.org/files/Articles/1431643/fnbot-18-1431643-HTML-r1/image_m/fnbot-18-1431643-t002.jpg) ](https://www.frontiersin.org/files/Articles/1431643/fnbot-18-1431643-HTML-r1/image_m/fnbot-18-1431643-t002.jpg)

**Table 2**. Specific parameters of LSTM prediction model.

### 4.2 Analysis of load forecasting results in distributed PV-PDN

To intuitively reflect the accuracy of voltage prediction results, this study draws corresponding voltage prediction curves with 1, 2 and 4-h as time scales, and compares them with other prediction models. The voltage data of 100-time points is selected as a display in the test set, and the voltage prediction results at different time scales are demonstrated in Figures 7–9. It can be found that the FDD-enhanced LSTM model consistently aligns more closely with actual voltage values than SVM (Kabilan et al., 2021), RNN (Shivam et al., 2021) and LSTM (Feng et al., 2020) models, especially as the prediction time scale increases. Quantitatively, the LSTM model's MAE is significantly lower, at 0.4554 for a 1-h scale, compared to 0.535 and 1.012 for RNN and SVM, respectively. Even at a 4-h scale, the LSTM's MAE remains the lowest at 1.085. The superior forecasting precision of the optimized LSTM model can be attributed to its ability to effectively capture and learn from the temporal dependencies inherent in voltage data over time. Unlike SVM and RNN models, LSTM's architecture allows it to remember information for longer periods, making it particularly adept at handling the sequence prediction problems characteristic of voltage forecasting in distributed PV-PDNs. This is crucial for accurately predicting voltage fluctuations over different time scales, as it can account for both short-term and long-term patterns in the data. Additionally, the integration of FDD likely enhances the model's capability to deal with the non-linear and complex nature of the voltage signals, further improving prediction accuracy.

Figure 7

[ ![www.frontiersin.org](https://www.frontiersin.org/files/Articles/1431643/fnbot-18-1431643-HTML-r1/image_m/fnbot-18-1431643-g007.jpg) ](https://www.frontiersin.org/files/Articles/1431643/fnbot-18-1431643-HTML-r1/image_m/fnbot-18-1431643-g007.jpg)

**Figure 7**. Voltage prediction results with a time scale of 1 h.

Figure 8

[ ![www.frontiersin.org](https://www.frontiersin.org/files/Articles/1431643/fnbot-18-1431643-HTML-r1/image_m/fnbot-18-1431643-g008.jpg) ](https://www.frontiersin.org/files/Articles/1431643/fnbot-18-1431643-HTML-r1/image_m/fnbot-18-1431643-g008.jpg)

**Figure 8**. Voltage prediction outcomes with a time scale of 2 h.

Figure 9

[ ![www.frontiersin.org](https://www.frontiersin.org/files/Articles/1431643/fnbot-18-1431643-HTML-r1/image_m/fnbot-18-1431643-g009.jpg) ](https://www.frontiersin.org/files/Articles/1431643/fnbot-18-1431643-HTML-r1/image_m/fnbot-18-1431643-g009.jpg)

**Figure 9**. Voltage prediction results with a time scale of 4 h.

### 4.3 Performance evaluation of load forecasting model under different seasons

Taking the time scale of 1-h and 4-h as the basis, this study further verifies the voltage prediction of different PDN's load forecasting models in the four seasons, and the comparison results are portrayed in Figures 10, 11. It can be concluded that the prediction results of the improved LSTM model based on FDD are optimal in all seasons, especially as the prediction time scale increases. Taking summer with a time scale of 1 h as an example, the prediction MAE of the improved LSTM model is only 0.24, which reduces the prediction error of this model by about 35%. Even at a 4-h scale, the LSTM's MAE remains the lowest at 1.064 in summer. Therefore, the capability of the model in load forecasting of PV-PDN is further verified.

Figure 10

[ ![www.frontiersin.org](https://www.frontiersin.org/files/Articles/1431643/fnbot-18-1431643-HTML-r1/image_m/fnbot-18-1431643-g010.jpg) ](https://www.frontiersin.org/files/Articles/1431643/fnbot-18-1431643-HTML-r1/image_m/fnbot-18-1431643-g010.jpg)

**Figure 10**. Comparison of load forecasting models for PDN in different seasons with a time scale of 1 h.

Figure 11

[ ![www.frontiersin.org](https://www.frontiersin.org/files/Articles/1431643/fnbot-18-1431643-HTML-r1/image_m/fnbot-18-1431643-g011.jpg) ](https://www.frontiersin.org/files/Articles/1431643/fnbot-18-1431643-HTML-r1/image_m/fnbot-18-1431643-g011.jpg)

**Figure 11**. Comparison of load forecasting models for PDN in different seasons with a time scale of 4 h.

The proposed algorithm demonstrates significant practical value and effectiveness in the PV-PDN scenario. It can accurately predict voltage variations under different environmental conditions, and its prediction accuracy surpasses that of other models, especially as the prediction time scale increases. This capability provides strong support for the safe, reliable and efficient operation of PV power stations, helping maintenance personnel to promptly identify and resolve potential issues, thereby improving the operational efficiency and long-term stability of the PV power stations.

## 5 Conclusion

Driven by the rapid development of new power systems, the proportion of new energy is continuously increasing, and the scale of application and access rate of distributed PV in the low-voltage PDN are also steadily rising. The integration of distributed PV power generation, nonetheless, often exerts a substantial impact on the voltage distribution within PDN, giving rise to issues such as low voltage and voltage fluctuations. These issues severely impact the quality of daily life and production for users, further augmenting the uncertainty in grid operation and hindering the development of the social economy. Consequently, enhancing the state awareness capability of PDN is of paramount importance. Effective voltage prediction can provide data support for the safe and stable operation of PDN, thereby facilitating the resolution of voltage issues arising from the integration of distributed PV systems. In recent years, LSTM networks have demonstrated remarkable application potential in the realm of power load forecasting, and it offer a novel solution for PDN voltage prediction. Thus, a LSTM is extensively used in power load forecasting model of actual PDN based on DL and FDD is proposed in this study. By fast Fourier decomposition of the original quantity, the phase and amplitude of each frequency sine wave are acquired. Then LSTM is used to predict the low-frequency and high frequency components, respectively. The effectiveness of the proposed FDD-based LSTM model is verified by testing the historical operating data of PV-PDN. With the increase of the prediction time scale of the improved model, the error of the predicted results does not increase significantly. At a 1-h time scale, the MAE of the improved LSTM model is only 0.4554, much lower than that of other models. However, the proposed model requires a large amount of data for training and cannot be directly deployed on edge clients with limited computational resources for prediction. In the future, with the continuous development of edge computing and deep learning technologies, optimizing model computation efficiency to accommodate hardware constraints of edge devices and developing lightweight deep learning algorithms to reduce resource consumption, deploying prediction models at the edge side will become more feasible.

## Data availability statement

The dataset for this research was provided by a collaborating institution and contains sensitive information and usage restrictions. If other researchers need to obtain this dataset for further research or other reasonable requests, please contact the corresponding author.

## Author contributions

XZ: Conceptualization, Formal analysis, Investigation, Methodology, Validation, Writing – original draft, Writing – review & editing. JunlW: Conceptualization, Methodology, Validation, Writing – original draft, Writing – review & editing. JunW: Methodology, Resources, Supervision, Writing – original draft, Writing – review & editing. HW: Data curation, Validation, Writing – original draft, Writing – review & editing. LL: Conceptualization, Validation, Writing – original draft, Writing – review & editing.

## Funding

The author(s) declare that no financial support was received for the research, authorship, and/or publication of this article.

## Conflict of interest

XZ and JunlW were employed by State Grid Hebei Electric Power Company.

JunW, HW, and LL were employed by Henan XJ Metering Co., Ltd.

## Publisher's note

All claims expressed in this article are solely those of the authors and do not necessarily represent those of their affiliated organizations, or those of the publisher, the editors and the reviewers. Any product that may be evaluated in this article, or claim that may be made by its manufacturer, is not guaranteed or endorsed by the publisher.

## References

Ahmad, T., Madonski, R., Zhang, D., Huang, C., and Mujeeb, A. (2022). Data-driven probabilistic machine learning in sustainable smart energy/smart energy systems: Key developments, challenges, and future research opportunities in the context of smart grid paradigm. _Renew. Sustain. Energy Rev._ 160, 112128. doi: 10.1016/j.rser.2022.112128

[Crossref Full Text](https://doi.org/10.1016/j.rser.2022.112128) | [Google Scholar](http://scholar.google.com/scholar_lookup?author=T.+Ahmad&author=R.+Madonski&author=D.+Zhang&author=C.+Huang&author=A.+Mujeeb+&publication_year=2022&title=Data-driven+probabilistic+machine+learning+in+sustainable+smart+energy%2Fsmart+energy+systems%3A+Key+developments,+challenges,+and+future+research+opportunities+in+the+context+of+smart+grid+paradigm&journal=Renew.+Sustain.+Energy+Rev.&volume=160&pages=112128)

Akram, M. W., Li, G., Jin, Y., Chen, X., Zhu, C., and Ahmad, A. (2020). Automatic detection of photovoltaic module defects in infrared images with isolated and develop-model transfer deep learning. _Solar Energy_ 198, 175–186. doi: 10.1016/j.solener.2020.01.055

[Crossref Full Text](https://doi.org/10.1016/j.solener.2020.01.055) | [Google Scholar](http://scholar.google.com/scholar_lookup?author=M.+W.+Akram&author=G.+Li&author=Y.+Jin&author=X.+Chen&author=C.+Zhu&author=A.+Ahmad+&publication_year=2020&title=Automatic+detection+of+photovoltaic+module+defects+in+infrared+images+with+isolated+and+develop-model+transfer+deep+learning&journal=Solar+Energy&volume=198&pages=175-186)

Alipour, M., Aghaei, J., Norouzi, M., Niknam, T., Hashemi, S., and Lehtonen, M. (2020). A novel electrical net-load forecasting model based on deep neural networks and wavelet transform integration. _Energy_ 205:118106. doi: 10.1016/j.energy.2020.118106

[Crossref Full Text](https://doi.org/10.1016/j.energy.2020.118106) | [Google Scholar](http://scholar.google.com/scholar_lookup?author=M.+Alipour&author=J.+Aghaei&author=M.+Norouzi&author=T.+Niknam&author=S.+Hashemi&author=M.+Lehtonen+&publication_year=2020&title=A+novel+electrical+net-load+forecasting+model+based+on+deep+neural+networks+and+wavelet+transform+integration&journal=Energy&volume=205&pages=118106)

Bae, D. J., Kwon, B. S., and Song, K. B. (2021). XGBoost-based day-ahead load forecasting algorithm considering behind-the-meter solar PV generation. _Energies_ 15:128. doi: 10.3390/en15010128

[Crossref Full Text](https://doi.org/10.3390/en15010128) | [Google Scholar](http://scholar.google.com/scholar_lookup?author=D.+J.+Bae&author=B.+S.+Kwon&author=K.+B.+Song+&publication_year=2021&title=XGBoost-based+day-ahead+load+forecasting+algorithm+considering+behind-the-meter+solar+PV+generation&journal=Energies&volume=15&pages=128)

Borges, C. E., Kamara-Esteban, O., Castillo-Calzadilla, T., Andonegui, C. M., and Alonso-Vicaria, A. (2020). Enhancing the missing data imputation of primary substation load demand records. _Sustain. Energy, Grids Netw._ 23:100369. doi: 10.1016/j.segan.2020.100369

[Crossref Full Text](https://doi.org/10.1016/j.segan.2020.100369) | [Google Scholar](http://scholar.google.com/scholar_lookup?author=C.+E.+Borges&author=O.+Kamara-Esteban&author=T.+Castillo-Calzadilla&author=C.+M.+Andonegui&author=A.+Alonso-Vicaria+&publication_year=2020&title=Enhancing+the+missing+data+imputation+of+primary+substation+load+demand+records&journal=Sustain.+Energy,+Grids+Netw.&volume=23&pages=100369)

Dairi, A., Harrou, F., Sun, Y., and Khadraoui, S. (2020). Short-term forecasting of photovoltaic solar power production using variational auto-encoder driven deep learning approach. _Appl. Sci._ 10:8400. doi: 10.3390/app10238400

[Crossref Full Text](https://doi.org/10.3390/app10238400) | [Google Scholar](http://scholar.google.com/scholar_lookup?author=A.+Dairi&author=F.+Harrou&author=Y.+Sun&author=S.+Khadraoui+&publication_year=2020&title=Short-term+forecasting+of+photovoltaic+solar+power+production+using+variational+auto-encoder+driven+deep+learning+approach&journal=Appl.+Sci.&volume=10&pages=8400)

Ding, S., Li, R., and Tao, Z. A. (2021). novel adaptive discrete grey model with time-varying parameters for long-term photovoltaic power generation forecasting. _Energy Convers. Manage._ 227:113644. doi: 10.1016/j.enconman.2020.113644

[Crossref Full Text](https://doi.org/10.1016/j.enconman.2020.113644) | [Google Scholar](http://scholar.google.com/scholar_lookup?author=S.+Ding&author=R.+Li&author=Z.+A.+Tao+&publication_year=2021&title=novel+adaptive+discrete+grey+model+with+time-varying+parameters+for+long-term+photovoltaic+power+generation+forecasting&journal=Energy+Convers.+Manage.&volume=227&pages=113644)

Eom, H., Son, Y., and Choi, S. (2020). Feature-selective ensemble learning-based long-term regional PV generation forecasting. _IEEE Access_ 8, 54620–54630. doi: 10.1109/ACCESS.2020.2981819

[Crossref Full Text](https://doi.org/10.1109/ACCESS.2020.2981819) | [Google Scholar](http://scholar.google.com/scholar_lookup?author=H.+Eom&author=Y.+Son&author=S.+Choi+&publication_year=2020&title=Feature-selective+ensemble+learning-based+long-term+regional+PV+generation+forecasting&journal=IEEE+Access&volume=8&pages=54620-54630)

Espinoza, M., Joye, C., Belmans, R., and De Moor, B. (2005). Short-term load forecasting, profile identification, and customer segmentation: a methodology based on periodic time series. _IEEE Trans. Power Syst._ 20, 1622–1630. doi: 10.1109/TPWRS.2005.852123

[Crossref Full Text](https://doi.org/10.1109/TPWRS.2005.852123) | [Google Scholar](http://scholar.google.com/scholar_lookup?author=M.+Espinoza&author=C.+Joye&author=R.+Belmans&author=B.+De+Moor+&publication_year=2005&title=Short-term+load+forecasting,+profile+identification,+and+customer+segmentation%3A+a+methodology+based+on+periodic+time+series&journal=IEEE+Trans.+Power+Syst.&volume=20&pages=1622-1630)

Feng, Y., Hao, W., Li, H., Cui, N., Gong, D., and Gao, L. (2020). Machine learning models to quantify and map daily global solar radiation and photovoltaic power. _Renew. Sustain. Energy Rev._ 118:109393. doi: 10.1016/j.rser.2019.109393

[Crossref Full Text](https://doi.org/10.1016/j.rser.2019.109393) | [Google Scholar](http://scholar.google.com/scholar_lookup?author=Y.+Feng&author=W.+Hao&author=H.+Li&author=N.+Cui&author=D.+Gong&author=L.+Gao+&publication_year=2020&title=Machine+learning+models+to+quantify+and+map+daily+global+solar+radiation+and+photovoltaic+power&journal=Renew.+Sustain.+Energy+Rev.&volume=118&pages=109393)

Hafiz, F., Awal, M. A., de Queiroz, A. R., and Hussain, I. (2020). Real-time stochastic optimization of energy storage management using deep learning-based forecasts for residential PV applications. _IEEE Trans. Ind. Appl._ 56, 2216–2226. doi: 10.1109/TIA.2020.2968534

[Crossref Full Text](https://doi.org/10.1109/TIA.2020.2968534) | [Google Scholar](http://scholar.google.com/scholar_lookup?author=F.+Hafiz&author=M.+A.+Awal&author=A.+R.+de+Queiroz&author=I.+Hussain+&publication_year=2020&title=Real-time+stochastic+optimization+of+energy+storage+management+using+deep+learning-based+forecasts+for+residential+PV+applications&journal=IEEE+Trans.+Ind.+Appl.&volume=56&pages=2216-2226)

Hayes, B. P., Gruber, J. K., and Prodanovic, M. A. (2014). closed-loop state estimation tool for MV network monitoring and operation. _IEEE Trans. Smart Grid_ 6, 2116–2125. doi: 10.1109/TSG.2014.2378035

[Crossref Full Text](https://doi.org/10.1109/TSG.2014.2378035) | [Google Scholar](http://scholar.google.com/scholar_lookup?author=B.+P.+Hayes&author=J.+K.+Gruber&author=M.+A.+Prodanovic+&publication_year=2014&title=closed-loop+state+estimation+tool+for+MV+network+monitoring+and+operation&journal=IEEE+Trans.+Smart+Grid&volume=6&pages=2116-2125)

Jin, L., Liu, L., Wang, X., Shang, M., Wang, F. -Y., et al. (2024). “Physical-informed neural network for mpc-based trajectory tracking of vehicles with noise considered,” in _IEEE Transactions on Intelligent Vehicles_.

[Google Scholar](http://scholar.google.com/scholar_lookup?author=L.+Jin&author=L.+Liu&author=X.+Wang&author=M.+Shang&author=F.++-Y.+Wang+&publication_year=2024&title=“Physical-informed+neural+network+for+mpc-based+trajectory+tracking+of+vehicles+with+noise+considered,”&journal=IEEE+Transactions+on+Intelligent+Vehicles)

Kabilan, R., Chandran, V., Yogapriya, J., Karthick, A., Gandhi, P., Mohanavei, V., et al. (2021). Short-term power prediction of building integrated photovoltaic (BIPV) system based on machine learning algorithms. _Int. J. Photoen._ 2021, 1–11. doi: 10.1155/2021/5582418

[Crossref Full Text](https://doi.org/10.1155/2021/5582418) | [Google Scholar](http://scholar.google.com/scholar_lookup?author=R.+Kabilan&author=V.+Chandran&author=J.+Yogapriya&author=A.+Karthick&author=P.+Gandhi&author=V.+Mohanavei+&publication_year=2021&title=Short-term+power+prediction+of+building+integrated+photovoltaic+\(BIPV\)+system+based+on+machine+learning+algorithms&journal=Int.+J.+Photoen.&volume=2021&pages=1-11)

Karimi, A. M., Fada, J. S., Parrilla, N. A., Pierce, B., Koyutürk, M., French, R. H., et al. (2020). Generalized and mechanistic PV module performance prediction from computer vision and machine learning on electroluminescence images. _IEEE J. Photovolt._ 10, 878–887. doi: 10.1109/JPHOTOV.2020.2973448

[Crossref Full Text](https://doi.org/10.1109/JPHOTOV.2020.2973448) | [Google Scholar](http://scholar.google.com/scholar_lookup?author=A.+M.+Karimi&author=J.+S.+Fada&author=N.+A.+Parrilla&author=B.+Pierce&author=M.+Koyutürk&author=R.+H.+French+&publication_year=2020&title=Generalized+and+mechanistic+PV+module+performance+prediction+from+computer+vision+and+machine+learning+on+electroluminescence+images&journal=IEEE+J.+Photovolt.&volume=10&pages=878-887)

Khan, W., Walker, S., and Zeiler, W. (2022). Improved solar photovoltaic energy generation forecast using deep learning-based ensemble stacking approach. _Energy_ 240:122812. doi: 10.1016/j.energy.2021.122812

[Crossref Full Text](https://doi.org/10.1016/j.energy.2021.122812) | [Google Scholar](http://scholar.google.com/scholar_lookup?author=W.+Khan&author=S.+Walker&author=W.+Zeiler+&publication_year=2022&title=Improved+solar+photovoltaic+energy+generation+forecast+using+deep+learning-based+ensemble+stacking+approach&journal=Energy&volume=240&pages=122812)

Kim, D., Kwon, D., Park, L., Kim, J., and Cho, S. (2020). Multiscale LSTM-based deep learning for very-short-term photovoltaic power generation forecasting in smart city energy management. _IEEE Syst. J._ 15, 346–354. doi: 10.1109/JSYST.2020.3007184

[Crossref Full Text](https://doi.org/10.1109/JSYST.2020.3007184) | [Google Scholar](http://scholar.google.com/scholar_lookup?author=D.+Kim&author=D.+Kwon&author=L.+Park&author=J.+Kim&author=S.+Cho+&publication_year=2020&title=Multiscale+LSTM-based+deep+learning+for+very-short-term+photovoltaic+power+generation+forecasting+in+smart+city+energy+management&journal=IEEE+Syst.+J.&volume=15&pages=346-354)

Korkmaz, D. (2021). SolarNet: a hybrid reliable model based on convolutional neural network and variational mode decomposition for hourly photovoltaic power forecasting. _Appl. Energy_ 300:117410. doi: 10.1016/j.apenergy.2021.117410

[Crossref Full Text](https://doi.org/10.1016/j.apenergy.2021.117410) | [Google Scholar](http://scholar.google.com/scholar_lookup?author=D.+Korkmaz+&publication_year=2021&title=SolarNet%3A+a+hybrid+reliable+model+based+on+convolutional+neural+network+and+variational+mode+decomposition+for+hourly+photovoltaic+power+forecasting&journal=Appl.+Energy&volume=300&pages=117410)

Lee, Y. M., An, L., Liu, F., Horesh, R., Chae, Y. T., and Zhang, R. (2013). Applying science and mathematics to big data for smarter buildings. _Ann. N. Y. Acad. Sci._ 1295, 18–25. doi: 10.1111/nyas.12193

[PubMed Abstract](https://pubmed.ncbi.nlm.nih.gov/23819911) | [Crossref Full Text](https://doi.org/10.1111/nyas.12193) | [Google Scholar](http://scholar.google.com/scholar_lookup?author=Y.+M.+Lee&author=L.+An&author=F.+Liu&author=R.+Horesh&author=Y.+T.+Chae&author=R.+Zhang+&publication_year=2013&title=Applying+science+and+mathematics+to+big+data+for+smarter+buildings&journal=Ann.+N.+Y.+Acad.+Sci.&volume=1295&pages=18-25)

Li, N., Ma, L., Yu, G., Xue, B., Zhang, M., and Jin, J. (2023). Survey on evolutionary deep learning: Principles, algorithms, applications, and open issues. _ACM Comp. Surv._ 56, 1–34. doi: 10.1145/3603704

[Crossref Full Text](https://doi.org/10.1145/3603704) | [Google Scholar](http://scholar.google.com/scholar_lookup?author=N.+Li&author=L.+Ma&author=G.+Yu&author=B.+Xue&author=M.+Zhang&author=J.+Jin+&publication_year=2023&title=Survey+on+evolutionary+deep+learning%3A+Principles,+algorithms,+applications,+and+open+issues&journal=ACM+Comp.+Surv.&volume=56&pages=1-34)

Litjens, G., Worrell, E., and Van Sark, W. (2018). Assessment of forecasting methods on performance of photovoltaic-battery systems. _Appl. Energy_ 221, 358–373. doi: 10.1016/j.apenergy.2018.03.154

[Crossref Full Text](https://doi.org/10.1016/j.apenergy.2018.03.154) | [Google Scholar](http://scholar.google.com/scholar_lookup?author=G.+Litjens&author=E.+Worrell&author=W.+Van+Sark+&publication_year=2018&title=Assessment+of+forecasting+methods+on+performance+of+photovoltaic-battery+systems&journal=Appl.+Energy&volume=221&pages=358-373)

Liu, W., Tang, P., Liu, H., and Zhao, P. (2022). Intelligent voltage prediction of active distribution network with high proportion of distributed photovoltaics. _Energy Rep._ 8, 894–903. doi: 10.1016/j.egyr.2022.08.142

[Crossref Full Text](https://doi.org/10.1016/j.egyr.2022.08.142) | [Google Scholar](http://scholar.google.com/scholar_lookup?author=W.+Liu&author=P.+Tang&author=H.+Liu&author=P.+Zhao+&publication_year=2022&title=Intelligent+voltage+prediction+of+active+distribution+network+with+high+proportion+of+distributed+photovoltaics&journal=Energy+Rep.&volume=8&pages=894-903)

Liu, Z. F., Li, L. L., Tseng, M. L., and Lim, M. (2020). Prediction short-term photovoltaic power using improved chicken swarm optimizer-extreme learning machine model. _J. Clean. Prod._ 248:119272. doi: 10.1016/j.jclepro.2019.119272

[Crossref Full Text](https://doi.org/10.1016/j.jclepro.2019.119272) | [Google Scholar](http://scholar.google.com/scholar_lookup?author=Z.+F.+Liu&author=L.+L.+Li&author=M.+L.+Tseng&author=M.+Lim+&publication_year=2020&title=Prediction+short-term+photovoltaic+power+using+improved+chicken+swarm+optimizer-extreme+learning+machine+model&journal=J.+Clean.+Prod.&volume=248&pages=119272)

Liufu, Y., Jin, L., Shang, M., Wang, X., and Wang, F.-Y. (2024). ACP-incorporated perturbation-resistant neural dynamics controller for autonomoius vehicles. _IEEE Trans. Intell. Vehicles_ 9, 4675–4686. doi: 10.1109/TIV.2023.3348632

[Crossref Full Text](https://doi.org/10.1109/TIV.2023.3348632) | [Google Scholar](http://scholar.google.com/scholar_lookup?author=Y.+Liufu&author=L.+Jin&author=M.+Shang&author=X.+Wang&author=F.+-Y.+Wang+&publication_year=2024&title=ACP-incorporated+perturbation-resistant+neural+dynamics+controller+for+autonomoius+vehicles&journal=IEEE+Trans.+Intell.+Vehicles&volume=9&pages=4675-4686)

Lopes, S. M. A., Cari, E. P. T., and Hajimirza, S. A. (2022). Comparative analysis of Artificial Neural Networks for Photovoltaic Power Forecast using remotes and local measurements. _J. Solar Energy Eng._ 144:021007. doi: 10.1115/1.4053031

[Crossref Full Text](https://doi.org/10.1115/1.4053031) | [Google Scholar](http://scholar.google.com/scholar_lookup?author=S.+M.+A.+Lopes&author=E.+P.+T.+Cari&author=S.+A.+Hajimirza+&publication_year=2022&title=Comparative+analysis+of+Artificial+Neural+Networks+for+Photovoltaic+Power+Forecast+using+remotes+and+local+measurements&journal=J.+Solar+Energy+Eng.&volume=144&pages=021007)

Luo, X., Zhang, D., and Zhu, X. (2021). Deep learning based forecasting of photovoltaic power generation by incorporating domain knowledge. _Energy_ 225:120240. doi: 10.1016/j.energy.2021.120240

[Crossref Full Text](https://doi.org/10.1016/j.energy.2021.120240) | [Google Scholar](http://scholar.google.com/scholar_lookup?author=X.+Luo&author=D.+Zhang&author=X.+Zhu+&publication_year=2021&title=Deep+learning+based+forecasting+of+photovoltaic+power+generation+by+incorporating+domain+knowledge&journal=Energy&volume=225&pages=120240)

Ma, L., Kang, H., Yu, G., Li, Q., and He, Q. (2024a). Single-domain generalized predictor for neural architecture search system. _IEEE Trans. Comp._ 73, 1400–1413. doi: 10.1109/TC.2024.3365949

[PubMed Abstract](https://pubmed.ncbi.nlm.nih.gov/35062661) | [Crossref Full Text](https://doi.org/10.1109/TC.2024.3365949) | [Google Scholar](http://scholar.google.com/scholar_lookup?author=L.+Ma&author=H.+Kang&author=G.+Yu&author=Q.+Li&author=Q.+He+&publication_year=2024a&title=Single-domain+generalized+predictor+for+neural+architecture+search+system&journal=IEEE+Trans.+Comp.&volume=73&pages=1400-1413)

Ma, L., Li, N., Guo, Y., Wang, X., Yang, S., Huang, M., et al. (2021). Learning to optimize: reference vector reinforcement learning adaption to constrained many-objective optimization of industrial copper burdening system. _IEEE Trans. Cybernet_. 52, 12698–12711. doi: 10.1109/TCYB.2021.3086501

[PubMed Abstract](https://pubmed.ncbi.nlm.nih.gov/34260364) | [Crossref Full Text](https://doi.org/10.1109/TCYB.2021.3086501) | [Google Scholar](http://scholar.google.com/scholar_lookup?author=L.+Ma&author=N.+Li&author=Y.+Guo&author=X.+Wang&author=S.+Yang&author=M.+Huang+&publication_year=2021&title=Learning+to+optimize%3A+reference+vector+reinforcement+learning+adaption+to+constrained+many-objective+optimization+of+industrial+copper+burdening+system&journal=IEEE+Trans.+Cybernet&volume=52&pages=12698-12711)

Ma, L., Li, N., Yu, G., Geng, X., Cheng, S., Wang, X., et al. (2023). Pareto-wise ranking classifier for multi-objective evolutionary neural architecture search. _IEEE Trans. Evol_. 28, 570–581. doi: 10.1109/TEVC.2023.3314766

[Crossref Full Text](https://doi.org/10.1109/TEVC.2023.3314766) | [Google Scholar](http://scholar.google.com/scholar_lookup?author=L.+Ma&author=N.+Li&author=G.+Yu&author=X.+Geng&author=S.+Cheng&author=X.+Wang+&publication_year=2023&title=Pareto-wise+ranking+classifier+for+multi-objective+evolutionary+neural+architecture+search&journal=IEEE+Trans.+Evol&volume=28&pages=570-581)

Ma, L., Li, N., Zhu, P., Tang, K., Khan, A., Wang, F., et al. (2024b). A novel fuzzy neural network architecture search framework for defect recognition with uncertainties. _IEEE Trans. Fuzzy Syst_. 32, 3274–3285. doi: 10.1109/TFUZZ.2024.3373792

[Crossref Full Text](https://doi.org/10.1109/TFUZZ.2024.3373792) | [Google Scholar](http://scholar.google.com/scholar_lookup?author=L.+Ma&author=N.+Li&author=P.+Zhu&author=K.+Tang&author=A.+Khan&author=F.+Wang+&publication_year=2024b&title=A+novel+fuzzy+neural+network+architecture+search+framework+for+defect+recognition+with+uncertainties&journal=IEEE+Trans.+Fuzzy+Syst&volume=32&pages=3274-3285)

Marinescu, A., Harris, C., Dusparic, I., Clarke, S., and Cahill, V. (2013). “Residential electrical demand forecasting in very small scale: an evaluation of forecasting methods,” in _2013 2nd International Workshop on Software Engineering Challenges for the Smart Grid (SE4SG)_ (San Francisco, CA: IEEE), 25–32. doi: 10.1109/SE4SG.2013.6596108

[Crossref Full Text](https://doi.org/10.1109/SE4SG.2013.6596108) | [Google Scholar](http://scholar.google.com/scholar_lookup?author=A.+Marinescu&author=C.+Harris&author=I.+Dusparic&author=S.+Clarke&author=V.+Cahill+&publication_year=2013&title=“Residential+electrical+demand+forecasting+in+very+small+scale%3A+an+evaluation+of+forecasting+methods,”&journal=2013+2nd+International+Workshop+on+Software+Engineering+Challenges+for+the+Smart+Grid+\(SE4SG\))

Markovics, D., and Mayer, M. J. (2022). Comparison of machine learning methods for photovoltaic power forecasting based on numerical weather prediction. _Renew. Sustain. Energy Rev._ 161:112364. doi: 10.1016/j.rser.2022.112364

[Crossref Full Text](https://doi.org/10.1016/j.rser.2022.112364) | [Google Scholar](http://scholar.google.com/scholar_lookup?author=D.+Markovics&author=M.+J.+Mayer+&publication_year=2022&title=Comparison+of+machine+learning+methods+for+photovoltaic+power+forecasting+based+on+numerical+weather+prediction&journal=Renew.+Sustain.+Energy+Rev.&volume=161&pages=112364)

Mellit, A., Pavan, A. M., and Lughi, V. (2021). Deep learning neural networks for short-term photovoltaic power forecasting. _Renew. Energy_ 172, 276–288. doi: 10.1016/j.renene.2021.02.166

[Crossref Full Text](https://doi.org/10.1016/j.renene.2021.02.166) | [Google Scholar](http://scholar.google.com/scholar_lookup?author=A.+Mellit&author=A.+M.+Pavan&author=V.+Lughi+&publication_year=2021&title=Deep+learning+neural+networks+for+short-term+photovoltaic+power+forecasting&journal=Renew.+Energy&volume=172&pages=276-288)

Polo, J., Martín-Chivelet, N., Alonso-Abella, M., Sanz-Salz, C., and de la Cuz, M. (2023). Exploring the PV power forecasting at building façades using gradient boosting methods. _Energies_ 16:1495. doi: 10.3390/en16031495

[Crossref Full Text](https://doi.org/10.3390/en16031495) | [Google Scholar](http://scholar.google.com/scholar_lookup?author=J.+Polo&author=N.+Martín-Chivelet&author=M.+Alonso-Abella&author=C.+Sanz-Salz&author=M.+de+la+Cuz+&publication_year=2023&title=Exploring+the+PV+power+forecasting+at+building+façades+using+gradient+boosting+methods&journal=Energies&volume=16&pages=1495)

Qadir, Z., Khan, S. I., Khalaji, E., Munawar, H. S., Al-Turjman, F., Mohmud, P., et al. (2021). Predicting the energy output of hybrid PV–wind renewable energy system using feature selection technique for smart grids. _Energy Rep._ 7, 8465–8475. doi: 10.1016/j.egyr.2021.01.018

[Crossref Full Text](https://doi.org/10.1016/j.egyr.2021.01.018) | [Google Scholar](http://scholar.google.com/scholar_lookup?author=Z.+Qadir&author=S.+I.+Khan&author=E.+Khalaji&author=H.+S.+Munawar&author=F.+Al-Turjman&author=P.+Mohmud+&publication_year=2021&title=Predicting+the+energy+output+of+hybrid+PV–wind+renewable+energy+system+using+feature+selection+technique+for+smart+grids&journal=Energy+Rep.&volume=7&pages=8465-8475)

Rai, P., Londhe, N. D., and Raj, R. (2021). Fault classification in power system distribution network integrated with distributed generators using CNN. _Electric Power Syst. Res._ 192:106914. doi: 10.1016/j.epsr.2020.106914

[Crossref Full Text](https://doi.org/10.1016/j.epsr.2020.106914) | [Google Scholar](http://scholar.google.com/scholar_lookup?author=P.+Rai&author=N.+D.+Londhe&author=R.+Raj+&publication_year=2021&title=Fault+classification+in+power+system+distribution+network+integrated+with+distributed+generators+using+CNN&journal=Electric+Power+Syst.+Res.&volume=192&pages=106914)

Rana, M., and Rahman, A. (2020). Multiple steps ahead solar photovoltaic power forecasting based on univariate machine learning models and data re-sampling. _Sustain. Energy, Grids Netw._ 21:100286. doi: 10.1016/j.segan.2019.100286

[Crossref Full Text](https://doi.org/10.1016/j.segan.2019.100286) | [Google Scholar](http://scholar.google.com/scholar_lookup?author=M.+Rana&author=A.+Rahman+&publication_year=2020&title=Multiple+steps+ahead+solar+photovoltaic+power+forecasting+based+on+univariate+machine+learning+models+and+data+re-sampling&journal=Sustain.+Energy,+Grids+Netw.&volume=21&pages=100286)

Razavi, S. E., Arefi, A., Ledwich, G., Nourbakhsh, G., Smith, D. B., Minakshi, M., et al. (2020). From load to net energy forecasting: short-term residential forecasting for the blend of load and PV behind the meter. _IEEE Access_ 8, 224343–224353. doi: 10.1109/ACCESS.2020.3044307

[Crossref Full Text](https://doi.org/10.1109/ACCESS.2020.3044307) | [Google Scholar](http://scholar.google.com/scholar_lookup?author=S.+E.+Razavi&author=A.+Arefi&author=G.+Ledwich&author=G.+Nourbakhsh&author=D.+B.+Smith&author=M.+Minakshi+&publication_year=2020&title=From+load+to+net+energy+forecasting%3A+short-term+residential+forecasting+for+the+blend+of+load+and+PV+behind+the+meter&journal=IEEE+Access&volume=8&pages=224343-224353)

Shivam, K., Tzou, J. C., and Wu, S. C. A. (2021). multi-objective predictive energy management strategy for residential grid-connected PV-battery hybrid systems based on machine learning technique. _Energy Conv. Manage._ 237:114103. doi: 10.1016/j.enconman.2021.114103

[Crossref Full Text](https://doi.org/10.1016/j.enconman.2021.114103) | [Google Scholar](http://scholar.google.com/scholar_lookup?author=K.+Shivam&author=J.+C.+Tzou&author=S.+C.+A.+Wu+&publication_year=2021&title=multi-objective+predictive+energy+management+strategy+for+residential+grid-connected+PV-battery+hybrid+systems+based+on+machine+learning+technique&journal=Energy+Conv.+Manage.&volume=237&pages=114103)

Tsekouras, G. J., Dialynas, E. N., Hatziargyriou, N. D., and Kavatza, S. (2007). A non-linear multivariable regression model for midterm energy forecasting of power systems. _Electric Power Syst. Res._ 77, 1560–1568. doi: 10.1016/j.epsr.2006.11.003

[Crossref Full Text](https://doi.org/10.1016/j.epsr.2006.11.003) | [Google Scholar](http://scholar.google.com/scholar_lookup?author=G.+J.+Tsekouras&author=E.+N.+Dialynas&author=N.+D.+Hatziargyriou&author=S.+Kavatza+&publication_year=2007&title=A+non-linear+multivariable+regression+model+for+midterm+energy+forecasting+of+power+systems&journal=Electric+Power+Syst.+Res.&volume=77&pages=1560-1568)

Wang, S., Dong, P., and Tian, Y. A. (2017). novel method of statistical line loss estimation for distribution feeders based on feeder cluster and modified XGBoost. _Energies_ 10:2067. doi: 10.3390/en10122067

[Crossref Full Text](https://doi.org/10.3390/en10122067) | [Google Scholar](http://scholar.google.com/scholar_lookup?author=S.+Wang&author=P.+Dong&author=Y.+A.+Tian+&publication_year=2017&title=novel+method+of+statistical+line+loss+estimation+for+distribution+feeders+based+on+feeder+cluster+and+modified+XGBoost&journal=Energies&volume=10&pages=2067)

Zang, H., Cheng, L., Ding, T., Cheung, K., Wei, Z., and Sun, G. (2020). Day-ahead photovoltaic power forecasting approach based on deep convolutional neural networks and meta learning. _Int. J. Elect. Power Energy Syst._ 118:105790. doi: 10.1016/j.ijepes.2019.105790

[Crossref Full Text](https://doi.org/10.1016/j.ijepes.2019.105790) | [Google Scholar](http://scholar.google.com/scholar_lookup?author=H.+Zang&author=L.+Cheng&author=T.+Ding&author=K.+Cheung&author=Z.+Wei&author=G.+Sun+&publication_year=2020&title=Day-ahead+photovoltaic+power+forecasting+approach+based+on+deep+convolutional+neural+networks+and+meta+learning&journal=Int.+J.+Elect.+Power+Energy+Syst.&volume=118&pages=105790)

Zhang, Y., Qin, C., Srivastava, A. K., Jin, C., and Sharma, R. (2020). Data-driven day-ahead PV estimation using autoencoder-LSTM and persistence model. _IEEE Trans. Ind. Appl._ 56, 7185–7192. doi: 10.1109/TIA.2020.3025742

[Crossref Full Text](https://doi.org/10.1109/TIA.2020.3025742) | [Google Scholar](http://scholar.google.com/scholar_lookup?author=Y.+Zhang&author=C.+Qin&author=A.+K.+Srivastava&author=C.+Jin&author=R.+Sharma+&publication_year=2020&title=Data-driven+day-ahead+PV+estimation+using+autoencoder-LSTM+and+persistence+model&journal=IEEE+Trans.+Ind.+Appl.&volume=56&pages=7185-7192)

Keywords: distributed photovoltaic, power distribution network, load forecasting, deep learning, long short-term memory

Citation: Zhang X, Wang J, Wang J, Wang H and Lu L (2024) Enhanced LSTM-based robotic agent for load forecasting in low-voltage distributed photovoltaic power distribution network. _Front. Neurorobot._ 18:1431643. doi: 10.3389/fnbot.2024.1431643

Received: 12 May 2024; Accepted: 31 May 2024;  
Published: 11 July 2024.

Edited by:

[Long Jin](http://loop.frontiersin.org/people/449093/overview), Lanzhou University, China

Reviewed by:

[Yichuan Zhang](http://loop.frontiersin.org/people/2739519/overview), Northeastern University, China  
[Ming Wan](http://loop.frontiersin.org/people/1782537/overview), Liaoning University, China

Copyright © 2024 Zhang, Wang, Wang, Wang and Lu. This is an open-access article distributed under the terms of the [Creative Commons Attribution License (CC BY)](http://creativecommons.org/licenses/by/4.0/). The use, distribution or reproduction in other forums is permitted, provided the original author(s) and the copyright owner(s) are credited and that the original publication in this journal is cited, in accordance with accepted academic practice. No use, distribution or reproduction is permitted which does not comply with these terms.

*Correspondence: Jun Wang, d2FuZ2p1bkB4ai5jZWUtZ3JvdXAuY24=

Disclaimer:  All claims expressed in this article are solely those of the authors and do not necessarily represent those of their affiliated organizations, or those of the publisher, the editors and the reviewers. Any product that may be evaluated in this article or claim that may be made by its manufacturer is not guaranteed or endorsed by the publisher. 

Download article

  * [Download PDF](https://public-pages-files-2025.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1431643/pdf)
  * [ReadCube](http://www.readcube.com/articles/10.3389/fnbot.2024.1431643)
  * [EPUB](https://public-pages-files-2025.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1431643/epub)
  * [XML](https://public-pages-files-2025.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1431643/xml)



Share on

  * [](https://www.twitter.com/share?url=https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1431643/full "Share on X")
  * [](https://www.linkedin.com/share?url=https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1431643/full "Share on Linkedin")
  * [](https://www.facebook.com/sharer/sharer.php?u=https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1431643/full "Share on Facebook")



Export citation

  * [EndNote](https://public-pages-files-2025.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1431643/endNote)
  * [Reference Manager](https://public-pages-files-2025.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1431643/reference)
  * [Simple Text file](https://public-pages-files-2025.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1431643/text)
  * [BibTex](https://public-pages-files-2025.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1431643/bibTex)



2,1K

Total views

637

Downloads

5

Citations

Citation numbers are available from Dimensions 

[View article impact](http://loop-impact.frontiersin.org/impact/article/1431643#totalviews/views)

[View altmetric score](https://www.altmetric.com/details/doi/10.3389/fnbot.2024.1431643)

Share on

  * [](https://www.twitter.com/share?url=https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1431643/full "Share on X")
  * [](https://www.linkedin.com/share?url=https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1431643/full "Share on Linkedin")
  * [](https://www.facebook.com/sharer/sharer.php?u=https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1431643/full "Share on Facebook")



Edited by

[LJLong JinLanzhou University, China](https://loop.frontiersin.org/people/449093/overview)

Reviewed by

[MWMing WanLiaoning University, China](https://loop.frontiersin.org/people/1782537/overview)[YZYichuan ZhangNortheastern University, China](https://loop.frontiersin.org/people/2739519/overview)

Table of contents

  * Abstract
  * 1 Introduction
  * 2 Related work
  * 3 Methods
  * 4 Results and discussion
  * 5 Conclusion
  * Data availability statement
  * Author contributions
  * Funding
  * Conflict of interest
  * Publisher's note
  * References



Export citation

  * [EndNote](https://public-pages-files-2025.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1431643/endNote)
  * [Reference Manager](https://public-pages-files-2025.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1431643/reference)
  * [Simple Text file](https://public-pages-files-2025.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1431643/text)
  * [BibTex](https://public-pages-files-2025.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1431643/bibTex)



![Crossmark icon](/ap-2024/images/crossmark.svg)

Check for updates

Frontiers' quality

![](https://images-provider.frontiersin.org/api/ipx/s=366x408&fit=outside&f=jpg/https://brand.frontiersin.org/m/208b12fe47a59b7a/webimage-RI_AuditingTeam_Workshop-2.png)

## Guardians of scientific integrity

Our research integrity and auditing teams lead the rigorous process that protects the quality of the scientific record

  
[More about research integrity](https://www.frontiersin.org/about/research-integrity)

Supplementary Material




Download article

Download

  * [Download PDF](https://public-pages-files-2025.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1431643/pdf)
  * [ReadCube](http://www.readcube.com/articles/10.3389/fnbot.2024.1431643)
  * [EPUB](https://public-pages-files-2025.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1431643/epub)
  * [XML](https://public-pages-files-2025.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1431643/xml)


