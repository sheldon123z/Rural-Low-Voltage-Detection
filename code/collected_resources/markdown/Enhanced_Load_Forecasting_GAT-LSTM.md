---
title: Enhanced Load Forecasting with GAT-LSTM: Leveraging Grid and Temporal Features
source: https://arxiv.org/html/2502.08376v1
scraped_date: 2026-01-20 10:28:21
---

# Enhanced Load Forecasting with GAT-LSTM: Leveraging Grid and Temporal Features

Ugochukwu Orji  [0009-0003-9973-8231](https://orcid.org/0009-0003-9973-8231 "ORCID identifier") Jheronimus Academy of Data Science (JADS), Tilburg UniversityTilburgThe Netherlands [u.e.orji@tilburguniversity.edu](mailto:u.e.orji@tilburguniversity.edu) ,Â  Ã‡iÃ§ek GÃ¼ven  [0000-0002-1939-8325](https://orcid.org/0000-0002-1939-8325 "ORCID identifier") Department of Cognitive Science and Artificial Intelligence, Tilburg UniversityTilburgThe Netherlands [c.guven@tilburguniversity.edu](mailto:c.guven@tilburguniversity.edu) Â andÂ  Dan Stowell  [0000-0001-8068-3769](https://orcid.org/0000-0001-8068-3769 "ORCID identifier") Department of Cognitive Science and Artificial Intelligence, Tilburg UniversityTilburgThe Netherlands Naturalis Biodiversity CenterLeidenThe Netherlands [d.stowell@tilburguniversity.edu](mailto:d.stowell@tilburguniversity.edu)

###### Abstract.

Accurate power load forecasting is essential for the efficient operation and planning of electrical grids, particularly given the increased variability and complexity introduced by renewable energy sources. This paper introduces GAT-LSTM, a hybrid model that combines Graph Attention Networks (GAT) and Long Short-Term Memory (LSTM) networks. A key innovation of the model is the incorporation of edge attributes, such as line capacities and efficiencies, into the attention mechanism, enabling it to dynamically capture spatial relationships grounded in grid-specific physical and operational constraints. Additionally, by employing an early fusion of spatial graph embeddings and temporal sequence features, the model effectively learns and predicts complex interactions between spatial dependencies and temporal patterns, providing a realistic representation of the dynamics of power grids. Experimental evaluations on the Brazilian Electricity System dataset demonstrate that the GAT-LSTM model significantly outperforms state-of-the-art models, achieving reductions of 21. 8% in MAE, 15. 9% in RMSE and 20. 2% in MAPE. These results underscore the robustness and adaptability of the GAT-LSTM model, establishing it as a powerful tool for applications in grid management and energy planning.

Short-Term Load Forecasting, Spatial-Temporal Analysis, Graph Attention Network, Long Short-Term Memory, Hybrid models 

â€ â€ copyright: acmlicensedâ€ â€ conference: Proceedings of the 2025 ACM International Conference on Future Energy Systems; June 17â€“20, 2025; Rotterdam, Netherlands

##  1\. Introduction

Modern power systems are intricate networks of interconnected components responsible for generating, transmitting, distributing, and utilizing electricity. A key challenge in these systems is maintaining the balance between supply and demand, especially with the rapid integration of renewable energy sources (RES) such as solar and wind into grid operations. The intermittent nature of RES introduces variability, which requires the deployment of new technologies that enhance grid flexibility and enable real-time responses.

Consequently, accurate demand estimation is fundamental for effective power planning as matching supply with demand is essential to maintain grid stability. Note that errors in demand forecasting can result in significant financial costs, with even a 1% forecasting error potentially leading to hundreds of thousands of dollars in losses per GWh (Mansoor etÂ al., [2024](https://arxiv.org/html/2502.08376v1#bib.bib40)). Since forecasting plays a critical role in numerous tasks, improving both the accuracy and efficiency of forecasting processes is crucial.

Traditionally, load forecasting has relied on statistical methods that estimate relationships within time-series data, sometimes incorporating external factors. Classical statistical models such as ARIMAÂ (Chodakowska etÂ al., [2021](https://arxiv.org/html/2502.08376v1#bib.bib14)) and exponential smoothingÂ (Lima etÂ al., [2019](https://arxiv.org/html/2502.08376v1#bib.bib37); Taylor, [2009](https://arxiv.org/html/2502.08376v1#bib.bib48)), as well as various machine learning approachesÂ (Bunnoon etÂ al., [2010](https://arxiv.org/html/2502.08376v1#bib.bib11)), have been utilized. In addition, load forecasting is influenced by numerous factors, many of which are location-specific and depend on the equipment usedÂ (AlÂ Mamun etÂ al., [2020](https://arxiv.org/html/2502.08376v1#bib.bib6)). Factors such as weather conditions, demographics, socioeconomic variables, and special events such as holidays that can impact power demand are often incorporated as input variables in forecast modelsÂ (Ahmed etÂ al., [2020](https://arxiv.org/html/2502.08376v1#bib.bib5)).

In modern electricity markets, forecasts are generated for various time horizons, each supporting grid operations and market management. Very short-term forecasts, spanning seconds to minutes, are critical for emergency operations and optimizing Distributed Energy Resources (DER). Short-term forecasts, ranging from minutes to a day ahead, are vital for real-time market trading, power plant operations, grid balancing, and managing operating reserves. Medium-term forecasts, covering days to months, aid in pre-dispatch, unit commitment, and maintenance planning, while long-term forecasts, extending months to years, support system planning, investment decisions, and maintenance schedulingÂ (Pinson, [2013](https://arxiv.org/html/2502.08376v1#bib.bib45); Zavadil, [2013](https://arxiv.org/html/2502.08376v1#bib.bib55); Dannecker, [2015](https://arxiv.org/html/2502.08376v1#bib.bib15)).

Among these, short-term load forecasting (STLF) is particularly crucial for daily balancing in grid operations and facilitates real-time decision-making for unit dispatching, peak load analysis, and automatic generation control, ensuring efficient grid operation under dynamic conditions, especially as RES are integratedÂ (Zavadil, [2013](https://arxiv.org/html/2502.08376v1#bib.bib55); Dannecker, [2015](https://arxiv.org/html/2502.08376v1#bib.bib15); Kyriakides and Polycarpou, [2007](https://arxiv.org/html/2502.08376v1#bib.bib35); Venayagamoorthy etÂ al., [2012](https://arxiv.org/html/2502.08376v1#bib.bib50)).

Artificial intelligence (AI) techniques and deep neural networks (DNNs) have gained popularity for load forecasting, including expert systemsÂ (Rahman and Bhatnagar, [1988](https://arxiv.org/html/2502.08376v1#bib.bib46)), support vector machines (SVM)Â (Dong etÂ al., [2022](https://arxiv.org/html/2502.08376v1#bib.bib18)), fuzzy logicÂ (Pandian etÂ al., [2006](https://arxiv.org/html/2502.08376v1#bib.bib43)), artificial neural networks (ANNs)Â (Kouhi and Keynia, [2013](https://arxiv.org/html/2502.08376v1#bib.bib34)) and long-short-term memory networks (LSTMs)Â (Jin etÂ al., [2022b](https://arxiv.org/html/2502.08376v1#bib.bib30)). However, despite their ability to incorporate external influences, these models have limitations, such as the risk of getting stuck in local minima, overfittingÂ (Zhang etÂ al., [2018](https://arxiv.org/html/2502.08376v1#bib.bib56)), and inability to fully capture complex spatial-temporal dependencies.

To address these challenges, hybrid models that combine multiple forecasting methods have been proposed to further enhance accuracy and reliabilityÂ (AlÂ Mamun etÂ al., [2020](https://arxiv.org/html/2502.08376v1#bib.bib6); Eandi etÂ al., [2022](https://arxiv.org/html/2502.08376v1#bib.bib19); Lin etÂ al., [2021](https://arxiv.org/html/2502.08376v1#bib.bib38)). These hybrid models offer the advantage of capturing the spatial and temporal features of the electricity load while addressing the limitations of individual methods. However, despite their advantages, hybrid models also introduce additional complexity in model design, implementation, and parameter optimizationÂ (AlÂ Mamun etÂ al., [2020](https://arxiv.org/html/2502.08376v1#bib.bib6)).

### Key Challenges with Traditional and State-of-the-Art Load Forecasting Models

  1. (1)

Limited Incorporation of Grid-Specific Features: Traditional load forecasting models, and even the current advanced methods, often overlook or inadequately handle grid-specific information (e.g. transmission capacities, efficiencies), which are essential for capturing power flow constraints and line losses in the grid. These models focus mainly on dynamic data (e.g., energy demand at each location, external covariates, etc.). Failing to incorporate these features limits the modelâ€™s ability to accurately represent real-world grid dynamics, leading to less robust predictions.

  2. (2)

Limited Fusion of Spatial and Temporal Information: Many load forecasting models treat spatial and temporal information separately, either through sequential models (like LSTMs) that ignore spatial context or graph-based models that lack robust temporal modeling. This split approach fails to capture how spatial dependencies and temporal patterns interact, limiting the modelâ€™s ability to adapt to sudden changes in load patterns or energy flow dynamics.

  3. (3)

Lack of Integration of RES Data: The intermittent and variable nature of RES, such as solar and wind, introduces significant uncertainty into load forecasts. Most current models do not account for RES data, leading to predictions that may not accurately reflect fluctuations in energy supply and demand.




In this study, we present a hybrid GAT-LSTM fusion model for STLF that addresses key limitations of existing forecasting approaches. The proposed model integrates grid-specific attributes, handles spatial-temporal data fusion, and incorporates RES data to provide more realistic and reliable load predictions. By capturing complex interactions within modern power grids, our approach offers enhanced accuracy and robustness, making it well suited to the intricacies of todayâ€™s dynamic energy landscape.

The rest of the paper is organized as follows; SectionÂ [2](https://arxiv.org/html/2502.08376v1#S2 "2. Review of Key Concepts â€£ Enhanced Load Forecasting with GAT-LSTM: Leveraging Grid and Temporal Features") presents reviews of key concepts, including energy forecasting techniques and graph neural networks, SectionÂ [3](https://arxiv.org/html/2502.08376v1#S3 "3. Methodology â€£ Enhanced Load Forecasting with GAT-LSTM: Leveraging Grid and Temporal Features") describes the data and introduces our hybrid GAT-LSTM model, SectionÂ [4](https://arxiv.org/html/2502.08376v1#S4 "4. Experiments and Results â€£ Enhanced Load Forecasting with GAT-LSTM: Leveraging Grid and Temporal Features") presents the experimental results and discusses the efficacy of our approach, and finally, SectionÂ [5](https://arxiv.org/html/2502.08376v1#S5 "5. Conclusion, Limitations, and Future Work â€£ Enhanced Load Forecasting with GAT-LSTM: Leveraging Grid and Temporal Features") concludes with future directions.

##  2\. Review of Key Concepts

As discussed, forecasting models predict future energy demand and supply by analyzing historical time-series data and relevant covariates to uncover important dynamics within the data. This section provides a brief overview of some widely employed energy time-series forecasting techniques in the literature.

###  2.1. Traditional Energy Forecasting Models

Traditional statistical methods for energy forecasting, such as autoregressive models and exponential smoothing techniques, are foundational approaches that use historical data to predict future energy demand. These models are widely used for linear and stationary time series data.

Autoregressive models, such as ARMA and its extensions (ARIMA and SARIMA), and exponential smoothing techniques, have been widely used for STLF. ARIMA models relationships among current values, past values, and previous errors, with SARIMA incorporating seasonal differencing for datasets with seasonalityÂ (Dannecker, [2015](https://arxiv.org/html/2502.08376v1#bib.bib15); Makridakis and Hibon, [1997](https://arxiv.org/html/2502.08376v1#bib.bib39)).

Exponential smoothing methods, including Holt-linear and Holt-Winters extensions, use weighted averages of past observations to capture trends and seasonality in non-stationary dataÂ (Lima etÂ al., [2019](https://arxiv.org/html/2502.08376v1#bib.bib37)). These approaches are valued for their simplicity and effectiveness in linear time series dataÂ (Chodakowska etÂ al., [2021](https://arxiv.org/html/2502.08376v1#bib.bib14); Lima etÂ al., [2019](https://arxiv.org/html/2502.08376v1#bib.bib37); Deb etÂ al., [2017](https://arxiv.org/html/2502.08376v1#bib.bib16)). However, they are limited in handling non-linear, high-dimensional data and fail to capture the complex spatial and temporal dependencies inherent in energy systems. These limitations are especially evident in modern energy systems, where demand patterns are shaped by dynamic factors such as weather, renewable energy integration, and network topology.

###  2.2. Deep Learning-Based Energy Forecasting Models

Deep learning, a subfield of machine learning, leverages neural networks to automatically learn patterns from data, making it particularly effective for modeling complex, non-linear relationships in large datasetsÂ (Berriel etÂ al., [2017](https://arxiv.org/html/2502.08376v1#bib.bib8)). Sequence processing models such as recurrent neural networks (RNNs) and long-short-term memory networks (LSTMs) are widely used for time-based predictions, as they retain information across time steps and capture temporal dependenciesÂ (Werbos, [1990](https://arxiv.org/html/2502.08376v1#bib.bib51); Williams and Zipser, [1989](https://arxiv.org/html/2502.08376v1#bib.bib52)).

RNNs model sequential data by updating a hidden state based on the current input and the previous hidden state. While effective for short-term dependencies, they struggle with vanishing or exploding gradients, limiting their ability to capture long-term dependenciesÂ (Networks, [2015](https://arxiv.org/html/2502.08376v1#bib.bib42); Kag and Saligrama, [2021](https://arxiv.org/html/2502.08376v1#bib.bib32); Mozer, [1991](https://arxiv.org/html/2502.08376v1#bib.bib41)).

LSTMs address these challenges through a more sophisticated architecture featuring a memory cell and gating mechanisms (forget, input, and output gates) that regulate information flow. This design enables LSTMs to capture both long-term and short-term dependencies in time-series dataÂ (Gers etÂ al., [2000](https://arxiv.org/html/2502.08376v1#bib.bib22); Hochreiter and Schmidhuber, [1997](https://arxiv.org/html/2502.08376v1#bib.bib27); Gers etÂ al., [2002](https://arxiv.org/html/2502.08376v1#bib.bib23)).

RNNs and LSTMs have been extensively applied to load forecasting, where they effectively model temporal dependencies and dynamic patterns in energy time-series dataÂ (Jin etÂ al., [2022b](https://arxiv.org/html/2502.08376v1#bib.bib30); Bianchi etÂ al., [2017](https://arxiv.org/html/2502.08376v1#bib.bib9); Fang and He, [2023](https://arxiv.org/html/2502.08376v1#bib.bib20); PÃ©rez-Ortiz etÂ al., [2003](https://arxiv.org/html/2502.08376v1#bib.bib44)). However, these models are limited to temporal dependencies and do not account for the spatial interactions inherent in energy systems. To address this limitation, hybrid models such as the proposed GAT-LSTM combine the strengths of LSTMs for temporal modeling with GNNs for spatial dependencies. This integration provides a holistic representation of energy systems by capturing both time-series dynamics and grid topology.

###  2.3. Graph Neural Networks (GNNs) for Energy Forecasting

Recent advancements in deep neural networks (DNNs) have extended their applications to graph-structured (non-Euclidean) data, enabling the modeling of complex relationships inherent in graphsÂ (Gori etÂ al., [2005](https://arxiv.org/html/2502.08376v1#bib.bib24); Scarselli etÂ al., [2008](https://arxiv.org/html/2502.08376v1#bib.bib47); Liao etÂ al., [2021](https://arxiv.org/html/2502.08376v1#bib.bib36)). GNNs are particularly effective in capturing spatial dependencies in domains such as social networks, recommendation systems, and energy systems, where power grids naturally exhibit graph structuresÂ (Hu etÂ al., [2024](https://arxiv.org/html/2502.08376v1#bib.bib28)). By combining graph topology with node and edge attributes, GNNs facilitate the modeling of spatial interactions critical for understanding the dynamics of energy networks.

Graph-Structured Data and GNN Architecture: A graph G=(V,E)ğ��ºğ�‘‰ğ��¸G=(V,E)italic_G = ( italic_V , italic_E ) consists of nodes (Vğ�‘‰Vitalic_V) and edges (Eğ��¸Eitalic_E), which may be directed or undirected. The graph structure is represented by an adjacency matrix Ağ��´Aitalic_A, where aiâ�¢j=1subscriptğ�‘�ğ�‘–ğ�‘—1a_{ij}=1italic_a start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT = 1 if an edge exists between nodes iğ�‘–iitalic_i and jğ�‘—jitalic_j, and aiâ�¢j=0subscriptğ�‘�ğ�‘–ğ�‘—0a_{ij}=0italic_a start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT = 0 otherwise. Node features xisubscriptğ�‘¥ğ�‘–x_{i}italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT are organized into a feature matrix Xâˆˆâ„�NÃ—Fğ�‘‹superscriptâ„�ğ�‘�ğ��¹X\in\mathbb{R}^{N\times F}italic_X âˆˆ blackboard_R start_POSTSUPERSCRIPT italic_N Ã— italic_F end_POSTSUPERSCRIPT, where Nğ�‘�Nitalic_N is the number of nodes and Fğ��¹Fitalic_F is the number of features per node.

The core operation in GNNs is the graph convolution, where nodes aggregate information from their neighbors to update their representations. At layer lğ�‘™litalic_l, this is defined as:

(1) |  | hi(l+1)=Agg(hj(l):jâˆˆN(i)âˆªi)h_{i}^{(l+1)}=\text{Agg}\left({h_{j}^{(l)}:j\in N(i)\cup{i}}\right)italic_h start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_l + 1 ) end_POSTSUPERSCRIPT = Agg ( italic_h start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_l ) end_POSTSUPERSCRIPT : italic_j âˆˆ italic_N ( italic_i ) âˆª italic_i ) |   
---|---|---|---  
  
where hi(l)superscriptsubscriptâ„�ğ�‘–ğ�‘™h_{i}^{(l)}italic_h start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_l ) end_POSTSUPERSCRIPT is the embedding of node iğ�‘–iitalic_i at layer lğ�‘™litalic_l, Nâ�¢(i)ğ�‘�ğ�‘–N(i)italic_N ( italic_i ) denotes the neighbors of node iğ�‘–iitalic_i, and Agg is an aggregation function (e.g., sum, mean, or max). After aggregation, embeddings are passed through a non-linear activation function, such as ReLU.

![Refer to caption](x1.png) Figure 1.  Neighborhood Aggregation in GNNs. SourceÂ (Jin etÂ al., [2022a](https://arxiv.org/html/2502.08376v1#bib.bib31)).

FigureÂ [1](https://arxiv.org/html/2502.08376v1#S2.F1 "Figure 1 â€£ 2.3. Graph Neural Networks \(GNNs\) for Energy Forecasting â€£ 2. Review of Key Concepts â€£ Enhanced Load Forecasting with GAT-LSTM: Leveraging Grid and Temporal Features") illustrates neighborhood aggregation in GNNs. For energy systems, nodes may represent grid regions or substations, and edges capture relationships like shared demand or transmission lines. This process enables GNNs to learn spatial dependencies and provide context-aware predictions.

Spectral- and Spatial-Based GNNs: GNNs can be broadly categorized into spectral or spatial-based Â (Bruna etÂ al., [2013](https://arxiv.org/html/2502.08376v1#bib.bib10); Henaff etÂ al., [2015](https://arxiv.org/html/2502.08376v1#bib.bib26); Chiang etÂ al., [2019](https://arxiv.org/html/2502.08376v1#bib.bib13); Hamilton etÂ al., [2017](https://arxiv.org/html/2502.08376v1#bib.bib25); VeliÄ�koviÄ‡ etÂ al., [2017](https://arxiv.org/html/2502.08376v1#bib.bib49)). Spectral-based GNNs, such as Graph Convolutional Networks (GCNs), apply filters in the Fourier domain Â (Kipf and Welling, [2016](https://arxiv.org/html/2502.08376v1#bib.bib33); Yao etÂ al., [2019](https://arxiv.org/html/2502.08376v1#bib.bib54); Cao etÂ al., [2022](https://arxiv.org/html/2502.08376v1#bib.bib12)), leveraging the normalized adjacency matrix A^^ğ��´\hat{A}over^ start_ARG italic_A end_ARG to update node embeddings:

(2) |  | H(l+1)=Ïƒâ�¢(A^â�¢H(l)â�¢W(l))superscriptğ��»ğ�‘™1ğ�œ�^ğ��´superscriptğ��»ğ�‘™superscriptğ�‘Šğ�‘™H^{(l+1)}=\sigma(\hat{A}H^{(l)}W^{(l)})italic_H start_POSTSUPERSCRIPT ( italic_l + 1 ) end_POSTSUPERSCRIPT = italic_Ïƒ ( over^ start_ARG italic_A end_ARG italic_H start_POSTSUPERSCRIPT ( italic_l ) end_POSTSUPERSCRIPT italic_W start_POSTSUPERSCRIPT ( italic_l ) end_POSTSUPERSCRIPT ) |   
---|---|---|---  
  
where H(l)superscriptğ��»ğ�‘™H^{(l)}italic_H start_POSTSUPERSCRIPT ( italic_l ) end_POSTSUPERSCRIPT is the feature matrix at layer lğ�‘™litalic_l, W(l)superscriptğ�‘Šğ�‘™W^{(l)}italic_W start_POSTSUPERSCRIPT ( italic_l ) end_POSTSUPERSCRIPT is the weight matrix, and Ïƒğ�œ�\sigmaitalic_Ïƒ is the activation function. GCNs are effective for tasks requiring localized information aggregationÂ (Liao etÂ al., [2021](https://arxiv.org/html/2502.08376v1#bib.bib36); Mansoor etÂ al., [2024](https://arxiv.org/html/2502.08376v1#bib.bib40)).

Spatial-based GNNs, such as Graph Attention Networks (GATs), use attention mechanisms to assign importance to neighboring nodes during aggregationÂ (Bahdanau, [2014](https://arxiv.org/html/2502.08376v1#bib.bib7); Gehring etÂ al., [2016](https://arxiv.org/html/2502.08376v1#bib.bib21)). The attention coefficient Î±iâ�¢jsubscriptğ�›¼ğ�‘–ğ�‘—\alpha_{ij}italic_Î± start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT is computed as:

(3) |  | Î±iâ�¢j=exp(Ïƒ(aT[Whi||Whj]))âˆ‘kâˆˆNâ�¢(i)exp(Ïƒ(aT[Whi||Whk]))\alpha_{ij}=\frac{\exp(\sigma(a^{T}[Wh_{i}||Wh_{j}]))}{\sum_{k\in N(i)}\exp(% \sigma(a^{T}[Wh_{i}||Wh_{k}]))}italic_Î± start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT = divide start_ARG roman_exp ( italic_Ïƒ ( italic_a start_POSTSUPERSCRIPT italic_T end_POSTSUPERSCRIPT [ italic_W italic_h start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT | | italic_W italic_h start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ] ) ) end_ARG start_ARG âˆ‘ start_POSTSUBSCRIPT italic_k âˆˆ italic_N ( italic_i ) end_POSTSUBSCRIPT roman_exp ( italic_Ïƒ ( italic_a start_POSTSUPERSCRIPT italic_T end_POSTSUPERSCRIPT [ italic_W italic_h start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT | | italic_W italic_h start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT ] ) ) end_ARG |   
---|---|---|---  
  
where ağ�‘�aitalic_a is a learnable attention vector and ||||| | denotes concatenation. Multi-head attention refines this process further, enhancing robustnessÂ (Huang etÂ al., [2023](https://arxiv.org/html/2502.08376v1#bib.bib29)).

Applications and Limitations in Load Forecasting: GNNs, including GCNs and GATs, have been used to improve load forecasting by modeling grid topology and spatial dependenciesÂ (Mansoor etÂ al., [2024](https://arxiv.org/html/2502.08376v1#bib.bib40); Liao etÂ al., [2021](https://arxiv.org/html/2502.08376v1#bib.bib36); Huang etÂ al., [2023](https://arxiv.org/html/2502.08376v1#bib.bib29)). However, GNNs alone cannot capture temporal dependencies inherent in energy systems, which are critical for forecasting dynamic load patterns.

###  2.4. Hybrid Models for Energy Forecasting

Hybrid models, such as the proposed GAT-LSTM, address the limitations of standalone GNNs and LSTMs by combining their strengths. While GATs model spatial dependencies in grid topology, LSTMs handle temporal dynamics by capturing sequential patterns in energy consumption. This integration enables a holistic representation of energy systems, accounting for both spatial interactions and time-series dynamicsÂ (Mansoor etÂ al., [2024](https://arxiv.org/html/2502.08376v1#bib.bib40); Eandi etÂ al., [2022](https://arxiv.org/html/2502.08376v1#bib.bib19); Lin etÂ al., [2021](https://arxiv.org/html/2502.08376v1#bib.bib38); Huang etÂ al., [2023](https://arxiv.org/html/2502.08376v1#bib.bib29)).

The novelty of the GAT-LSTM lies in its ability to adapt graph-structured relationships and temporal patterns simultaneously. For example, GATs use attention mechanisms to focus on the most relevant grid regions, while LSTMs capture the evolving energy consumption trends. Together, they synergize to improve forecasting accuracy, particularly for complex, interconnected energy networks where traditional models and standalone deep learning methods fall short.

This hybrid approach has been extensively studied in domains like traffic forecastingÂ (Wu etÂ al., [2018](https://arxiv.org/html/2502.08376v1#bib.bib53); Zhu etÂ al., [2024](https://arxiv.org/html/2502.08376v1#bib.bib58); Zhang and Guo, [2020](https://arxiv.org/html/2502.08376v1#bib.bib57)), but remains underexplored in energy systems. By incorporating grid-specific features and leveraging the spatio-temporal interplay, the GAT-LSTM provides robust and context-aware predictions for energy time series forecasting, marking a significant advancement in the field.

##  3\. Methodology

###  3.1. Data

####  3.1.1. Data Source and Description:

For this study, we used the Brazilian power system as a case study with actual data on various aspects of the power system and covariate factors. The data sources and description is given in TablesÂ [1](https://arxiv.org/html/2502.08376v1#S3.T1 "Table 1 â€£ 3.1.1. Data Source and Description: â€£ 3.1. Data â€£ 3. Methodology â€£ Enhanced Load Forecasting with GAT-LSTM: Leveraging Grid and Temporal Features") andÂ [2](https://arxiv.org/html/2502.08376v1#S3.T2 "Table 2 â€£ 3.1.1. Data Source and Description: â€£ 3.1. Data â€£ 3. Methodology â€£ Enhanced Load Forecasting with GAT-LSTM: Leveraging Grid and Temporal Features"), respectively. Except for the graph data, which are static, all other aspects of the dataset represent 2-years worth of data from 2019â€“2020.

Table 1. Data Sources Data | Details | Source  
---|---|---  
Electricity | Load, PV, wind, etc. | (Deng etÂ al., [2023](https://arxiv.org/html/2502.08376v1#bib.bib17))  
Grid | Line length, capacity, efficiency, etc. | (Deng etÂ al., [2023](https://arxiv.org/html/2502.08376v1#bib.bib17))  
Weather | Temperature, pressure, rainfall, etc. | (Bra, [2024a](https://arxiv.org/html/2502.08376v1#bib.bib2))  
Socio-economic | State-wise GDP | (Bra, [2024c](https://arxiv.org/html/2502.08376v1#bib.bib4))  
Population | State-wise population | (Bra, [2024b](https://arxiv.org/html/2502.08376v1#bib.bib3))  
Table 2. Data Description Category |  Details | Horizon  
---|---|---  
Sequence Data |  Atmospheric pressure, Total hourly rain, Global radiation, Air temperature, Dew point temperature, Relative humidity, Wind direction, Wind maximum gust, Wind speed, PV generation, Onshore wind generation, Offshore wind generation, Load profile (by consumption) | 1-hour  
Graph Data |  Node features: Source (state), Target (state), PV potential, Onshore wind potential, Offshore wind potential, geometry (longitude & latitude)   
Edge attributes: Line capacity, Line efficiency, Line length, Line carrier | Static  
Socio-economic |  Population, GDP, Total plant capacity | Annual  
Calendar |  Year, Quarter, Month, Day, Hour, Day-of-week, Week-of-Year, Holiday, Season | Annual  
  
####  3.1.2. Data Preprocessing:

The data were cleaned and preprocessed for modeling through the following steps:

  1. (1)

Consolidation of Weather Data: The weather data consisted of observations from various stations in different states of Brazil, with some states having multiple stations. To ensure consistent hourly data for each weather variable per state, we calculated the mean and standard deviation (STD) of each variable, grouped by state and datetime.

  2. (2)

Missing Value Imputation: Analysis showed approximately 21.42% of missing data across the weather dataset, which was filled by interpolation of time series to maintain continuity in the data.

  3. (3)

Handling Insufficient Data: One state had only a single weather station, resulting in NaNs for the STD values. We addressed this by setting the STD values for that state to zero.

  4. (4)

Negative Value Correction: Negative values were observed in the PV variable, which were corrected by replacing them with zero.

  5. (5)

Data Scaling: To handle variance and outliers, we scaled the dataset using RobustScaler from the Python Scikit-learn package.

  6. (6)

Data Splitting: A dynamic split was performed to create training, validation, and test sets as follows:

     * â€¢

Training set: January-December 2019

     * â€¢

Validation set: Januaryâ€“June 2020

     * â€¢

Test set: Julyâ€“December 2020

  7. (7)

Target Variable Creation: Target values for prediction were created by shifting the current load by one hour to facilitate next-hour load forecasting.

  8. (8)

State-wise Sequencing: Finally, we generated state-wise sequences, aligning data by state and timestamp to feed into the model.




###  3.2. Problem Definition and Model Architecture

Let X={x1,x2,â€¦,xp}ğ�‘‹subscriptğ�‘¥1subscriptğ�‘¥2â€¦subscriptğ�‘¥ğ�‘�X=\\{x_{1},x_{2},\dots,x_{p}\\}italic_X = { italic_x start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_x start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , â€¦ , italic_x start_POSTSUBSCRIPT italic_p end_POSTSUBSCRIPT } represent a historical sequence of multidimensional variables in pğ�‘�pitalic_p time steps, where each xtâˆˆâ„�Nsubscriptğ�‘¥ğ�‘¡superscriptâ„�ğ�‘�x_{t}\in\mathbb{R}^{N}italic_x start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT âˆˆ blackboard_R start_POSTSUPERSCRIPT italic_N end_POSTSUPERSCRIPT is a vector of Nğ�‘�Nitalic_N features at time tğ�‘¡titalic_t. Each vector xtsubscriptğ�‘¥ğ�‘¡x_{t}italic_x start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT includes the power load at time tğ�‘¡titalic_t and other covariates that influence future load (e.g., temperature, seasonal and holiday variables, etc.). The objective of this study is to forecast the next hourâ€™s power load, yp+1subscriptğ�‘¦ğ�‘�1y_{p+1}italic_y start_POSTSUBSCRIPT italic_p + 1 end_POSTSUBSCRIPT, by leveraging these historical data alongside the spatial grid information and RES data.

![Refer to caption](x2.png) Figure 2. Model Architecture

To achieve this, we propose a hybrid GAT-LSTM model 111Our model code is available under an open license: <https://github.com/ugoorji12/Load-Forecasting-using-GAT-LSTM> as shown in FigureÂ [2](https://arxiv.org/html/2502.08376v1#S3.F2 "Figure 2 â€£ 3.2. Problem Definition and Model Architecture â€£ 3. Methodology â€£ Enhanced Load Forecasting with GAT-LSTM: Leveraging Grid and Temporal Features") and described below:

#### Phase 1 (Get Node Embeddings):

The model employs 2 parallel GAT layers to compute node embeddings that reflect spatial dependencies and grid-specific constraints. Unlike traditional attention mechanisms, which typically rely only on node features, our approach also incorporates edge attributes (i.e., line capacities, efficiencies, etc.) into the calculation of attention coefficients. Thus, we modify EquationÂ [3](https://arxiv.org/html/2502.08376v1#S2.E3 "In 2.3. Graph Neural Networks \(GNNs\) for Energy Forecasting â€£ 2. Review of Key Concepts â€£ Enhanced Load Forecasting with GAT-LSTM: Leveraging Grid and Temporal Features") to incorporate edge attributes such that, for a pair of connected nodes iğ�‘–iitalic_i and jğ�‘—jitalic_j with respective node features hisubscriptâ„�ğ�‘–h_{i}italic_h start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT and hjsubscriptâ„�ğ�‘—h_{j}italic_h start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT and edge attribute eiâ�¢jsubscriptğ�‘’ğ�‘–ğ�‘—e_{ij}italic_e start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT, the attention coefficient Î±iâ�¢jsubscriptğ�›¼ğ�‘–ğ�‘—\alpha_{ij}italic_Î± start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT becomes:

(4) |  | Î±iâ�¢jk=expâ�¡(Ïƒâ�¢(akâ‹…[Wkâ�¢hiâ€²â�¢â€–Wkâ�¢hjâ€²â€–â�¢Ukâ�¢eiâ�¢jâ€²]))âˆ‘kâˆˆNâ�¢(i)expâ�¡(Ïƒâ�¢(akâ‹…[Wkâ�¢hiâ€²â�¢â€–Wkâ�¢hkâ€²â€–â�¢Ukâ�¢eiâ�¢kâ€²]))superscriptsubscriptğ�›¼ğ�‘–ğ�‘—ğ�‘˜ğ�œ�â‹…superscriptğ�‘�ğ�‘˜delimited-[]superscriptğ�‘Šğ�‘˜superscriptsubscriptâ„�ğ�‘–â€²normsuperscriptğ�‘Šğ�‘˜superscriptsubscriptâ„�ğ�‘—â€²superscriptğ�‘ˆğ�‘˜superscriptsubscriptğ�‘’ğ�‘–ğ�‘—â€²subscriptğ�‘˜ğ�‘�ğ�‘–ğ�œ�â‹…superscriptğ�‘�ğ�‘˜delimited-[]superscriptğ�‘Šğ�‘˜superscriptsubscriptâ„�ğ�‘–â€²normsuperscriptğ�‘Šğ�‘˜superscriptsubscriptâ„�ğ�‘˜â€²superscriptğ�‘ˆğ�‘˜superscriptsubscriptğ�‘’ğ�‘–ğ�‘˜â€²\alpha_{ij}^{k}=\frac{\exp\left(\sigma\left(a^{k}\cdot[W^{k}h_{i}^{{}^{\prime}% }\parallel W^{k}h_{j}^{{}^{\prime}}\parallel U^{k}e_{ij}^{{}^{\prime}}]\right)% \right)}{\sum_{k\in N(i)}\exp\left(\sigma\left(a^{k}\cdot[W^{k}h_{i}^{{}^{% \prime}}\parallel W^{k}h_{k}^{{}^{\prime}}\parallel U^{k}e_{ik}^{{}^{\prime}}]% \right)\right)}italic_Î± start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_k end_POSTSUPERSCRIPT = divide start_ARG roman_exp ( italic_Ïƒ ( italic_a start_POSTSUPERSCRIPT italic_k end_POSTSUPERSCRIPT â‹… [ italic_W start_POSTSUPERSCRIPT italic_k end_POSTSUPERSCRIPT italic_h start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT start_FLOATSUPERSCRIPT â€² end_FLOATSUPERSCRIPT end_POSTSUPERSCRIPT âˆ¥ italic_W start_POSTSUPERSCRIPT italic_k end_POSTSUPERSCRIPT italic_h start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT start_FLOATSUPERSCRIPT â€² end_FLOATSUPERSCRIPT end_POSTSUPERSCRIPT âˆ¥ italic_U start_POSTSUPERSCRIPT italic_k end_POSTSUPERSCRIPT italic_e start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT start_FLOATSUPERSCRIPT â€² end_FLOATSUPERSCRIPT end_POSTSUPERSCRIPT ] ) ) end_ARG start_ARG âˆ‘ start_POSTSUBSCRIPT italic_k âˆˆ italic_N ( italic_i ) end_POSTSUBSCRIPT roman_exp ( italic_Ïƒ ( italic_a start_POSTSUPERSCRIPT italic_k end_POSTSUPERSCRIPT â‹… [ italic_W start_POSTSUPERSCRIPT italic_k end_POSTSUPERSCRIPT italic_h start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT start_FLOATSUPERSCRIPT â€² end_FLOATSUPERSCRIPT end_POSTSUPERSCRIPT âˆ¥ italic_W start_POSTSUPERSCRIPT italic_k end_POSTSUPERSCRIPT italic_h start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT start_POSTSUPERSCRIPT start_FLOATSUPERSCRIPT â€² end_FLOATSUPERSCRIPT end_POSTSUPERSCRIPT âˆ¥ italic_U start_POSTSUPERSCRIPT italic_k end_POSTSUPERSCRIPT italic_e start_POSTSUBSCRIPT italic_i italic_k end_POSTSUBSCRIPT start_POSTSUPERSCRIPT start_FLOATSUPERSCRIPT â€² end_FLOATSUPERSCRIPT end_POSTSUPERSCRIPT ] ) ) end_ARG |   
---|---|---|---  
  
Where:

  * â€¢

Wksuperscriptğ�‘Šğ�‘˜W^{k}italic_W start_POSTSUPERSCRIPT italic_k end_POSTSUPERSCRIPT = learnable transformation matrix specific to attention head kğ�‘˜kitalic_k of node features hiâ€²superscriptsubscriptâ„�ğ�‘–â€²h_{i}^{{}^{\prime}}italic_h start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT start_FLOATSUPERSCRIPT â€² end_FLOATSUPERSCRIPT end_POSTSUPERSCRIPT

  * â€¢

Uksuperscriptğ�‘ˆğ�‘˜U^{k}italic_U start_POSTSUPERSCRIPT italic_k end_POSTSUPERSCRIPT = learnable transformation matrix specific to attention head kğ�‘˜kitalic_k of edge attributes eiâ�¢jâ€²superscriptsubscriptğ�‘’ğ�‘–ğ�‘—â€²e_{ij}^{{}^{\prime}}italic_e start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT start_FLOATSUPERSCRIPT â€² end_FLOATSUPERSCRIPT end_POSTSUPERSCRIPT

  * â€¢

aksuperscriptğ�‘�ğ�‘˜a^{k}italic_a start_POSTSUPERSCRIPT italic_k end_POSTSUPERSCRIPT = learnable vector that projects the concatenated inputs into a scalar (used for attention scoring)

  * â€¢

Ïƒğ�œ�\sigmaitalic_Ïƒ = LeakyReLU activation function




Based on the multi-head attention mechanism, the aggregation and update function for each head is given as:

(5) |  | hiâ€²=Concatâ�¢(Ïƒâ�¢(âˆ‘jâˆˆNâ�¢(i)Î±iâ�¢jkâ�¢Wkâ�¢hjâ€²),âˆ€kâˆˆ{1,â€¦,K})superscriptsubscriptâ„�ğ�‘–â€²Concatğ�œ�subscriptğ�‘—ğ�‘�ğ�‘–superscriptsubscriptğ�›¼ğ�‘–ğ�‘—ğ�‘˜superscriptğ�‘Šğ�‘˜superscriptsubscriptâ„�ğ�‘—â€²for-allğ�‘˜1â€¦ğ��¾h_{i}^{{}^{\prime}}=\text{Concat}\left(\sigma\left(\sum_{j\in N(i)}\alpha_{ij}% ^{k}W^{k}h_{j}^{{}^{\prime}}\right),\forall k\in\\{1,\dots,K\\}\right)italic_h start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT start_FLOATSUPERSCRIPT â€² end_FLOATSUPERSCRIPT end_POSTSUPERSCRIPT = Concat ( italic_Ïƒ ( âˆ‘ start_POSTSUBSCRIPT italic_j âˆˆ italic_N ( italic_i ) end_POSTSUBSCRIPT italic_Î± start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_k end_POSTSUPERSCRIPT italic_W start_POSTSUPERSCRIPT italic_k end_POSTSUPERSCRIPT italic_h start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT start_FLOATSUPERSCRIPT â€² end_FLOATSUPERSCRIPT end_POSTSUPERSCRIPT ) , âˆ€ italic_k âˆˆ { 1 , â€¦ , italic_K } ) |   
---|---|---|---  
  
This formulation allows the model to dynamically learn the importance of each connection based on both node and edge attributes, yielding node embeddings that better capture the underlying power grid structure and constraints.

#### Phase 2 (Early Fusion of Spatial and Temporal Data):

The goal of this phase is to combine the graph-derived node embeddings hiâ€²superscriptsubscriptâ„�ğ�‘–â€²h_{i}^{{}^{\prime}}italic_h start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT start_FLOATSUPERSCRIPT â€² end_FLOATSUPERSCRIPT end_POSTSUPERSCRIPT from the GAT layer (Phase 1) with temporal sequence data. We achieve this by first expanding the hiâ€²superscriptsubscriptâ„�ğ�‘–â€²h_{i}^{{}^{\prime}}italic_h start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT start_FLOATSUPERSCRIPT â€² end_FLOATSUPERSCRIPT end_POSTSUPERSCRIPT along the temporal dimension to match the sequence data as represented in EquationÂ [6](https://arxiv.org/html/2502.08376v1#S3.E6 "In Phase 2 \(Early Fusion of Spatial and Temporal Data\): â€£ 3.2. Problem Definition and Model Architecture â€£ 3. Methodology â€£ Enhanced Load Forecasting with GAT-LSTM: Leveraging Grid and Temporal Features").

(6) |  | Zi=[xi1hiâ€²xi2hiâ€²â‹®â‹®xiThiâ€²],Ziâˆˆâ„�TÃ—(ds+dg)formulae-sequencesubscriptğ�‘�ğ�‘–matrixsuperscriptsubscriptğ�‘¥ğ�‘–1superscriptsubscriptâ„�ğ�‘–â€²superscriptsubscriptğ�‘¥ğ�‘–2superscriptsubscriptâ„�ğ�‘–â€²â‹®â‹®superscriptsubscriptğ�‘¥ğ�‘–ğ�‘‡superscriptsubscriptâ„�ğ�‘–â€²subscriptğ�‘�ğ�‘–superscriptâ„�ğ�‘‡subscriptğ�‘‘ğ�‘ subscriptğ�‘‘ğ�‘”Z_{i}=\begin{bmatrix}x_{i}^{1}&h_{i}^{{}^{\prime}}\\\ x_{i}^{2}&h_{i}^{{}^{\prime}}\\\ \vdots&\vdots\\\ x_{i}^{T}&h_{i}^{{}^{\prime}}\end{bmatrix},\,Z_{i}\in\mathbb{R}^{T\times(d_{s}% +d_{g})}italic_Z start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT = [ start_ARG start_ROW start_CELL italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 1 end_POSTSUPERSCRIPT end_CELL start_CELL italic_h start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT start_FLOATSUPERSCRIPT â€² end_FLOATSUPERSCRIPT end_POSTSUPERSCRIPT end_CELL end_ROW start_ROW start_CELL italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT end_CELL start_CELL italic_h start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT start_FLOATSUPERSCRIPT â€² end_FLOATSUPERSCRIPT end_POSTSUPERSCRIPT end_CELL end_ROW start_ROW start_CELL â‹® end_CELL start_CELL â‹® end_CELL end_ROW start_ROW start_CELL italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_T end_POSTSUPERSCRIPT end_CELL start_CELL italic_h start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT start_FLOATSUPERSCRIPT â€² end_FLOATSUPERSCRIPT end_POSTSUPERSCRIPT end_CELL end_ROW end_ARG ] , italic_Z start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT âˆˆ blackboard_R start_POSTSUPERSCRIPT italic_T Ã— ( italic_d start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT + italic_d start_POSTSUBSCRIPT italic_g end_POSTSUBSCRIPT ) end_POSTSUPERSCRIPT |   
---|---|---|---  
  
The final concatenation is given as:

(7) |  | Zi=Concatâ�¢(Xi,Repeatâ�¢(hiâ€²,T)),Ziâˆˆâ„�TÃ—(ds+dg)formulae-sequencesubscriptğ�‘�ğ�‘–Concatsubscriptğ�‘‹ğ�‘–Repeatsuperscriptsubscriptâ„�ğ�‘–â€²ğ�‘‡subscriptğ�‘�ğ�‘–superscriptâ„�ğ�‘‡subscriptğ�‘‘ğ�‘ subscriptğ�‘‘ğ�‘”Z_{i}=\text{Concat}(X_{i},\text{Repeat}(h_{i}^{{}^{\prime}},T)),\,Z_{i}\in% \mathbb{R}^{T\times(d_{s}+d_{g})}italic_Z start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT = Concat ( italic_X start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT , Repeat ( italic_h start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT start_FLOATSUPERSCRIPT â€² end_FLOATSUPERSCRIPT end_POSTSUPERSCRIPT , italic_T ) ) , italic_Z start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT âˆˆ blackboard_R start_POSTSUPERSCRIPT italic_T Ã— ( italic_d start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT + italic_d start_POSTSUBSCRIPT italic_g end_POSTSUBSCRIPT ) end_POSTSUPERSCRIPT |   
---|---|---|---  
  
Where:

  * â€¢

hiâ€²âˆˆâ„�dgsuperscriptsubscriptâ„�ğ�‘–â€²superscriptâ„�subscriptğ�‘‘ğ�‘”h_{i}^{{}^{\prime}}\in\mathbb{R}^{d_{g}}italic_h start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT start_FLOATSUPERSCRIPT â€² end_FLOATSUPERSCRIPT end_POSTSUPERSCRIPT âˆˆ blackboard_R start_POSTSUPERSCRIPT italic_d start_POSTSUBSCRIPT italic_g end_POSTSUBSCRIPT end_POSTSUPERSCRIPT = The graph-derived embedding for node iğ�‘–iitalic_i, where dgsubscriptğ�‘‘ğ�‘”d_{g}italic_d start_POSTSUBSCRIPT italic_g end_POSTSUBSCRIPT is the GAT output dimension (after concatenation across heads).

  * â€¢

Xi={xit}t=1Tâˆˆâ„�TÃ—dssubscriptğ�‘‹ğ�‘–superscriptsubscriptsuperscriptsubscriptğ�‘¥ğ�‘–ğ�‘¡ğ�‘¡1ğ�‘‡superscriptâ„�ğ�‘‡subscriptğ�‘‘ğ�‘ X_{i}=\\{x_{i}^{t}\\}_{t=1}^{T}\in\mathbb{R}^{T\times d_{s}}italic_X start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT = { italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT } start_POSTSUBSCRIPT italic_t = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_T end_POSTSUPERSCRIPT âˆˆ blackboard_R start_POSTSUPERSCRIPT italic_T Ã— italic_d start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT end_POSTSUPERSCRIPT = The temporal sequence data for node iğ�‘–iitalic_i, Tğ�‘‡Titalic_T is the sequence length (e.g., 24 hours), and dssubscriptğ�‘‘ğ�‘ d_{s}italic_d start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT is the feature dimension of the sequence data.




This early fusion strategy ensures that both spatial and temporal information are jointly modeled in the downstream LSTM layer.

#### Phase 3 (LSTM Layer for Sequential Processing):

The combined spatial-temporal data are fed into an LSTM layer, which learns the combined dependencies, capturing how past load trends and covariates evolve over time to influence future loads. This layer enables the model to retain and process the long-term spatio-temporal dependencies in the data.

#### Phase 4 (Final Prediction Layer):

The output from the LSTM layer is fed into a fully connected layer that generates the forecast for the next hourâ€™s load, yp+1subscriptğ�‘¦ğ�‘�1y_{p+1}italic_y start_POSTSUBSCRIPT italic_p + 1 end_POSTSUBSCRIPT.

###  3.3. Training Process

#### Loss Function and Optimization:

The training process used the mean squared error (MSE) as the loss function for both training and validation. MSE is particularly suited for regression tasks as it heavily penalizes larger errors, ensuring a focus on minimizing significant deviations in predictions.

For optimization, the Adam optimizer was chosen due to its effective combination of momentum and adaptive learning rates, which makes it well suited for handling complex models. Additionally, a ReduceLROnPlateau learning rate scheduler was used to dynamically adjust the learning rate. If the validation loss did not improve for five consecutive epochs, the learning rate was reduced by a factor of 0.1 (90% decrease), promoting more effective convergence.

To prevent overfitting, Early Stopping was applied, halting the training process if validation loss showed no improvement for 10 consecutive epochs. This ensured that the training process was stopped once convergence was achieved.

These techniques work together to ensure efficient learning, smoother convergence, and better generalization.

##  4\. Experiments and Results

###  4.1. Experimental Setup

The experiments were carried out in a high performance computing environment featuring a dual-socket architecture with 12 physical cores (24 logical CPUs) and 192 GB of memory. The system includes 3 NVIDIA Tesla GPUs and ran on Linux (Debian 6.1), with Slurm (version 22.05.8) used for job scheduling. The hyperparameters used in the model were carefully selected, with details provided in TableÂ [3](https://arxiv.org/html/2502.08376v1#S4.T3 "Table 3 â€£ 4.1. Experimental Setup â€£ 4. Experiments and Results â€£ Enhanced Load Forecasting with GAT-LSTM: Leveraging Grid and Temporal Features").

Table 3. Description of Model Parameters Parameter | Value  
---|---  
Sequence length | 24  
Batch size | 27  
GAT output layer | 64  
GAT attention heads | 8  
LSTM hidden-state | 128  
Number of LSTM layers | 4  
Learning rate | 0.0001  
Weight decay | 1Ã—10âˆ’51superscript1051\times 10^{-5}1 Ã— 10 start_POSTSUPERSCRIPT - 5 end_POSTSUPERSCRIPT  
GAT Dropout | 0.2  
LSTM Dropout | 0.3  
Epochs | 200  
  
###  4.2. Baseline Models

The following baseline models were used to compare the performance of our proposed approach:

GCN-LSTM: This hybrid model combines Graph Convolutional Network (GCN) layers with a Long Short-Term Memory (LSTM) network for load forecasting. It mirrors the GAT-LSTM architecture by applying two parallel 1-hop GCN layers, each performing a 1-hop convolution on the node features, followed by dropout for regularization. The outputs of these GCN layers are concatenated and indexed for the nodes corresponding to the input sequences. These combined GCN features are expanded to match the temporal sequence length and concatenated with the sequence data. The LSTM layer captures temporal dependencies, and a fully connected layer generates the final output. Unlike GAT-LSTM, this model does not incorporate edge attributes, as standard GCNs focus exclusively on node features and graph structure.

EdgeGCN-LSTM: An extension of GCN-LSTM, this model integrates edge attributes into the message-passing process using a custom EdgeAttrGCNConv layer. Edge attributes are transformed via a linear layer before aggregation, allowing the model to utilize both node and edge information effectively. The aggregated features are passed to an LSTM layer to capture temporal dynamics and generate predictions.

LSTM: Serving as a sequence forecasting baseline, this model focuses solely on temporal data without incorporating graph-based features. It uses an LSTM network to learn temporal dependencies and produce forecasts based on dynamic time-series data.

XGBoost: A popular tree-based regression model, XGBoost is used here as a baseline for evaluating performance on time-series data. Like the LSTM model, it does not include graph-based features and relies entirely on dynamic data for forecasting.

###  4.3. Evaluation Metrics

Our evaluation metrics include Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and Mean Absolute Percentage Error (MAPE).

Concretely, these metrics are widely used to evaluate the accuracy of single or hybrid load forecasting modelsÂ (Pinson, [2013](https://arxiv.org/html/2502.08376v1#bib.bib45); AlÂ Mamun etÂ al., [2020](https://arxiv.org/html/2502.08376v1#bib.bib6)).

###  4.4. Experiment Results and Discussion

Training Performance and Early Stopping Analysis

![Refer to caption](x3.png) Figure 3. Learning curve for all models.

FigureÂ [3](https://arxiv.org/html/2502.08376v1#S4.F3 "Figure 3 â€£ 4.4. Experiment Results and Discussion â€£ 4. Experiments and Results â€£ Enhanced Load Forecasting with GAT-LSTM: Leveraging Grid and Temporal Features") presents the learning curves for all models during training, highlighting their convergence behavior and potential overfitting risks. GAT-LSTM demonstrates rapid convergence with stable training and validation losses, suggesting efficient training and generalization. Early stopping was applied around optimal epochs to prevent overfitting. In contrast, EdgeGCN-LSTM and GCN-LSTM show steady declines in training loss but exhibit fluctuations in validation loss, indicating minor risks of overfitting. LSTM converges more slowly, with its validation loss stabilizing at a higher value, reflecting its weaker capacity to model the data comprehensively. Lastly, XGBoost displays the poorest training performance, with consistently high errors throughout, underscoring its limitations in capturing temporal dependencies.

Model Comparison Based on Accuracy Metrics

Table 4. Experiment Results

Model | MAE (MW) | RMSE (MW) | MAPE (%)  
---|---|---|---  
GAT-LSTM | 64.64 | 119.06 | 4.59  
LSTM | 82.68 | 141.55 | 5.75  
EdgeGCN-LSTM | 84.63 | 148.09 | 7.24  
GCN-LSTM | 89.11 | 184.12 | 5.72  
XGBoost | 297.47 | 517.69 | 40.50  
  
TableÂ [4](https://arxiv.org/html/2502.08376v1#S4.T4 "Table 4 â€£ 4.4. Experiment Results and Discussion â€£ 4. Experiments and Results â€£ Enhanced Load Forecasting with GAT-LSTM: Leveraging Grid and Temporal Features") summarizes the performance of all models based on MAE, RMSE, and MAPE. GAT-LSTM achieves the best overall accuracy, outperforming other models across all metrics. Specifically, it shows a 21. 82% improvement in MAE compared to LSTM and a 23.62% improvement over EdgeGCN-LSTM. This highlights the ability of GAT-LSTM to effectively capture both spatial and temporal dependencies, leveraging the graph-attention mechanism. LSTM, despite lacking spatial awareness, performs better than both EdgeGCN-LSTM and GCN-LSTM due to its robust temporal modeling capabilities. The underperformance of EdgeGCN-LSTM relative to LSTM suggests that incorporating spatial relationships without attention mechanisms might introduce irrelevant or noisy information, hindering forecasting accuracy. Similarly, GCN-LSTM performs the worst among GNN-based models, indicating that suboptimal spatial features can reduce model effectiveness. Lastly, XGBoost demonstrates the highest errors across all metrics, reflecting its inability to model non-linear temporal dependencies and spatial relationships critical for load forecasting.

Analysis of Mean Actual vs Predicted Load Curves

![Refer to caption](x4.png) Figure 4. Mean Actual vs Predicted Load Values for all Models.

FigureÂ [4](https://arxiv.org/html/2502.08376v1#S4.F4 "Figure 4 â€£ 4.4. Experiment Results and Discussion â€£ 4. Experiments and Results â€£ Enhanced Load Forecasting with GAT-LSTM: Leveraging Grid and Temporal Features") compares the mean actual and predicted load profiles across all models. GAT-LSTM closely follows the actual load curve throughout the day, reflecting its superior accuracy and ability to generalize across varying load conditions. LSTM and EdgeGCN-LSTM show reasonable alignment with the actual load curve but exhibit larger deviations during peak and off-peak transitions, particularly in high-gradient periods. GCN-LSTM struggles more during peak hours, with noticeable deviations during high-load periods. In contrast, XGBoost displays significant errors, including peak and off-peak times, confirming its limited ability to generalize load dynamics effectively.

Performance of GAT-LSTM During Peak vs Off-peak Hours

![Refer to caption](x5.png) Figure 5. Peak vs Off-peak Performance for GAT-LSTM.

FigureÂ [5](https://arxiv.org/html/2502.08376v1#S4.F5 "Figure 5 â€£ 4.4. Experiment Results and Discussion â€£ 4. Experiments and Results â€£ Enhanced Load Forecasting with GAT-LSTM: Leveraging Grid and Temporal Features") illustrates the average performance of GAT-LSTM during peak and off-peak periods across the test set. During peak hours (7:00 AMâ€“10:00 AM and 4:00 PMâ€“7:00 PM), GAT-LSTM consistently tracks actual load trends but exhibits slight deviations during sharp increases, such as the morning ramp-up. Evening peaks are generally better captured, although minor overestimations occur after the peak (e.g., hours 18â€“19). During off-peak hours (12:00 AMâ€“6:00 AM and 8:00 PMâ€“11:00 PM), predictions align well with actual loads, although a pattern of overestimation is observed during early off-peak periods (e.g., hours 2â€“5). This performance demonstrates the modelâ€™s robustness, but also highlights areas for further improvement, such as better handling of high-gradient patterns.

###  4.5. Key Insights and Discussion

The experimental results underscore several critical insights:

Effectiveness of Attention Mechanisms: The GAT-LSTM model demonstrates superior accuracy by effectively capturing spatial and temporal dependencies using attention mechanisms. Its ability to prioritize relevant node and edge attributes enables robust modeling of grid-specific constraints, such as line capacities. However, the model struggles slightly during rapid transitions, such as morning ramp-ups, suggesting that the temporal component could benefit from further fine-tuning. The integration of spatial embeddings and temporal features remains a key strength, enabling GAT-LSTM to model complex interactions with high robustness for real-world electricity forecasting.

Limitations of Basic Graph Architectures: Both EdgeGCN-LSTM and GCN-LSTM perform worse than LSTM, indicating that spatial features, when modeled without attention mechanisms, may introduce noise or irrelevant information, reducing forecasting accuracy.

Temporal Strength of LSTM: Despite its lack of spatial awareness, LSTMâ€™s strong temporal modeling capabilities allow it to outperform EdgeGCN-LSTM and GCN-LSTM. This highlights the importance of robust temporal modeling in energy forecasting tasks.

Limitations of XGBoost: XGBoost exhibits the highest errors, confirming its inability to model sequential relationships and temporal dependencies effectively. Its tree-based approach further limits its ability to capture the non-linear temporal dynamics critical for load forecasting.

##  5\. Conclusion, Limitations, and Future Work

In this paper, we introduce and evaluate the GAT-LSTM model for hourly power load forecasting, combining GAT and LSTM to effectively capture spatial and temporal dependencies in electricity grids. Our results demonstrate that GAT-LSTM consistently outperforms state-of-the-art models, across key metrics such as MAE, RMSE, and MAPE. This superior performance arises from the modelâ€™s ability to leverage graph-based attention mechanisms to extract meaningful spatial features while utilizing the LSTMâ€™s strength in modeling sequential patterns.

Despite its strong performance, the model has notable limitations. The integration of graph attention mechanisms and LSTM layers introduces significant computational complexity, potentially hindering scalability for large datasets or resource-constrained environments. Furthermore, the modelâ€™s accuracy is influenced by the quality of the graph structure; incomplete or inaccurate node connections, often due to missing or imprecise data in electrical grid representations, can reduce its effectiveness. These issues are relatively common in real-world grids due to challenges such as missing data, approximations in topology, and outdated infrastructure records. Additionally, while GAT-LSTM performs well during stable off-peak periods, it struggles to fully capture rapid transitions, such as morning ramp-ups, leading to deviations from actual load values. This limitation is not unique to GAT-LSTM, but reflects a broader challenge in forecasting highly dynamic events, where past information may not adequately represent future behavior.

Future work should address these challenges by enhancing the modelâ€™s ability to handle rapid load transitions during peak hours and reducing sensitivity to low-magnitude variations in off-peak periods. Adaptive learning techniques, automated graph refinement, and the inclusion of additional external covariates, such as market data and maintenance records, could improve the modelâ€™s robustness and contextual understanding. Leveraging attention mechanisms to identify and prioritize critical graph elements, such as influential nodes or edges, can further enhance interpretability and provide deeper insights into the spatial-temporal factors driving predictions. Additionally, integrating uncertainty quantification methods alongside these mechanisms would improve decision-making reliability and broaden the modelâ€™s applicability in real-world scenarios.

In conclusion, the GAT-LSTM model represents a significant advancement in power load forecasting by effectively modeling spatial-temporal relationships in electricity grids. Its ability to identify and prioritize informative graph components allows it to capture critical spatial dependencies that underpin accurate predictions. Addressing its limitations related to computational complexity and peak-hour accuracy, along with further refinements for scalability and reliability, will enhance its practicality. Visualizing the graph structure to identify central or influential nodes could also provide valuable insights, improving both interpretability and optimization. These advancements will solidify GAT-LSTMâ€™s role as a powerful tool for grid management, demand response, and energy planning in dynamic and interconnected energy systems.

###### Acknowledgements.

This research is part of the project Innovation Lab for Utilities on Sustainable Technology and Renewable Energy (ILUSTRE), No.KICH3.LTP.20.006 of the research program LTP ROBUST which is partly financed by the Dutch Research Council (NWO). 

## References

  * (1)
  * Bra (2024a) 2024a.  Brazil Weather Information by INMET.  <https://www.kaggle.com/datasets/gregoryoliveira/brazil-weather-information-by-inmet?resource=download>.  Accessed: 14/06/2024. 
  * Bra (2024b) 2024b.  Estimates of Resident Population for Municipalities and Federation Units â€” IBGE.  <https://www.ibge.gov.br/en/statistics/social/population/18448-estimates-of-resident-population-for-municipalities-and-federation-units.html?edicao=28688>.  Accessed: 14/06/2024. 
  * Bra (2024c) 2024c.  Regional Accounts of Brazil â€” IBGE.  <https://www.ibge.gov.br/en/statistics/economic/national-accounts/16855-regional-accounts-of-brazil.html>.  Accessed: 14/06/2024. 
  * Ahmed etÂ al. (2020) R. Ahmed, V. Sreeram, Y. Mishra, and M.Â D. Arif. 2020.  A review and evaluation of the state-of-the-art in PV solar power forecasting: Techniques and optimization.  _Renewable and Sustainable Energy Reviews_ 124 (2020), 109792. 
  * AlÂ Mamun etÂ al. (2020) A. AlÂ Mamun, M. Sohel, N. Mohammad, M.Â S.Â H. Sunny, D.Â R. Dipta, and E. Hossain. 2020.  A comprehensive review of the load forecasting techniques using single and hybrid predictive models.  _IEEE Access_ 8 (2020), 134911â€“134939. 
  * Bahdanau (2014) Dzmitry Bahdanau. 2014.  Neural machine translation by jointly learning to align and translate.  _arXiv preprint arXiv:1409.0473_ (2014). 
  * Berriel etÂ al. (2017) R.Â F. Berriel, A.Â T. Lopes, A. Rodrigues, F.Â M. Varejao, and T. Oliveira-Santos. 2017.  Monthly energy consumption forecast: A deep learning approach. In _2017 International Joint Conference on Neural Networks (IJCNN)_. IEEE, 4283â€“4290.  <https://doi.org/10.1109/IJCNN.2017.7966398>
  * Bianchi etÂ al. (2017) FilippoÂ Maria Bianchi, Enrico Maiorino, MichaelÂ C Kampffmeyer, Antonello Rizzi, and Robert Jenssen. 2017.  Recurrent neural networks for short-term load forecasting: an overview and comparative analysis.  (2017). 
  * Bruna etÂ al. (2013) Joan Bruna, Wojciech Zaremba, Arthur Szlam, and Yann LeCun. 2013.  Spectral networks and locally connected networks on graphs.  _arXiv preprint arXiv:1312.6203_ (2013). 
  * Bunnoon etÂ al. (2010) P. Bunnoon, K. Chalermyanont, and C. Limsakul. 2010.  A computing model of artificial intelligent approaches to mid-term load forecasting: a state-of-the-art survey for the researcher.  _International Journal of Engineering and Technology_ 2, 1 (2010), 94â€“100. 
  * Cao etÂ al. (2022) Pingping Cao, Zeqi Zhu, Ziyuan Wang, Yanping Zhu, and Qiang Niu. 2022.  Applications of graph convolutional networks in computer vision.  _Neural computing and applications_ 34, 16 (2022), 13387â€“13405. 
  * Chiang etÂ al. (2019) Wei-Lin Chiang, Xuanqing Liu, Si Si, Yang Li, Samy Bengio, and Cho-Jui Hsieh. 2019.  Cluster-gcn: An efficient algorithm for training deep and large graph convolutional networks. In _Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining_. 257â€“266. 
  * Chodakowska etÂ al. (2021) E. Chodakowska, J. Nazarko, and Å� Nazarko. 2021.  Arima models in electrical load forecasting and their robustness to noise.  _Energies_ 14, 23 (2021).  <https://doi.org/10.3390/en14237952>
  * Dannecker (2015) L. Dannecker. 2015.  _Energy time series forecasting: Efficient and accurate forecasting of evolving time series from the energy domain_.  Springer. 
  * Deb etÂ al. (2017) C. Deb, F. Zhang, J. Yang, S.Â E. Lee, and K.Â W. Shah. 2017.  A review on time series forecasting techniques for building energy consumption.  _Renewable and Sustainable Energy Reviews_ 74 (2017), 902â€“924. 
  * Deng etÂ al. (2023) Ying Deng, Karl-KiÃªn Cao, Wenxuan Hu, Ronald Stegen, Kai von Krbek, Rafael Soria, Pedro RuaÂ Rodriguez Rochedo, and Patrick Jochem. 2023.  Harmonized and open energy dataset for modeling a highly renewable Brazilian power system.  _Scientific Data_ 10, 1 (2023), 103. 
  * Dong etÂ al. (2022) X. Dong, S. Deng, and D. Wang. 2022.  A short-term power load forecasting method based on k-means and SVM.  _Journal of Ambient Intelligence and Humanized Computing_ 13, 11 (2022), 5253â€“5267. 
  * Eandi etÂ al. (2022) S. Eandi, A. Cini, S. Lukovic, and C. Alippi. 2022.  Spatio-temporal graph neural networks for aggregate load forecasting. In _2022 International Joint Conference on Neural Networks (IJCNN)_. IEEE, 1â€“8. 
  * Fang and He (2023) L. Fang and B. He. 2023.  A deep learning framework using multi-feature fusion recurrent neural networks for energy consumption forecasting.  _Applied Energy_ 348 (2023), 121563. 
  * Gehring etÂ al. (2016) Jonas Gehring, Michael Auli, David Grangier, and YannÂ N Dauphin. 2016.  A convolutional encoder model for neural machine translation.  _arXiv preprint arXiv:1611.02344_ (2016). 
  * Gers etÂ al. (2000) F.Â A. Gers, J. Schmidhuber, and F. Cummins. 2000.  Learning to forget: Continual prediction with LSTM.  _Neural Computation_ 12, 10 (2000), 2451â€“2471. 
  * Gers etÂ al. (2002) F.Â A. Gers, N.Â N. Schraudolph, and J. Schmidhuber. 2002.  Learning precise timing with LSTM recurrent networks.  _Journal of Machine Learning Research_ 3 (2002), 115â€“143. 
  * Gori etÂ al. (2005) M. Gori, G. Monfardini, and F. Scarselli. 2005.  A new model for learning in graph domains. In _Proceedings of the 2005 IEEE International Joint Conference on Neural Networks_. IEEE, 729â€“734. 
  * Hamilton etÂ al. (2017) Will Hamilton, Zhitao Ying, and Jure Leskovec. 2017.  Inductive representation learning on large graphs.  _Advances in neural information processing systems_ 30 (2017). 
  * Henaff etÂ al. (2015) Mikael Henaff, Joan Bruna, and Yann LeCun. 2015.  Deep convolutional networks on graph-structured data.  _arXiv preprint arXiv:1506.05163_ (2015). 
  * Hochreiter and Schmidhuber (1997) S. Hochreiter and J. Schmidhuber. 1997.  Long short-term memory.  _Neural Computation_ 9, 8 (1997), 1735â€“1780. 
  * Hu etÂ al. (2024) X. Hu, J. Yang, Y. Gao, M. Zhu, Q. Zhang, H. Chen, and J. Zhao. 2024.  Adaptive power flow analysis for power system operation based on graph deep learning.  _International Journal of Electrical Power & Energy Systems_ 161 (2024), 110166. 
  * Huang etÂ al. (2023) Chaokai Huang, Ning Du, Jiahan He, Na Li, Yifan Feng, and Weihong Cai. 2023.  Multidimensional Feature-Based Graph Attention Networks and Dynamic Learning for Electricity Load Forecasting.  _Energies_ 16, 18 (2023), 6443. 
  * Jin etÂ al. (2022b) N. Jin, F. Yang, Y. Mo, Y. Zeng, X. Zhou, K. Yan, and X. Ma. 2022b.  Highly accurate energy consumption forecasting model based on parallel LSTM neural networks.  _Advanced Engineering Informatics_ 51 (2022), 101442. 
  * Jin etÂ al. (2022a) Z. Jin, Y. Wang, Q. Wang, Y. Ming, T. Ma, and H. Qu. 2022a.  Gnnlens: A visual analytics approach for prediction error diagnosis of graph neural networks.  _IEEE Transactions on Visualization and Computer Graphics_ 29, 6 (2022), 3024â€“3038. 
  * Kag and Saligrama (2021) A. Kag and V. Saligrama. 2021.  Training recurrent neural networks via forward propagation through time. In _International Conference on Machine Learning_. PMLR, 5189â€“5200. 
  * Kipf and Welling (2016) ThomasÂ N Kipf and Max Welling. 2016.  Semi-supervised classification with graph convolutional networks.  _arXiv preprint arXiv:1609.02907_ (2016). 
  * Kouhi and Keynia (2013) S. Kouhi and F. Keynia. 2013.  A new cascade NN based method to short-term load forecast in deregulated electricity market.  _Energy Conversion and Management_ 71 (2013), 76â€“83. 
  * Kyriakides and Polycarpou (2007) E. Kyriakides and M. Polycarpou. 2007.  Short-term electric load forecasting: A tutorial.  _Trends in Neural Computation, Studies in Computational Intelligence_ 35 (2007), 391â€“418. 
  * Liao etÂ al. (2021) W. Liao, B. Bak-Jensen, J.Â R. Pillai, Y. Wang, and Y. Wang. 2021.  A review of graph neural networks and their applications in power systems.  _Journal of Modern Power Systems and Clean Energy_ 10, 2 (2021), 345â€“360. 
  * Lima etÂ al. (2019) S. Lima, A.Â M. GonÃ§alves, and M. Costa. 2019.  Time series forecasting using Holt-Winters exponential smoothing: An application to economic data. In _AIP Conference Proceedings_ , Vol.Â 2186. AIP Publishing.  <https://doi.org/10.1063/1.5137999>
  * Lin etÂ al. (2021) W. Lin, D. Wu, and B. Boulet. 2021.  Spatial-temporal residential short-term load forecasting via graph neural networks.  _IEEE Transactions on Smart Grid_ 12, 6 (2021), 5373â€“5384. 
  * Makridakis and Hibon (1997) S. Makridakis and M. Hibon. 1997.  ARMA models and the Boxâ€“Jenkins methodology.  _Journal of Forecasting_ 16, 3 (1997), 147â€“163. 
  * Mansoor etÂ al. (2024) H. Mansoor, M.Â S. Gull, H. Rauf, M. Khalid, and N. Arshad. 2024.  Graph Convolutional Networks based short-term load forecasting: Leveraging spatial information for improved accuracy.  _Electric Power Systems Research_ 230 (2024), 110263. 
  * Mozer (1991) M.Â C. Mozer. 1991.  Induction of multiscale temporal structure. In _Advances in Neural Information Processing Systems_ , Vol.Â 4. 
  * Networks (2015) UnderstandingÂ LSTM Networks. 2015.  _colahâ€™s blog_.  <https://colah.github.io/posts/2015-08-Understanding-LSTMs/> Accessed: 13/09/2024. 
  * Pandian etÂ al. (2006) S.Â C. Pandian, K. Duraiswamy, C.Â C.Â A. Rajan, and N. Kanagaraj. 2006.  Fuzzy approach for short term load forecasting.  _Electric Power Systems Research_ 76, 6â€“7 (2006), 541â€“548. 
  * PÃ©rez-Ortiz etÂ al. (2003) J.Â A. PÃ©rez-Ortiz, F.Â A. Gers, D. Eck, and J. Schmidhuber. 2003.  Kalman filters improve LSTM network performance in problems unsolvable by traditional recurrent nets.  _Neural Networks_ 16, 2 (2003), 241â€“250. 
  * Pinson (2013) P. Pinson. 2013.  Wind energy: Forecasting challenges for its operational management.  _Statist. Sci._ 28, 4 (2013), 564â€“585. 
  * Rahman and Bhatnagar (1988) S. Rahman and R. Bhatnagar. 1988.  An expert system based algorithm for short term load forecast.  _IEEE Transactions on Power Systems_ 3, 2 (1988), 392â€“399. 
  * Scarselli etÂ al. (2008) F. Scarselli, M. Gori, A.Â C. Tsoi, M. Hagenbuchner, and G. Monfardini. 2008.  The graph neural network model.  _IEEE Transactions on Neural Networks_ 20, 1 (2008), 61â€“80. 
  * Taylor (2009) J.W. Taylor. 2009.  Triple seasonal methods for short-term electricity demand forecasting.  _European Journal of Operational Research_ 204, 1 (2009), 139â€“152. 
  * VeliÄ�koviÄ‡ etÂ al. (2017) Petar VeliÄ�koviÄ‡, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio. 2017.  Graph attention networks.  _arXiv preprint arXiv:1710.10903_ (2017). 
  * Venayagamoorthy etÂ al. (2012) G.Â K. Venayagamoorthy, K. Rohrig, and I. Erlich. 2012.  One step ahead: Short-term wind power forecasting and intelligent predictive control based on data analytics.  _IEEE Power and Energy Magazine_ 10, 5 (2012), 70â€“78. 
  * Werbos (1990) P.Â J. Werbos. 1990.  Backpropagation through time: What it does and how to do it.  _Proc. IEEE_ 78, 10 (1990), 1550â€“1560. 
  * Williams and Zipser (1989) R.Â J. Williams and D. Zipser. 1989.  A learning algorithm for continually running fully recurrent neural networks.  _Neural Computation_ 1, 2 (1989), 270â€“280. 
  * Wu etÂ al. (2018) Tianlong Wu, Feng Chen, and Yun Wan. 2018.  Graph attention LSTM network: A new model for traffic flow forecasting. In _2018 5th international conference on information science and control engineering (ICISCE)_. IEEE, 241â€“245. 
  * Yao etÂ al. (2019) Liang Yao, Chengsheng Mao, and Yuan Luo. 2019.  Graph convolutional networks for text classification. In _Proceedings of the AAAI conference on artificial intelligence_ , Vol.Â 33. 7370â€“7377. 
  * Zavadil (2013) R. Zavadil. 2013.  Renewable generation forecasting: The science, applications, and outlook. In _Proceedings of the 46th Hawaii International Conference on System Sciences (HICSS)_. 2252â€“2260. 
  * Zhang etÂ al. (2018) J. Zhang, Y.Â M. Wei, D. Li, Z. Tan, and J. Zhou. 2018.  Short term electricity load forecasting using a hybrid model.  _Energy_ 158 (2018), 774â€“781. 
  * Zhang and Guo (2020) Tianqi Zhang and Ge Guo. 2020.  Graph attention LSTM: A spatiotemporal approach for traffic flow forecasting.  _IEEE Intelligent Transportation Systems Magazine_ 14, 2 (2020), 190â€“196. 
  * Zhu etÂ al. (2024) Taomei Zhu, Maria JesusÂ Lopez Boada, and BeatrizÂ Lopez Boada. 2024.  Adaptive Graph Attention and Long Short-Term Memory-Based Networks for Traffic Prediction.  _Mathematics_ 12, 2 (2024), 255. 


