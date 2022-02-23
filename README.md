# Approximation of complex simulations using deep neural networks on example of tumour simulation

 Approximation of EMT6Ro (https://github.com/banasraf/EMT6-Ro) using deep neural networks trained on dataset (https://raw.githubusercontent.com/mkmkl93/ml-ca/master/data/uniform_200k/dataset1_200.csv). 


How to start?

For deep learning experiments:

step 1 - Download data

    python cancer_nn/preporcess_data.py 

Saves preprocessed dataset to data/data.csv

step 2 - Fill configs/neptune_config with your own neptune cridentials

step 3

Prepare subnetwork yml config e.g single/lstm_2.yml:

    n_h: 32
    n_l: 3

Prepare main network config yml for example:

    network:
        name: MultiHeadTaskRegressor
        config_list: ['single/lstm_2.yml']
        losses: ['L1']
        main_loss: "L1"
        margin_loss: True
        margin_loss_w: 0
        w_losses: [1, 0]
        mode: 'cnn_lstm_att'

step 4

Run

    python main.py -config <PATH TO MAIN CONFIG> -seed 2


For deep LightGBM experiments:

step 1 - Download data

    python cancer_nn/preprocess_data_lgbm.py

Saves preprocessed dataset to data/data_lgbm.csv

step 2 - Fill configs/neptune_config with your own neptune cridentials

step 3

Run:

    python lgbm_main.py



Classical ML methods:
* LightGBM

Deep learning methods:
* FCNN
* CNN 1D
* CNN 1D + Att
* U-shaped CNN 1D + Att + context Att
* LSTM
* LSTM + Att
* LSTM + ATT + CNN 1D + context Att


Results:

Current best solution:

 Subnetwork (CNNAttLSTM): (LSTM + attention with contextual attention where context are features obtained by U-shaped CNN 1D network with attention at the end followed by final Dense Layer)

 Main network (MultiHeadTaskRegressor): Ensembler of 3 base subnetworks, trained with L1 loss functions (with weight 1 for each) and margin_loss with weight equal to 5.

    config_list: ['single/lstm_2.yml', 'single/lstm_2.yml', 'single/lstm_2.yml']
    losses: ['L1', 'L1', 'L1']
    lr: 0.005
    main_loss: L1
    margin_loss: True
    margin_loss_w: 5
    mode: cnn_lstm
    name: MultiHeadTaskRegressor
    seed: 1
    series_len: 20
    w_losses: [1, 1, 1, 1]

where single/lstm_2.yml is:

    n_h: 32
    n_l: 3

Test MAPE (%) | Test MRL | Test MAE | Test Max Error | Test Max Error 99 | correct %
--- | --- | --- | --- |--- | --- |
0.99  | 0.22 | 4.2 | 43 | 15.78 | 95.3

Explainability example:

![Alt text](images/explainability_example.png?raw=true "Explainability example")




Link for repo with experiments with TFT (Temporal Fusion transformer [0]): https://github.com/AWarno/CancerOptimization


Bibliography:

[0] Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). \textit{Temporal fusion transformers for interpretable multi-horizon time series forecasting.} International Journal of Forecasting, 37(4), 1748-1764.
