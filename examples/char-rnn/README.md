# char-rnn

Example showcases capabilities of current implementation  of RNN/LSTM layers. The goal is to train very simple character level language model. We train on polish epic poem titled "Pan Tadeusz".

Below are some samples outputted by plain rnn. The resulted text looks kind of okay, the most of the outputed strings looks like a words, but at the same time the most of them are incomprehensible (well, the training data wasn't the easiest one) 

Sample 0

    Pietram cządu.
    Bionąc wale, bo padnąp w strzeluli, pannych kowelice wzdupnące, oby ot marszerzeniem.
    Mieszno panków środze s toczą siędwie ażorce dtym, kcillem.
    Pole.
    Chwoły stani sąłta skrzy czy po.

Sample 1

    Godzu, padząc szanoną, zalcoł pomcaty, ubrosza;
    Welcz przydajie, sto, z niez strwał niedojet wypulinościała, alabnął że oskan sorzą siędo we świadka.
    Trudapny wojski po krzął stani sąło, toprz41; -- 
