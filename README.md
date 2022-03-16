# SGEL
Code for WWW 2020 paper: "High Quality Candidate Generation and Sequential Graph Attention Network for Entity Linking"

## Dependencies
This project is based on ```python>=3.6```. The dependent package for this project is listed as below:
```
tensorflow>=1.8.0
scikit-learn==0.21.3
xgboost==0.9
```

## Training Command
1.Extract statistical features
```
python model/local_feature.py
```

2.Calculate xgboost score and filter candidate 
```
python model/xgboost_rank.py
```

3.Get BERT embedding
```
python model/process_bert.py
```
Supplement: Due to historical factors, we train the local model based on the BERT source code. 
Now you can choose to use huggingface to train the BERT local model. 


4.Rank mention
```
python model/local_ranker.py
```

5.Train the global GAT model
```
python model/selector.py
```