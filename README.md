# Deep Learning for Sequential Recommendation: Algorithms, InfluentialFactors, and Evaluations
Here is the code to reproduce the experiment result in our paper "Deep Learning for Sequential Recommendation: Algorithms, InfluentialFactors, and Evaluations"

- **GRU4Rec** is the code to reproduce the result of basic GRU4Rec, sample size, sample alpha, loss function and data augmentation. 
- **GRU4Rec-with-dwell-time** is the code to reproduce the result of Dwell time.
- **GRU4Rec-with-knn** is the code to reproduce the result of KNN
- **category-and-behavior** is the code to reproduce the result of P-GRU, C-GRU and B-GRU, GRU4Rec(behavior) and GRU4Rec(category)
- **attention-mechanism** is the code to reproduce the result of Attention mechanism
- **user** is the code to reproduce the result of user representation(implicit, embedded and recurrent)



# Acknowledgement
- **GRU4Rec** is the original [Theano implementation](https://github.com/hidasib/GRU4Rec) GRU4Rec.
- **GRU4Rec-with-dwell-time** is modified based on the original [Theano implementation](https://github.com/hidasib/GRU4Rec) GRU4Rec.
- **GRU4Rec-with-knn** is the original code for paper "When Recurrent Neural Networks meet the Neighborhood for
Session-Based Recommendation" (http://bit.ly/2nfNldD)
- **category-and-behavior** is modified based on the [TensorFlow implementation](https://github.com/Songweiping/GRU4Rec_TensorFlow) GRU4Rec.
- **attention-mechanism** is modified based on the [Pytorch implementation](https://github.com/Wang-Shuo/Neural-Attentive-Session-Based-Recommendation-PyTorch) NARM 
- **user** implicit and embedded are modified based on the [TensorFlow implementation](https://github.com/Songweiping/GRU4Rec_TensorFlow) GRU4Rec.
recurrent is the original code for HRNN (https://github.com/mquad/hgru4rec)
