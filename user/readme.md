Here is the code to reproduce the results of user representation

#### How To Run 
run data_preprocess/LASTFM.py  to produce train ans test set for LASTFM

run data_preprocess/Recsys19_user.py  to produce train ans test set for recsys19(user)

parameter **user** == 1 represents the embedded model;
parameter **user** == 0 represents the embedded model

run implicit_and_embedded/main_user.py  to  reproduce the result of implicit or embedded

run run src/train_hier_gru.py to reproduce the result of recurrent
