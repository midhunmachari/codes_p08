# codes_p08
Link: https://github.com/midhunmachari/codes_p08.git


Q01: Trained on ERA --> MSWX
Q02: Generate ESM downscaled data from Q01 models
Q03: Complete unfreeze and retrain of Q01 models
Q04: N/A
Q05: FIne tuning with ConstraintLossAuto on additional layers
Q06: Same as Q05, but the batch normalisation layers removed from the additional layers
Q07: Fine tune with 7 unfreeze layers and weighted_mae loss