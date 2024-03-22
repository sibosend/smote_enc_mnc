# smote_enc_mnc

This repository is a modulerized SMOTE-ENC and SMOTE-MNC algorithm based on https://github.com/Mimimkh/SMOTE-ENC-code.

# how to use
1. pip install
    ```
    pip install git+https://github.com/sibosend/smote_enc_mnc
    ```
2. calculate index of categorical features
    ```
    categorical_idx = [X.columns.get_loc(c) for c in categorical_features if c in X]
    ```
1. smoteenc
   ```
    from imblearn.under_sampling import RandomUnderSampler

    from smote_enc_mnc.sample.smoke_enc import  SMOTEENC

    #用SMOTE提升少数类别样本数目，使之达到多数样本数目的30%
    over = SMOTEENC(sampling_strategy=0.3, categorical_features=categorical_idx, n_jobs=-1)
    #用RandomUnderSampler降低多数类别样本数目，使之比少数样本数目多40%
    under = RandomUnderSampler(sampling_strategy=0.4)
    #用pineline进行整合
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    # 转换
    X_resample_enc, y_resample_enc= pipeline.fit_resample(X, y)
   ``` 
2. smotemnc
   ```
    from smote_enc_mnc.sample.smoke_mnc import  SMOTEMNC

    over = SMOTEMNC(sampling_strategy=0.3, categorical_features=categorical_idx, n_jobs=-1)

    under = RandomUnderSampler(sampling_strategy=0.4)
    #用pineline进行整合
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    # 转换
    X_resample_mnc, y_resample_mnc= pipeline.fit_resample(X, y)
   ```
3. SMOTENC
    ```
      from imblearn.over_sampling import SMOTE, SMOTENC
      
      over = SMOTENC(sampling_strategy=0.3, categorical_features=categorical_idx, n_jobs=-1)
      
      under = RandomUnderSampler(sampling_strategy=0.4)
      #用pineline进行整合
      steps = [('o', over), ('u', under)]
      pipeline = Pipeline(steps=steps)
      # 转换
      X_resample_nc, y_resample_nc = pipeline.fit_resample(X, y)
    ```
4. evaluate
    ```
      from sklearn.tree import DecisionTreeClassifier
      # 定义采样评估方法
      def evaluate_smote(data_X, data_y):
          model_dt = DecisionTreeClassifier(random_state = 42)
          cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
          scoring = ['neg_log_loss', 'roc_auc']
          scores = cross_validate(model_dt, data_X, data_y,scoring=scoring, cv=cv, n_jobs=-1)
          print('DecisionTreeClassifier: Mean LOGLOSS: %.3f,  ROC AUC: %.3f'%(mean(scores['test_neg_log_loss'])*-1, mean(scores['test_roc_auc'])))

          for cols in  data_X.select_dtypes(include = ['object']).columns:
              data_X[cols] = data_X[cols].astype(float)
          params = {
              "learning_rate": 0.01,
              "max_depth": 6,
              "min_data_in_leaf": 200,
              "num_leaves": 20,
              "application": 'binary',
              "num_boost_round": 100,
              "verbose": -1

          }
          train_data=lgb.Dataset(data_X, label=data_y)
          cv_results = lgb.cv(params,
                  train_data,
                  seed=42,
                  nfold=10,
                  metrics=['auc', 'binary_logloss'],
                  early_stopping_rounds=10,
                  eval_train_metric=True,
                  verbose_eval=-1
          )

        
    ```
5. DataSet
   > LendingClub's Loan data in 2019 from [kaggle](https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1)
   > [Paipaidai](https://www.heywhale.com/mw/dataset/5eb6ace3366f4d002d77e823/file)
6. Results
   <aside>
   <table>
    <tr>
        <th></th>
        <th colspan=2>DecisionTreeClassifier</th>
        <th colspan=2>LightGBM</th>
    </tr>
    <tr>
        <th></th>
        <td>Logloss</td>
        <td>AUC</td>
        <td>Logloss</td>
        <td>AUC</td>
    </tr>
    <tr>
        <td>LendingClub</td>
        <td>1.907</td>
        <td>0.720</td>
        <td>0.134</td>
        <td>0.890</td>
    </tr>
    <tr>
        <td>LendingClub_SMOTENC</td>
        <td>2.178</td>
        <td>0.881</td>
        <td>0.439</td>
        <td>0.917</td>
    </tr>
    <tr>
        <td>LendingClub_SMOTEENC</td>
        <td>2.115</td>
        <td>0.908</td>
        <td>0.445</td>
        <td>0.911</td>
    </tr>
    <tr>
        <td>LendingClub_SMOTEMNC</td>
        <td>1.947</td>
        <td>0.909</td>
        <td>0.441</td>
        <td>0.948</td>
    </tr>
    <tr>
        <td>Paipaidai</td>
        <td>4.860</td>
        <td>0.529</td>
        <td>0.252</td>
        <td>0.696</td>
    </tr>
    <tr>
        <td>Paipaidai_SMOTENC</td>
        <td>5.770</td>
        <td>0.802</td>
        <td>0.494</td>
        <td>0.865</td>
    </tr>
    <tr>
        <td>Paipaidai_SMOTEENC</td>
        <td>5.764</td>
        <td>0.804</td>
        <td>0.498</td>
        <td>0.873</td>
    </tr>
    <tr>
        <td>Paipaidai_SMOTEMNC</td>
        <td>5.659</td>
        <td>0.806</td>
        <td>0.488</td>
        <td>0.879</td>
    </tr>
</table>
   
   </aside> 