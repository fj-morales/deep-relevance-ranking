# Deep Relevance Ranking Using Enhanced Document-Query Interactions

This is a modified fork of the [[deep-relevance-ranking original](https://github.com/nlpaueb/deep-relevance-ranking)] repository for reproducibility purposes.

## Replication of POSIT-DRMM+MV effectiveness results

To replicate he effectiveness results for POSIT-DRMM+MV, please, follow the [[installation](https://github.com/nlpaueb/deep-relevance-ranking))] and [[running](https://github.com/nlpaueb/deep-relevance-ranking/tree/master/models/drmm)] instructions of the original repository.

## Reproduction of BM25+extra model


**Step 1**: Install the required Python packages and extra requirements: 

`conda env create -f environment.yml`
`sudo apt-get install oracle-java11-installer-local`

**Step 2**: Preprocess corpus and building indexes

  `dataset=bioasq; python ir_indexing.py --dataset $dataset --preprocess --pool_size 10` 

  `dataset=robust; python ir_indexing.py --dataset $dataset`

**Step 3**: Queries and qrels preprocessing

  `dataset=bioasq; python ir_query_preprocessing.py --dataset $dataset --data_split all --pool_size 20 --fold all`
  `dataset=robust; python ir_query_preprocessing.py --dataset $dataset --data_split all --pool_size 20 --fold all`

**Step 4**: Extract features

  `dataset=bioasq; python ir_gen_features.py --dataset $dataset --data_split all --fold all`
  `dataset=robust; python ir_gen_features.py --dataset $dataset --data_split all --fold all`

**Step 5**: BM25+extra and vanilla bm25 features model evaluation

  `dataset=bioasq; python ir_bm25_extra_features.py --dataset $dataset --data_split all --fold all`
  `dataset=robust; python ir_bm25_extra_features.py --dataset $dataset --data_split all --fold all`

**Step 6**: LambdaMART default (n_tree = 1000; learning_rate = 0.1; n_leaves = 10; early_Stop = 100)

  `dataset=bioasq; python3 ir_hpo.py --dataset $dataset --default_config`
  `dataset=robust; python3 ir_hpo.py --dataset $dataset --default_config`

**Step 7**: LambdaMART HPO: RS and BOHB

  `dataset=bioasq; hpo=rs; python3 ir_hpo.py --dataset $dataset --hpo_method $hpo --min_budget 100 --max_budget 100 --n_iterations 200 --n_workers 1`
  `dataset=bioasq; hpo=bohb; python3 ir_hpo.py --dataset $dataset --hpo_method $hpo --min_budget 30 --max_budget 100 --n_iterations 200 --n_workers 1`

  `dataset=robust; hpo=rs; python3 ir_hpo.py --dataset $dataset --hpo_method $hpo --min_budget 100 --max_budget 100 --n_iterations 200 --n_workers 1`
  `dataset=robust; hpo=bohb; python3 ir_hpo.py --dataset $dataset --hpo_method $hpo --min_budget 30 --max_budget 100 --n_iterations 200 --n_workers 1`

**Step 8** LambdaMART test best found model:
 
  `dataset=bioasq; python3 ir_hpo.py --dataset $dataset --test --leaf 15 --tree 1400 --lr 0.07`
  `dataset=robust; python3 ir_hpo.py --dataset $dataset --test --leaf 25 --tree 450 --lr 0.03`
