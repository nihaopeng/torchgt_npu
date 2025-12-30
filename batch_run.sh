# 第一批
nohup bash run.sh origin 0 cora True 5 './exps/cora_max_dist_5' 8080 > cora_max_dist_5.log 2>&1 &
nohup bash run.sh origin 1 cora True 10 './exps/cora_max_dist_10' 8081 > cora_max_dist_10.log 2>&1 &
nohup bash run.sh origin 2 citeseer True 5 './exps/citeseer_max_dist_5' 8082 > citeseer_max_dist_5.log 2>&1 &
nohup bash run.sh origin 3 citeseer True 10 './exps/citeseer_max_dist_10' 8083 > citeseer_max_dist_10.log 2>&1 &