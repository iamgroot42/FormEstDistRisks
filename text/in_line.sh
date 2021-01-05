#!/bib/bash

for i in {1..2..1}
do
    if [ $3 == 0 ]
    then
        python train_model.py --base_path $1 --data_path /p/adversarialml/as9rw/amazon_test_splits/first --data_path_2 /p/adversarialml/as9rw/amazon_test_splits/second --merge_ratio $2 --model_num $i
    else
        python train_model.py --base_path $1 --data_path /p/adversarialml/as9rw/amazon_test_splits/first --data_path_2 /p/adversarialml/as9rw/amazon_test_splits/second --merge_ratio $2 --model_num $i --not_want_prop True
    fi
done