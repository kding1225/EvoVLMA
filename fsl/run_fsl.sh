# RN50, ViT-B16
method=$1
net=$2
ntrials=$3
dtype=$4
seed=$5
save_dir=${net}_caches
cfg_dir=configs/${net}/few_shots/${method}

export CUDA_VISIBLE_DEVICES=$6

for shots in 1 2 4 8 16
do
  for data in caltech101 dtd eurosat fgvc food101 oxford_flowers oxford_pets stanford_cars sun397 ucf101 imagenet
  do
     mkdir -p ${save_dir}/${data}
     save_path=${save_dir}/${data}/fsl_${method}_shots${shots}_seed${seed}.txt
     if [ -f "$save_path" ]; then
        echo "$save_path exists!!!"
        continue
     else
        echo "process $save_path"
     fi
     
     python main_fsl_${method}_optuna.py \
       --config ${cfg_dir}/${data}.yaml \
       --save_dir ${save_dir} --shots ${shots} --seed ${seed} \
       --ntrials ${ntrials} --dtype ${dtype} > ${save_path}
  done
done
