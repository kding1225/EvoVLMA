export CUDA_VISIBLE_DEVICES=$9

# RN50, ViT-B16
method=$1
net=$2
srcdata=$3
inst=$4
sample_class=$5
ntrials=$6
dtype=$7
popsize=10
npop=10
seed=$8
ntest_per_class=10
save_dir=${net}_caches
cfg_dir=configs/${net}/few_shots/${method}
feat_func_path=RN50_evol/${srcdata}_inst${inst}_nt${ntest_per_class}/${method}_logit+fs-s2/ael_popsize${popsize}_npop${npop}_sc${sample_class}/results/pops_best/population_generation_${npop}.json
logit_func_path=RN50_evol/${srcdata}_inst${inst}_nt${ntest_per_class}/${method}_logit+fs-s1/ael_popsize${popsize}_npop${npop}_sc${sample_class}/results/pops_best/population_generation_${npop}.json
save_prefix=fsl_${srcdata}_inst${inst}_nt${ntest_per_class}_${method}_logit+fs_ael_popsize${popsize}_npop${npop}_sc${sample_class}

for shots in 1 2 4 8 16
do
  for data in caltech101 dtd eurosat fgvc food101 oxford_flowers oxford_pets stanford_cars sun397 ucf101 imagenet
  do
     mkdir -p ${save_dir}/${data}
     save_path=${save_dir}/${data}/${save_prefix}_shots${shots}_seed${seed}.txt
     if [ -f "$save_path" ]; then
        echo "$save_path exists!!!"
        continue
     else
        echo "process $save_path"
     fi
     
     python main_fsl_elm_optuna.py \
       --config ${cfg_dir}/${data}.yaml \
       --save_dir ${save_dir} --shots ${shots} --method ${method} \
       --feat_func ${feat_func_path} --logit_func ${logit_func_path} \
       --seed ${seed} --ntrials ${ntrials} --dtype ${dtype} > ${save_path}
  done
done
