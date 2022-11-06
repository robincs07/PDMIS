# Homwork 2

## How to run the program
```
# Data collection dataset_other
python data_generator.py --dataset_folder replica-dataset/ --output dataset_other/

# Data collection apartment0
python data_generator.py --dataset_folder replica-dataset/ --output dataset_apartment0/

# generate .odgt for dataset_other
python generate_odgt.py --train_folder dataset_other --val_folder dataset_apartment0

# generate .odgt for dataset_apartment0
python generate_odgt.py --train_folder dataset_apartment0 --val_folder dataset_apartment0

# Training for other
cd semantic-segmentation-pytorch
python train.py --gpus 1 --cfg config/other.yaml 
cd ..
# Training for apartment0
cd semantic-segmentation-pytorch
python train.py --gpus 1 --cfg config/model_apartment0.yaml 
cd ..

# evaluate for model_other
cd semantic-segmentation-pytorch
python eval_multipro.py --gpu 0 --cfg config/model_other.yaml
cd ..

# evaluate for model_apartment0
cd semantic-segmentation-pytorch
python eval_multipro.py --gpu 0 --cfg config/model_aparment0.yaml
cd ..

# output the HW1 rgb to semantic for model_other
cd semantic-segmentation-pytorch
python test.py  --imgs ../HW1_Datacollect/first_floor/rgb --cfg config/model_other.yaml --gpu 0
cd ..

# output the HW1 rgb to semantic for model_apartment0
cd semantic-segmentation-pytorch
python test.py  --imgs ../HW1_Datacollect/first_floor/rgb --cfg config/model_apartment0.yaml --gpu 0
cd ..

# reconstruct of model_other semantic map for first floor
python 3d_semantic_map.py 1 163 pred_semantic_other first_floor

# reconstruct of model_apartment0 semantic map first floor
python 3d_semantic_map.py 1 163 pred_semantic_apartment0 first_floor

# reconstruct of model_other semantic map for second floor
python 3d_semantic_map.py 1 163 pred_semantic_other second_floor

# reconstruct of model_apartment0 semantic map first floor
python 3d_semantic_map.py 1 163 pred_semantic_apartment0 second_floor
```
