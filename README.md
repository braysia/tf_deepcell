# tf_deepcell

```
python train.py -i data/nuc0.png -l data/labels0.tif -m data/tests_model.py -o output -n 100 -e 2
python predict.py -i data/nuc1.png -w output/cnn_model_weights.hdf5 -m data/tests_model.py -o output
```

```
python predict.py -i data/nuc1.png -w data/tests_pretrained.hdf5 -m data/tests_model.py -o output
```

Use tensorflow (1.3.0)