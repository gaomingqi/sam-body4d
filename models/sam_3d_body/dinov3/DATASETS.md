# Dataset setups for DINOv3

## Evaluations

### Depth Estimation on NYU

Create a folder to host the [NYU dataset](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) for example:

```
export DEPTH_DATASETS_ROOT=${HOME}/datasets
mkdir -p ${DEPTH_DATASETS_ROOT}/NYU
```

We use the NYU subset extracted by [BTS](https://github.com/cleinc/bts/blob/master/tensorflow/README.md) from the 120k samples of the original NYU raw dataset.

#### Option 1 -- Follow BTS's instructions
Please follow BTS instructions to create the dataset:
- [train set](https://github.com/cleinc/bts/blob/master/tensorflow/README.md#nyu-depvh-v2)
- [test set](https://github.com/cleinc/bts/blob/master/README.md#prepare-nyu-depth-v2-test-set).

Make sure you also download the train and test splits:
```
wget https://github.com/cleinc/bts/blob/master/train_test_inputs/nyudepthv2_train_files_with_gt.txt -O ${DEPTH_DATASETS_ROOT}/NYU/nyu_train.txt
wget https://github.com/cleinc/bts/blob/master/train_test_inputs/nyudepthv2_test_files_with_gt.txt -O ${DEPTH_DATASETS_ROOT}/NYU/nyu_test.txt
```

#### Option 2 (preferred) -- Download the readily availble dataset from BinsFormer
Alternatively, one can download the dataset from the following Google Drive [link](https://drive.google.com/file/d/1xI9ksHzCC_kUz6Z4FL_b1ttgj3RVHGwW/view?usp=sharing). If the Google Drive link is not available anymore, try Option 1.

Expected contents:
- `$DEPTH_DATASETS_ROOT/NYU/basement/[...]`
- `$DEPTH_DATASETS_ROOT/NYU/basement_0001a/[...]`
- `$DEPTH_DATASETS_ROOT/NYU/basement_0001b/[...]`
- `$DEPTH_DATASETS_ROOT/NYU/bathroom/[...]`
- `$DEPTH_DATASETS_ROOT/NYU/[...]`
- `$DEPTH_DATASETS_ROOT/NYU/study_room_0004/[...]`
- `$DEPTH_DATASETS_ROOT/NYU/study_room_0005a/[...]`
- `$DEPTH_DATASETS_ROOT/NYU/study_room_0005b/[...]`
- `$DEPTH_DATASETS_ROOT/NYU/nyu_test.txt`
- `$DEPTH_DATASETS_ROOT/NYU/nyu_train.txt`

Note: if data is downloaded with Option 2 make sure to rename `nyu` into `NYU`.

