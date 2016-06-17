The models are trained on OP2-4 and evaluated on OP1.

To train run

```bash
$ python train.py
```


## Results

| Model | Name | Hypes  | Max F1 | Average Precision | BestThresh   | Accuracy @ max | Thresh max. Acc.  | Accuracy @ 0.5 | Accuracy @ 0.25 | 
| ----- | ---------- | ---------------------|--------|---------| ------| ----------- | -----------  | -----------| ------------|
| 1     | FCN32_VGG  |  LR: 1e-6            | 83.86% |  86.72% | 0.28  | 96.73 %     | 0.38         |  96.65%    | 96.59 %     |
| 2     | FCN32_VGG  |  LR: 1e-5            | 85.90% |  88.51% | 0.32  | 97.13 %     | 0.42         |  97.11%    | 97.01 %     |
| 3     | FCN32_VGG  |  LR: 1e-5   head: 2  |        |         |       |             |              |            |             |

