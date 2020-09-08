# Machie Learning Practice by Python

## version

```
Python version: 3.7.6 (default, Feb  8 2020, 03:44:04) 
[Clang 11.0.0 (clang-1100.0.33.17)]
pandas version: 1.0.5
marplotlib version: 3.2.2
numpy version: 1.19.0
scipy version: 1.5.1
IPython version: 7.16.1
sklearn version: 0.23.1
mglearn version: 0.1.9
```

# 使用しているライブラリ

```python
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

# matplotlibの表示方法
%matplotlib inline
```

## インストール

```
pip install numpy scipy matplotlib ipython scikit-learn pandas pillow mglearn
```

### Jupytarもインストールする場合

```
pip install jupyter
```

# Tip
## 仮想環境の作成

```bash
# 特定のプロジェクト内で
% python3 -m venv [仮想環境名(venvにしとくといいよ)] # use: $ python3 -m venv venv
% source venv/bin/activate
# これでpipをするとvenv内の実行環境にライブラリがインストールされるから環境が汚染されない!

# 仮想環境から抜けるには
% deactivate
```

## 7章のデータをダウンロード＆解凍

```
-> % cd 7
-> % curl -o data/data.tar.gz --create-dirs http://ai.stanford.edu/\~amaas/data/sentiment/aclImdb_v1.tar.gz
-> % tar xzf data/data.tar.gz -C data
```

GNUのtarには-Cオプションという「オプションで指定したディレクトリに移動してからコマンドを開始する」というものがある