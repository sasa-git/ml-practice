# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

# データセット生成
X, y = mglearn.datasets.make_forge()
# プロット
mglearn.discrete_scatter(X[:,0], X[:,1], y)
# loc... 1,2,3,4でラベル説明の位置
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
print("X.shape: {}".format(X.shape))


# %%
X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel("Feature")
plt.ylabel("Target")


# %%
# 乳がんの腫瘍が悪性か予測してみる

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("Cancer keys(): \n{}".format(cancer.keys()))


# %%
print("Shape of cancer data: {}".format(cancer.data.shape))


# %%
# zip(): https://note.nkmk.me/python-zip-usage-for/
# n: v にそれぞれ zip() で合成した配列を代入していく
# np.bincount(): 0(良性), 1(悪性)が何個あるかを集計 [0,1]の順
print("Sample counts per class:\n{}".format(
    {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}
))


# %%
print("Feature names:\n{}".format(cancer.feature_names))


# %%
print(cancer.DESCR)


# %%
# ボストンの住宅地の住宅価格の中央値を推測する
from sklearn.datasets import load_boston
boston = load_boston()
print("Data shape: {}".format(boston.data.shape))


# %%
# 特徴量間の積を拡張したデータセット
X, y = mglearn.datasets.load_extended_boston()
print("X.shape: {}".format(X.shape))
# 13 + 91(13から2つを選ぶ重複あり組み合わせ: 13+2 -1C2| 14!/(12!2!))


# %%
# 用意した二つのデータセットを用いて機械学習アルゴリズムの特徴を見ていく

# k-NN
mglearn.plots.plot_knn_classification(n_neighbors=1)


# %%
mglearn.plots.plot_knn_classification(n_neighbors=3)


# %%
from sklearn.model_selection import train_test_split

X, y = mglearn.datasets.make_forge()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# %%
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)


# %%
clf.fit(X_train, y_train)


# %%
print("Test set predictions: {}".format(clf.predict(X_test)))


# %%
print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))


# %%
# 決定境界を見てみる
fig, axes = plt.subplots(1, 3, figsize=(10, 3))

for n_neighbors, ax in zip([1, 3, 9], axes):
    # fitメソッドは自身を返してるから1行でインスタンス生成してfitできる
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{} neighbor(s)".format(n_neighbors))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
axes[0].legend(loc=3)


# %%
from sklearn.datasets import load_breast_cancer

# 各n点で分類する時の訓練セットに対する性能とテストセットに対する性能をみる

cancer = load_breast_cancer()
# stratufy= https://note.nkmk.me/python-sklearn-train-test-split/
# 訓練データとテストデータでのデータの比率を揃える
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=66)

training_accuracy = []
test_accuracy = []
# 1~10まで試す
neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    # 訓練セット精度
    training_accuracy.append(clf.score(X_train, y_train))
    # 汎化精度
    test_accuracy.append(clf.score(X_test, y_test))

plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.xlabel("n_neighbors")
plt.ylabel("Accuracy")
plt.legend()

# %% [markdown]
# 上の例では、最近傍点が小さいほど複雑なモデルとなっている

# %%
# k-近傍回帰

mglearn.plots.plot_knn_regression(n_neighbors=1)
# 緑色がテストデータ、予測結果が青星


# %%
mglearn.plots.plot_knn_regression(n_neighbors=3)
# 最近傍点の平均をとっている


# %%
from sklearn.neighbors import KNeighborsRegressor

X, y = mglearn.datasets.make_wave(n_samples=40)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

reg = KNeighborsRegressor(n_neighbors=3)
reg.fit(X_train, y_train)


# %%
print("Test set predictions:\n{}".format(reg.predict(X_test)))


# %%
# モデルの評価 R^2スコアを返している。決定係数と呼ばれ、0~1までの値。
print("Test score R^2: {:.2f}".format(reg.score(X_test, y_test)))


# %%
# 1つのデータセットに対して、全ての値に対する予測値
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# -3から3までの間に1000点のデータポイントを作成
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
for n_neighbors, ax in zip([1, 3, 9], axes):
    # 1, 3, 9近傍点で予測
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train, y_train)
    ax.plot(line, reg.predict(line))
    ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
    ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=9)

    ax.set_title("{} neighbor(s)\n train score: {:.2f} test score: {:.2f}".format(n_neighbors, reg.score(X_train, y_train), reg.score(X_test, y_test)))
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")

# plotしたデータごとに名前をつけてる
axes[0].legend(["Model predictions", "Training data/target", "Test data/target"], loc="best")

# %% [markdown]
# ```python
# >>> import numpy as np
# >>> line = np.linspace(-3, 3, 10)
# >>> print(line)
# [-3.         -2.33333333 -1.66666667 -1.         -0.33333333  0.33333333
#   1.          1.66666667  2.33333333  3.        ]
# >>> line.reshape(-1, 1)
# array([[-3.        ],
#        [-2.33333333],
#        [-1.66666667],
#        [-1.        ],
#        [-0.33333333],
#        [ 0.33333333],
#        [ 1.        ],
#        [ 1.66666667],
#        [ 2.33333333],
#        [ 3.        ]])
# >>> 
# ```

# %%
# 線形モデル
mglearn.plots.plot_linear_regression_wave()


# %%
# 線形回帰(通常最小二乗法)

from sklearn.linear_model import LinearRegression

X, y = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

lr = LinearRegression().fit(X_train, y_train)

# 傾きを示すパラメータは重み・係数(coefficient)と言われ、coef_ 属性に格納される
# 切片(intercept)は intercept_ に


# %%
print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))


# %%
print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))

# スコアはよくはないが、訓練セットとテストセットの値が近い。適合不足であって、過剰適合ではない
# 特徴量が少ない=モデルが単純であれば過剰適合の危険は少ないが、特徴量が多いデータセットでは線形モデルは強力になり過剰適合になりやすい


# %%
X, y = mglearn.datasets.load_extended_boston()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train, y_train)


# %%
print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))

# 差が大きい=過剰適合が起きてる兆候。複雑度を制御できるモデルを探さなくてはならない。
# 線形回帰に代る一般的な手法は、リッジ回帰


# %%
# リッジ回帰は、個々の特徴量の重みをなるべく0にする。→ここの特徴量が出力に与える影響を小さくしたい
# 過剰適合を防ぐために明示的にモデルを制約する(正則化 regularization)

from sklearn.linear_model import Ridge

ridge = Ridge().fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))


# %%
# Rdgeモデルでは制約の強いモデルだから過剰適合の危険は少ない
# モデルの簡潔さ(0に近い係数の数)と訓練セットに対する性能はトレードオフ
# alphaパラメータを使ってどちらに重きをおくか指定できる
# alphaを増やすと係数wはより0に近くなる→訓練セットに対する性能↓、汎化性能↑？

ridge10 = Ridge(alpha=10).fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge10.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge10.score(X_test, y_test)))


# %%
# alphaを小さくすると係数の制約は小さくなる→モデルの複雑度があがる。めっさ小さくなると制約はほとんどなくなり、LinearRegressionと同じ挙動となる
# 今回は小さくしてみると・・・？

ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge01.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge01.score(X_test, y_test)))
# パラメータのチューニングについて詳しくは5章へ


# %%
# alphaパラメータのモデルへの影響をみる

plt.plot(ridge.coef_, 's', label="Ridge alpha=1")
plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
plt.plot(ridge01.coef_, 'v', label="Ridge alpha=0.1")

plt.plot(lr.coef_, 'o', label="LinearRegression")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
# 0,0の位置に係数分の長さ(特徴量の数)の水平線を引く
plt.hlines(0, 0, len(lr.coef_))
plt.ylim(-25, 25)
plt.legend()


# %%
print("Coefficient length: {}".format(len(lr.coef_)))
print("Linear Regression coefficients:\n{}".format(lr.coef_))


# %%
# 学習曲線の表示 訓練データ数ごとのスコアの変化
# 線形回帰とリッジ回帰(alpha=1)のモデルの比較

mglearn.plots.plot_ridge_n_samples()
# 十分な訓練データがある場合はリッジ回帰と線形回帰は同じ性能を示す。→正則化はあまり重要ではなくなってくる
# 訓練データが増えると、線形回帰は訓練性能が低下している。→過剰適合が難しくなる


# %%
# Lasso回帰
# 制約の掛け方が違う(L1正則化)。いくつかの係数が完全に0になっている。→いくつかの特徴量を無視し、自動的に特徴量を選択してモデルを解釈しやすくなる。

from sklearn.linear_model import Lasso

lasso = Lasso().fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso.coef_ != 0)))


# %%
print("Coefficients:\n{}".format(lasso.coef_))


# %%
# "max_iter(最大の繰り返し回数)"の値を増やす
# こうしないとモデルが"max_iter"を増やせと警告がくる

lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso001.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso001.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso001.coef_ != 0)))


# %%
# alphaを小さくするとリッジ同様に正則化の効果が薄れ、過剰適合が発生する

lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso00001.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso00001.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso00001.coef_ != 0)))


# %%
# alphaパラメータのモデルへの影響をみる

plt.plot(lasso.coef_, 's', label="Lasso alpha=1")
plt.plot(lasso001.coef_, '^', label="Lasso alpha=0.01")
plt.plot(lasso00001.coef_, 'v', label="Lasso alpha=0.0001")

plt.plot(ridge01.coef_, 'o', label="Ridge alpha=0.1")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.ylim(-25, 25)
# データの説明を2*2で、場所を0,1.05の位置に表示
plt.legend(ncol=2, loc=(0, 1.05))

# %% [markdown]
# 実際にはまずリッジ回帰を試してみると良い
# 
# 特徴量がたくさんあって重要なものがわずかしかなさそうならLasso回帰が向いている
# 
# また、解釈しやすいモデルが欲しい時も重要な特徴量を選んでくれるLassoが向いている
# 
# RidgeとLassoのペナルティを組み合わせたElasticNetがscikit-learnにはある
# 
# 実用上これが最良の結果をもたらすが、L1正則化とL2正則化のパラメータの2つを調整するコストがかかる
# %% [markdown]
# # クラス分類のための線形モデル
# 
# 分類したいデータが関数より大きいか小さいかで分類する。決定境界が入力の線形関数となっている
# 線形モデルのアルゴリズムは2点で区別される
# 
# - 係数と切片の特定の組み合わせと訓練データの適合度を図る尺度
# - 正規化を行うか、行うならどの方法か
# 
# 基本的には1番目のアイテム(ロス関数)はあまり意味がない。
# **ロジスティクス回帰(logistic regression)**と**線形サポートベクタマシン(linear support vector machines: SVM)**は一般的な線形クラス分類アルゴリズム

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

X, y = mglearn.datasets.make_forge()

fig, axes = plt.subplots(1, 2, figsize=(10, 3))

for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
    clf = model.fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5, ax=ax, alpha=.7)
    mglearn.discrete_scatter(X[:,0], X[:,1], y, ax=ax)
    ax.set_title("{}".format(clf.__class__.__name__))
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")

axes[0].legend()

# %% [markdown]
# これらのモデルは、リッジ回帰のようにL2正則化を行う
# 
# 正則化の強度を決定するパラメータは**C**と呼ばれる。
# 
# Cが大きくなると正則化は弱くなる→訓練データに対して適合度をあげようとする
# 
# Cが小さくなると係数ベクトル(w)を0に近づけることを重視する
# 
# alphaとは性質が逆？
# 
# Cを小さくするとデータポイントの「大多数に」対し適合しようとするが、Cを大きくするとここのデータポイントを正確にクラス分類することを重視するようになる

# %%
mglearn.plots.plot_linear_svc_regularization()
# 右のグラフ(C:大)は全ての点を正しく分類しようとして、過剰適合している
# 回帰の場合と同様に、線形モデルによるクラス分類は低次元空間においては制約が強く見える(境界が直線や平面にしかならないから)
# 高次元の場合には線形モデルによるクラス分類は非常に強力になっていく→特徴量が多い場合、過剰適合を回避する方法が重要になる


# %%
# LogisticRegressionをcancerデータセットを使って解析

 #    #   ##   #####  #    # # #    #  ####  
 #    #  #  #  #    # ##   # # ##   # #    # 
 #    # #    # #    # # #  # # # #  # #      
 # ## # ###### #####  #  # # # #  # # #  ### 
 ##  ## #    # #   #  #   ## # #   ## #    # 
 #    # #    # #    # #    # # #    #  ####  
# バージョン0.22からLogisticRegressionのsolverの引数がliblinearからlbfgsに変更されているため、本の内容と異なった出力・エラーが発生する
# そのため、引数を明示する必要がある
# => https://qiita.com/idontwannawork/items/86c5b833cdc0a4cf58b5
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
logreg = LogisticRegression(solver="liblinear").fit(X_train, y_train)
print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))


# %%
# C=100

logreg100 = LogisticRegression(C=100, solver="liblinear").fit(X_train, y_train)
print("Training set score: {:.3f}".format(logreg100.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg100.score(X_test, y_test)))


# %%
# C=0.01

logreg001 = LogisticRegression(C=0.01, solver="liblinear").fit(X_train, y_train)
print("Training set score: {:.3f}".format(logreg001.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg001.score(X_test, y_test)))


# %%
plt.plot(logreg.coef_.T, 'o', label="C=1")
plt.plot(logreg100.coef_.T, '^', label="C=100")
plt.plot(logreg001.coef_.T, 'v', label="C=0.01")
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.ylim(-5, 5)
plt.xlabel("Feature")
plt.ylabel("Coefficient magnitude")
plt.legend()

# mean perimeterだけ、C=0.01とC=100の係数が反転している→perimeterが大きいことが良性か悪性かを示唆しているか変わってしまう
# 係数の解釈には気をつけよう
# L2正則化は、係数は決して0にはならない
# より解釈しやすいモデルを得るには、L1正則化を使う


# %%
# logregのcoefficientは2次元配列
logreg.coef_


# %%
# これを分けている
logreg.coef_.T


# %%
# ちなみにRidgeモデルとかは1次元配列
ridge.coef_


# %%
for C, maker in zip([0.001, 1, 100], ['o', '^', 'v']):
    lr_l1 = LogisticRegression(C=C, solver="liblinear", penalty="l1").fit(X_train, y_train)
    print("Training accuracy of l1 logreg with C={:.3f}: {:.2f}".format(C, lr_l1.score(X_train, y_train)))
    print("Test accuracy of l1 logreg withC={:.3f}: {:.2f}".format(C, lr_l1.score(X_test, y_test)))
    print("Number of features used: {}\n".format(np.sum(lr_l1.coef_ != 0)))

# penalty パラメータがモデルの正則化と特徴量を全て使うかに影響を与える


# %%
# 多クラス分類: 1対他の分類器をクラス分だけ用意して分類。それぞれに重みwと切片を持っている
# 各クラスを正規分布でサンプリングした2次元データセットを用いる

from sklearn.datasets import make_blobs

X, y = make_blobs(random_state=42)
mglearn.discrete_scatter(X[:,0], X[:,1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(["Class 0", "Class 1", "Class 2"])


# %%
# LinearSVCクラス分類器をデータセットで学習させる

linear_svm = LinearSVC().fit(X, y)
print("Coefficient shape: {}".format(linear_svm.coef_.shape))
print("Intercept shape: {}".format(linear_svm.intercept_.shape))
# coef_の各行には各クラスに対応する係数ベクトル、各列には特徴量が入っている
# interceptには各クラスに対する切片が


# %%
# 分類器の直線の可視化

mglearn.discrete_scatter(X[:,0], X[:,1], y)
# -15 ~ 15まで50(default)コに区切る
line = np.linspace(-15, 15)
for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_, ['b', 'r', 'g']):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
plt.ylim(-10, 15)
plt.xlim(-10, 8)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(["Class 0", "Class 1", "Class 2", "Line Class 0", "Line Class 1", "Line Class 2"],loc=(1.01, 0.3) )

# Class0の点群: クラス0の分類器の直線より上→Class0、クラス1の分類器より上→その他、クラス2の分類器より上→その他 になっている
# クラス分類確信度の式がクラス0分類器では0より大きくなり、クラス1、2分類器では0より小さくなる


# %%
# 上の図では、中央に「どの分類器でその他」となった領域がある。この領域の分類は、「クラス分類式の値が一番大きいクラス」→その点に最も近い線をもつクラスとなる
# 全ての点に対する予測

mglearn.plots.plot_2d_classification(linear_svm, X, fill=True, alpha=.7)
mglearn.discrete_scatter(X[:,0], X[:,1], y)
for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_, ['b', 'r', 'g']):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(["Class 0", "Class 1", "Class 2", "Line Class 0", "Line Class 1", "Line Class 2"],loc=(1.01, 0.3) )

# %% [markdown]
# # 利点・欠点・パラメータ
# 
# ## alpha/C
# 
# 線形モデルの主要なパラメータは回帰モデルでは`alpha`, LinearSVCとLogisticRegressionでは`C`と呼ばれるパラメータ
# 
# alphaが大きい・Cが小さい場合は単純なモデルに対応する。特に回帰モデルの場合、パラメータの調整は非常に重要になる。Cやalphaを調整する場合、対数スケールで値を変える。
# 
# ## L1正則化を使うか、L2正則化を使うか
# 一部の特徴量たけが重要そうならL1を使う。そうでなければ基本L2を使う。
# 
# L1はモデルの解釈が重要な時にも有用。使う特徴量が絞られ、どの特徴量がそのモデルに重要か、どのような効果をもつかを説明しやすい。
# 
# ---
# 
# モデルの学習・予測が高速で、大きいデータセットにも対応できるし、疎なデータ(ほとんどの特徴量が0になるようなデータセット)に対してもうまく機能する。
# 
# 非常に大きいデータセットに対しては、LogisticRegressionとRidgeに`solver='sag`オプションを使うといい。大きなデータセットに対して、デフォルトの場合よりも高速になる時がある。もう一つの方法として、`SGDClassifier`クラスと`SGDRegression`クラスを使う方法がある。
# 
# 線形モデルのもう一つの利点として、予測手法が比較的理解しやすい。しかし、係数がどうしてその値になったかはブラックボックス。特に、強く相関した特徴量があると係数の意味を理解しにくい。
# 
# 線形モデルは、特徴量の数がサンプル数よりも多い時に性能を発揮する。また、大きなデータセットに対して適用されることも多い。
# 低次元空間では、他のモデルの方が良い汎化性能を示すこともある。
# 
# => 詳しくはカーネル法を用いたサポートベクタマシンで
# %% [markdown]
# ## 特徴
# 
# - 特徴量の数がサンプル数よりも多い時に性能を発揮する。
# 
# 
# ## 長所
# 
# - モデルの学習・予測が高速
# - 大きいデータセットに対応できる
# - 疎なデータ(ほとんどの特徴量が0になるようなデータセット)に対してうまく機能する
# - 予測手法が比較的理解しやすい。
# 
# ## 短所
# 
# - 係数がどうしてその値になったかはブラックボックス。特に、強く相関した特徴量があると係数の意味を理解しにくい。
# 
# ## パラメータ
# 
# - alpha/C
# 
#     線形モデルの主要なパラメータは回帰モデルではalpha, LinearSVCとLogisticRegressionではC
# 
#     alphaが大きい・Cが小さい場合は単純なモデルに対応する。特に回帰モデルの場合、パラメータの調整は非常に重要になる。Cやalphaを調整する場合、対数スケールで値を変える。
# 
# - L1、L2正則化
# 
#     一部の特徴量たけが重要そうならL1を使う。そうでなければ基本L2を使う。
# 
#     L1はモデルの解釈が重要な時にも有用。使う特徴量が絞られ、どの特徴量がそのモデルに重要か、どのような効果をもつかを説明しやすい。
# 
# 常に大きいデータセットに対しては、LogisticRegressionとRidgeに`solver='sag'`オプションを使うといい。大きなデータセットに対して、デフォルトの場合よりも高速になる時がある。もう一つの方法として、SGDClassifierクラスとSGDRegressionクラスを使う方法がある。
# 
# 低次元空間では、他のモデルの方が良い汎化性能を示すこともある。
# 
# => 詳しくはカーネル法を用いたサポートベクタマシンで
# %% [markdown]
# # ナイーブベイズクラス分類器
# 
# 線形モデルよりも高速だが、汎化性能がわずかに劣る
# 
# ナイーブベイズの速さは、クラスに対する統計値を個々の特徴量ごとに集めてパラメータを学習するためである。
# 
# ## scikit-learnに実装されているナイーブベイズクラス分類器
# 
# - GaussianNB
# 
#     任意の連続値データに適用
#      
# - BernoulliNB
# 
#     2値データを仮定
# 
# - MultinomialNB
#     
#     カウントデータ: 個々の特徴量が何らかの整数カウントを仮定
# 
# ## BernoulliNB
# 個々のクラスに対して、特徴量ごとに非ゼロである場合をカウントする。

# %%
# BernoulliNB

X = np.array([[0, 1, 0, 1],
              [1, 0, 1, 1],
              [0, 0, 0, 1],
              [1, 0, 1, 0]])

y = np.array([0, 1, 0, 1])


# %%
# 個々のクラスに対して、特徴量ごとに非ゼロである場合をカウント

counts = {}
for label in np.unique(y):
    # クラスごとにループ
    # label=0の時、y == labelでyが0になってるインデックスを取り出して、Xに適用している(Xの0番目と3番目を抽出)
    # axis=0: 列ごとの合計をとる 1: 行ごとの合計
    counts[label] = X[y == label].sum(axis=0)
print("Feature counts:\n{}".format(counts))

# %% [markdown]
# ## MultinomialNB
# 
# クラスごとの、個々の特徴量の平均値を考慮に入れる    
# 
# ## GaussianNB
# 
# 平均と標準偏差も考慮に入れる
# 
# 予測の際には個々のクラスの統計量とデータポイントが比較され、最もよく適合したクラスが採用される。
# 
# MultinomialNBとBernoulliNBでは、線形モデルと同じ形の予測式になるが、`coef_`は`w`と同じではない
# %% [markdown]
# ## 特徴
# - 線形モデルでも時間がかかる大規模データセットに対するベースラインとして有効
# 
# ## 長所
# - 利点は線形モデルと共通
# 
# ## パラメータ
# 
# - MultinomialNBとBernoulliNBにはパラメータが一つだけある。=> `alpha`
#     全ての特徴量に対してデータポイントがalphaの大きさに応じた量だけ追加されたように振舞う。
#     alphaが大きくなるとスムーズになり、モデルの複雑さが減る。
#     アルゴリズムの性能はalphaに対して頑健である。→alphaによって大きく性能が変わらない。
#     alphaを調整することで精度をあげることができる。
# 
# GaussianNBは高次元に対して用いられる。他の二つはテキストのような、疎なカウントデータに対して用いられる。
# 一般にMultinomialNBの方がBernoulliNBよりも若干性能が良いが、多くの非ゼロ特徴量がある場合には、MultinomialNBが有効。

# %%
# 決定木
# `AttributeError: module 'graphviz' has no attribute 'Digraph'`と出た
# `brew install graphviz`
# => https://graphviz.gitlab.io/download/
# これでだめそうだったら`pip install graphviz`
# 最後にIpython kernelを再起動させてもう一度runしてみる

mglearn.plots.plot_animal_tree()
# この場合、4つのクラス(鷹、ペンギン、イルカ、熊)を3つの特徴量(質問項目)で識別する

# %% [markdown]
# # 決定木の構築
# ## 本を見た方がわかりやすい P.71
# 
# ---
# 
# 回帰タスクにも決定木を使うことができる。予測を行うには、テストに基づいてノードを辿り、そのデータポイントが属する葉を求める。
# 出力はその葉の中にある訓練データポイントの平均ターゲット値となる。
# 

# %%
# 過剰適合を防ぐためには木の生成を早めに止める`事前枝刈り`と木を構築してから情報の少ないノードを削除する`事後枝刈り`がある。
# scikit-learnには事前枝刈りしか実装されていない

from sklearn.tree import DecisionTreeClassifier

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))


# %%
# 4つの質問まで制限
tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))


# %%
# 決定木の解析

from sklearn.tree import export_graphviz

export_graphviz(tree, out_file="tree.dot", class_names=["malignant", "benign"],
                feature_names=cancer.feature_names, impurity=False, filled=True)


# %%
import graphviz

with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

# value = [malignant(悪性), benign(良性))]

# %% [markdown]
# 決定木から導出できる、決定木の挙動を要約する特性値を見る。
# 要約に使われるのが、**特徴量の重要度(feature importance)**。
# 決定木が行う判断にとって、個々の特徴量がどの程度重要かを示す割合。
# それぞれの特徴に対し0~1の割合で、0は「全く使われてない」、1は「完全にターゲットを予測できる」を意味する。
# 特徴量の重要度の和は1となる。

# %%
print("Feature importances:\n{}".format(tree.feature_importances_))


# %%
def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1] # (特徴量の個数を抽出)
    plt.barh(range(n_features), model.feature_importances_, align="center") # bar horizontal?
    # np.arrange(3) => [0, 1, 2]
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")

plot_feature_importances_cancer(tree)


# %%
# ある特徴量が高い重要度だからと言って、その値が大きいと良性になるか悪性になるか決まる訳ではない
# あくまでも、「重要度」

tree = mglearn.plots.plot_tree_not_monotone()
display(tree)
# X[1]と出力クラスの関係は単調ではない


# %%
# 決定木による回帰では、注意がある
# 外挿(extrapolate)ができない。→訓練データの外側のレンジには予測ができない
# RAM価格の履歴を使って見てみる

import os
ram_prices = pd.read_csv(os.path.join(mglearn.datasets.DATA_PATH, "ram_price.csv"))

plt.semilogy(ram_prices.date, ram_prices.price)
plt.xlabel("Year")
plt.ylabel("Price in $/Mbyte")


# %%
# 上のデータを使って決定木と線形回帰を比較
# 線形回帰では、価格は対数スケールに直して関係が比較的直線になるようにする。線形回帰では重要な処理になっていく
# 詳しくは4章へ

from sklearn.tree import DecisionTreeRegressor

# 過去のデータを用いて2000年以降の価格を予測する
data_train = ram_prices[ram_prices.date < 2000]
data_test = ram_prices[ram_prices.date >= 2000]

# 日付によって価格を予想 pdの形式からnp配列に変換
X_train = data_train.date[:, np.newaxis]
# データとターゲットの関係を単純にするために対数変換
y_train = np.log(data_train.price)

tree = DecisionTreeRegressor().fit(X_train, y_train)
linear_reg = LinearRegression().fit(X_train, y_train)

#全ての価格を予想
X_all = ram_prices.date[:, np.newaxis]

pred_tree = tree.predict(X_all)
pred_lr = linear_reg.predict(X_all)

price_tree = np.exp(pred_tree)
price_lr = np.exp(pred_lr)

# %% [markdown]
# ```
# X_train = data_train.date
# X_train
# 
# 
# 0      1957.00
# 1      1959.00
# 2      1960.00
# 3      1965.00
# 4      1970.00
#         ...   
# 197    1999.50
# 198    1999.67
# 199    1999.75
# 200    1999.83
# 201    1999.92
# Name: date, Length: 202, dtype: float64
# ```
# 
# ---
# 
# ```
# X_train = data_train.date[:, np.newaxis]
# X_train[:10]
# 
# array([[1957.  ],
#        [1959.  ],
#        [1960.  ],
#        [1965.  ],
#        [1970.  ],
#        [1973.  ],
#        [1974.  ],
#        [1975.  ],
#        [1975.08],
#        [1975.25]])
# ```

# %%
plt.semilogy(data_train.date, data_train.price, label="Training data")
plt.semilogy(data_test.date, data_test.price, label="test data")
plt.semilogy(ram_prices.date, price_tree, label="Tree prediction")
plt.semilogy(ram_prices.date, price_lr, label="Linear prediction")
plt.legend()

# 決定木は訓練データに対して完全な予測を行う
# 訓練データにない領域には新しい答えを生成できない
# しかし、時系列データを予測するのに適していない、という訳ではない。(ポイントはあくまで予測方法の特性)
# 例えば、物の値段が上がるか下がるがなどで将来の予測はできる。

# %% [markdown]
# # 長所、短所、パラメータ
# 
# 決定木におけるモデルの複雑さを制御するパラメータは、事前枝切りパラメータ。
# 過剰適合を防ぐには、`max_depth`, `max_leaf_nodes`, `min_samples_leaf`のどれかを選ぶ。
# 
# 結果のモデルが容易に可視化できる。また、データのスケールに対して完全に不変。
# 個々の特徴量は独立に処理される。特徴量ごとにスケールが大きく異なっていても、2値特徴量と連続値特徴量が混ざっていても機能する。
# 
# 過剰適合しやすく、汎化性能が低く出る傾向がある。ほとんどのアプリケーションでは決定木を単体で使うことなく、次のアンサンブル法が用いられる。
# 
# %% [markdown]
# ## 長所
# - 結果のモデルが容易に可視化できる
# - データのスケールに対して完全に不変。個々の特徴量は独立に処理される。特徴量ごとにスケールが大きく異なっていても、2値特徴量と連続値特徴量が混ざっていても機能する。
# 
# ## 短所
# - 過剰適合しやすく、汎化性能が低く出る傾向がある
# - 予測特性的に訓練データにない領域に対して「新しい」予測(外挿)ができない(決定木に基づく全てのモデルに共通)
# 
# ## パラメータ
# 決定木におけるモデルの複雑さを制御するパラメータは、事前枝切りパラメータ。過剰適合を防ぐ役割がある
# 
# - max_depth
# 
#     木の深さ
# 
# - max_leaf_nodes
# 
#     葉(質問)の最大数
# 
# - min_samples_leaf
# 
#     分割する際にその中に含まれている点の最小数
# 
# 以上のどれかを選ぶ
# 
# ほとんどのアプリケーションでは決定木を単体で使うことなく、次のアンサンブル法が用いられる
# %% [markdown]
# # アンサンブル法(Ensenbles)
# 
# 複数の機械学習モデルを組み合わせることで、より強力なモデルを構築する手法。
# 様々なデータセットに対するクラス分類や回帰に有効なアンサンブル方が二つある
# 
# ## ランダムフォレスト
# 
# 詳しくはP.82
# 

# %%
# ランダムフォレストの解析

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

forest = RandomForestClassifier(n_estimators=5, random_state=2)
forest.fit(X_train, y_train)


# %%
# ランダムフォレストとして構築された決定木は`estimator_`属性に格納されている。
# それぞれの決定木で学習された決定境界と、ランダムフォレストによって行われる集合的な予測を可視化

fi, axes = plt.subplots(2, 3, figsize=(20, 10))
# axes.ravel()で2次元配列から1次元配列に変えている
for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
    ax.set_title("Tree {}".format(i))
    mglearn.plots.plot_tree_partition(X_train, y_train, tree, ax=ax)

mglearn.plots.plot_2d_separator(forest, X_train, fill=True, ax=axes[-1, -1], alpha=.4)
axes[-1, -1].set_title("Random forest")
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)

# 実際にはもっと多くの決定木を使うため、決定境界はさらに滑らかになる。


# %%
# cancerデータセットに対して100個の決定木を用いたランダムフォレストを適用する

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))


# %%
# ランダムフォレストの特徴量の重要度を見てみる。個々の決定木の重要度を平均した物で、多くの場合に、個々の決定木の重要度よりも信頼できる

plot_feature_importances_cancer(forest)
# max_featuresや個々の決定木に対して事前枝刈りをしてチューニングすることもある。

# %% [markdown]
# # 長所、短所、パラメータ P.86
# 
# 現在最も広く使われている機械学習手法。
# ランダムフォレストは決定木の利点を生かしたまま欠点の一部を補っている。
# 決定木を使う理由があるとしたら、決定プロセスの簡潔な説明をしたい時。
# 
# 大きいデータセットに対してランダムフォレストを作るのは時間がかかるが、`n_jobs`パラメータで並列処理できる。
# `n_jobs=-1`とすると全てのコアを使うようになる。
# 
# ランダムフォレストは本質的にランダムであるため、乱数のシード値によって構築されるモデルが大きく変わることがある。
# 決定木の数が増えると乱数シードの影響を受けにくくなる。
# 
# テキストデータなどの、非常に高次元で疎なデータに対してはうまく機能しない傾向にある。
# このようなデータに対しては、線形モデルが適している。
# 大きいデータセットに対して機能するし、並列化できるが、線形モデルよりも多くのメモリを消費し訓練も予測も遅くなる。
# 実行時間やメモリが重要なアプリケーションでは、線形モデルを使った方が良い。
# 
# 重要なパラメータは、`n_estimators`、`max_features`と、`max_depth`などの事前枝切りパラメータである。
# `n_estimators`は大きければ大きい方が良い。より多くの決定木の平均をとると過剰適合が軽減される。
# 
# `max_features`は個々の決定木の乱数性を決定すると共に、`max_features`が小さくなると過剰適合が提言する。
# `max_features`や`max_leaf_nodes`を追加すると性能が上がることがある。また、訓練や予測にかかる時間を大幅に縮めることができる。
# %% [markdown]
# ## 特徴
# - 最も広く使われている機械学習手法
# - 本質的にランダム。`random_state`によって大きくモデルが変わる可能性がある
# 
# ## 長所
# 
# - チューニングせずに使える・データのスケール変換が不要
# - 決定木の上位互換
# 
# ## 短所
# 
# - 大きいデータセットに対して訓練・予測時間が長い
# - テキストデータなどの非常に高次元で疎なデータが苦手←線形モデルのが得意
# - メモリ消費が大きい
# 
# ## パラメータ
# 
# - n_job
# 
#     並列処理するコア数 `n_job=-1`で全てのコアを使う
# 
# - random_state
# 
#     固定推奨
# 
# - n_estimators
# 
#     決定木の本数。多いほど良い
# 
# - max_features
# 
#     個々の決定木の乱数性を決める。小さくなると過剰適合が低減。デフォルト推奨(クラス分類では、sqrt(n_features)。回帰ではn_featuresがデフォルト)
# 
# - max_leaf_nodes
# 
#     ノードの最大数。
# 
# `max_features`と`max_leaf_nodes`を調整することで性能が上がることがある。また、訓練・予測時間が縮まる可能性がある。
# %% [markdown]
# ## 勾配ブースティング回帰木(勾配ブースティングマシン) P.87
# 
# ランダムフォレストとは対象的なアンサンブル手法。
# 乱数の代わりに強力な事前枝刈りを持つ。

# %%
# デフォルトでは深さ3の決定木が100個作られ、学習率は0.1

from sklearn.ensemble import GradientBoostingClassifier

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))
# 過剰適合を起こしている。深さの最大値を制限するか、学習率を下げてみる


# %%
# 深さを制限
gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))


# %%
# 学習りつを下げる
gbrt = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
gbrt.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))


# %%
gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt.fit(X_train, y_train)

plot_feature_importances_cancer(gbrt)
# 特徴量重要度はランダムフォレストに似ているが、いくつかの特徴量が完全に無視している。

# %% [markdown]
# ランダムフォレストのが頑健だから先にランダムフォレストを試して、予測時間が重要だったり予測精度をさらに上げたい場合は勾配ブースティングを試す。
# 
# 大きい問題に適用したい場合、`xgboost`パッケージがscikit-learnよりも高速でチューニングが容易。
# %% [markdown]
# P.90
# ## 長所
# 
# - 教師あり学習で最も強力
# - 様々な特徴量に対してうまく機能する
# 
# ## 短所
# 
# - パラメータのチューニングに注意が必要
# - 訓練に時間がかかる
# - 高次元の疎なデータは苦手
# 
# ## パラメータ
# 
# - n_estimate
# 
#     決定木の本数(多くすると複雑になり過剰適合の可能性)
# 
# - learning_late
# 
#     個々の決定木がそれまでの決定木の誤りを補正する度合い
# 
# - max_depth(or max_leaf_nodes)
# 
#     個々の決定木の深さ(5以上になることはあまりない)
# 
# 基本的には`n_estimate`を時間とメモリ量で決めてから`learning_late`に対して探索を行う。
# %% [markdown]
# P. 91
# # カーネル法を用いたサポートベクタマシン
# 
# 線形モデルで取り上げた線形サポートベクタマシンの入力空間の超平面のような簡単なモデルではなく、より複雑なモデルを可能にするために線形サポートベクタマシンを拡張したもの
# 
# 回帰でもクラス分類にも使えるが、ここではSVCとして話す。
# 
# 線形モデルは元々制約が強い。直線や超平面でしか分けられないから。
# 
# 線形モデルを柔軟にする方法が、特徴量を追加すること。

# %%
# 決定木の特徴量の重要性で用いた合成データセット

X, y = make_blobs(centers=4, random_state=8)
y = y % 2

mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")


# %%
# 線形モデルで学習

from sklearn.svm import LinearSVC

linear_svm = LinearSVC().fit(X, y)

mglearn.plots.plot_2d_separator(linear_svm, X)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")


# %%
# 入力特徴量を拡張する。
# feature ** 2 を新しい特徴量として加えてみる。

X_new = np.hstack([X, X[:, 1:] ** 2])

from mpl_toolkits.mplot3d import Axes3D, axes3d
figure = plt.figure()
# 3dで可視化
ax = Axes3D(figure, elev=-152, azim=-26)
# y == 0 の点をプロットしてからy == 1の点をプロット
mask = y == 0
ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b', cmap=mglearn.cm2, s=60)
ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^', cmap=mglearn.cm2, s=60)
ax.set_xlabel("feature0")
ax.set_ylabel("feature1")
ax.set_zlabel("feature1 ** 2")

# 平面を用いて分離できそう


# %%
mask
# X_new[mask, 0]でX_new配列のうちTrueになってる行を取り出し、0番目の列を抽出している


# %%
# 拡張されたデータセットに対して線形モデルを適用

linear_svm_3d = LinearSVC().fit(X_new, y)
coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_

# 線形決定境界を描画
figure = plt.figure()
ax = Axes3D(figure, elev=-152, azim=-26)
xx = np.linspace(X_new[:, 0].min() - 2, X_new[:, 0].max() + 2, 50)
yy = np.linspace(X_new[:, 1].min() - 2, X_new[:, 1].max() + 2, 50)
XX, YY = np.meshgrid(xx, yy)
ZZ = (coef[0] * XX + coef[1] * YY + intercept) / -coef[2]
ax.plot_surface(XX, YY, ZZ, rstride=8, cstride=8, alpha=.3)
ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b', cmap=mglearn.cm2, s=60)
ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^', cmap=mglearn.cm2, s=60)
ax.set_xlabel("feature0")
ax.set_ylabel("feature1")
ax.set_zlabel("feature1 ** 2")

# mashgrid: https://deepage.net/features/numpy-meshgrid.html


# %%
ZZ = YY ** 2
dec = linear_svm_3d.decision_function(np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()])
plt.contourf(XX, YY, dec.reshape(XX.shape), levels=[dec.min(), 0, dec.max()], cmap=mglearn.cm2, alpha=.5)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

# 決定境界を元の二つの特徴量の関数として表示。楕円に近くなっている


# %%
# RBF(radial basis function)カーネルを用いたSVM
# 2クラスの境界に位置している訓練データポイントをサポートベクタと呼ばれ、決定境界を定めている
# 縁取られた点がサポートベクタ

from sklearn.svm import SVC

X, y = mglearn.tools.make_handcrafted_dataset()
svm = SVC(kernel='rbf', C=10, gamma=0.1).fit(X, y)
mglearn.plots.plot_2d_separator(svm, X, eps=.5)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# サポートベクタをプロット
sv = svm.support_vectors_
sv_labels = svm.dual_coef_.ravel() > 0
mglearn.discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s=15, markeredgewidth=3)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

# %% [markdown]
# **詳しくはP96へ**
# 
# 新しいデータポイントに対して予測を行う時に、サポートベクタとデータポイントとの距離が測定される。
# クラス分類は、サポートベクタの距離と、訓練過程で学習されたサポートベクタの重要性(dual_coef_)によって決定される。
# 
# ### サポートベクタの表示
# 
# ```
# svm.support_vectors_
#                                      #クラス(グラフでいう青:0 赤:1)
# array([[ 8.1062269 ,  4.28695977],   #0
#        [ 9.50169345,  1.93824624],   #0
#        [11.563957  ,  1.3389402 ],   #0
#        [10.24028948,  2.45544401],   #1
#        [ 7.99815287,  4.8525051 ]])  #1
# ```
# 
# サポートベクタのクラスラベルはdual_coef_の正負によって与えられている
# 
# ```
# svm.dual_coef_
# 
# array([[-10.        ,  -6.25178295,  -3.73381586,  10.        ,
#           9.98559881]])
# ```
# 
# ```
# svm.dual_coef_ > 0
# 
# array([[False, False, False,  True,  True]])
# ```

# %%
# SVMのパラメータはC:正則化パラメータ。重要度(dual_coef_)を制限, gammma:ガウシアンカーネルの幅。点が近いことを意味するスケール
# パラメータをいじってみる

fig, axes = plt.subplots(3, 3, figsize=(15, 10))

for ax, C in zip(axes, [-1, 0, 3]):
    for a, gamma in zip(ax, range(-1, 2)):
        mglearn.plots.plot_svm(log_C=C, log_gamma=gamma, ax=a)

axes[0, 0].legend(["class 0", "class 1", "sv class 0", "sv class 1"], ncol=4, loc=(.9, 1.2))

# gamma:大→個々のデータポイントを重視するように(モデルが複雑) ←gammaはカウシアンカーネルの幅の逆数だから
# C:大→より正しくクラス分類されるように決定境界を曲げる(モデルが複雑)


# %%
# RBFカーネルを用いたSVMをcancerに適用してみる
# デフォルトC:1, gamma=1/n_featuresってことになってるけど、どうやらこのバージョンではgammaのデフォルトは違うっぽい？
# デフォルトでの結果
# Accuracy on training set: 0.90
# Accuracy on test set: 0.94

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

svc = SVC(gamma=1/cancer.data.shape[1])
svc.fit(X_train, y_train)

print("Accuracy on training set: {:.2f}".format(svc.score(X_train, y_train)))
print("Accuracy on test set: {:.2f}".format(svc.score(X_test, y_test)))

# 強く過剰適合している
# パラメータとデータのスケールに敏感。特に全ての特徴量が同じスケールである必要がある


# %%
# 個々の特徴量の最大値と最小値を対数でプロット

# axis=0: 列ごと 1: 行ごと
plt.plot(X_train.min(axis=0), 'o', label="min")
plt.plot(X_train.max(axis=0), '^', label="max")
plt.legend(loc=4)
plt.xlabel("Feature index")
plt.ylabel("Feature magnitude")
plt.yscale("log")


# %%
plt.boxplot(X_train, sym='+')
plt.ylim(10**-1, 10**4)
plt.xlabel("Feature index")
plt.ylabel("Feature magnitude")
plt.yscale("log")


# %%
# 同じスケールになるように変換する必要がある。大抵は0~1の間になるようにする
# 詳しくは3章で

# 訓練セットごとの特徴量ごとに最小値を計算
min_on_training = X_train.min(axis=0)
# 特徴量ごとにレンジ(max - min)を計算(各データから最小値で引いてその中の最大値を取る)
range_on_training = (X_train - min_on_training).max(axis=0)

# 最小値を引いてレンジで割る
# 個々の特徴量はmin=0, max=1になる
X_train_scaled = (X_train - min_on_training) / range_on_training
print("Minimum for each featire\n{}".format(X_train_scaled.min(axis=0)))
print("Maximum for each featire\n{}".format(X_train_scaled.max(axis=0)))


# %%
# テストセットに対しても変換
# 訓練セットの最小値とレンジを用いる

X_test_scaled = (X_test - min_on_training) / range_on_training


# %%
svc = SVC(gamma=1/cancer.data.shape[1])
svc.fit(X_train_scaled, y_train)

print("Accuracy on training set: {:.3f}".format(svc.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(svc.score(X_test_scaled, y_test)))

# 前処理を行うことで結果が大きく一変した。適合不足になっている。


# %%
# パラメータを調整する C:大→より正しくクラス分類されるように決定境界を曲げる(モデルが複雑)

svc = SVC(C=1000, gamma=1/cancer.data.shape[1])
svc.fit(X_train_scaled, y_train)

print("Accuracy on training set: {:.3f}".format(svc.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(svc.score(X_test_scaled, y_test)))

# %% [markdown]
# ## 特徴
# - 特徴量が似た測定器の測定結果(カメラのピクセルなど)のように、同じスケールになる場合に有用
# 
#     例えば、画像なら高次元(画素数)であり同じスケール(R,G,Bで0~255までの値までしかない)である
# 
# ## 長所
# - 様々なデータセットに対してうまく機能する
# - データの特徴量が少ない時にも複雑な決定境界を作成可能
# 
#     低次元でも高次元でもうまく機能する 
# 
# ## 短所
# - サンプルの個数が大きくなるとうまく機能しない
# 
#     SVMは10,000サンプルまでは機能するが、100,000サンプルになると実行時間・メモリ使用量の面で難しい
# 
# - 注意深くデータの前処理とパラメータ調整を行う必要がある
# 
# - 検証が難しい。あるモデルが予測された理由を理解しづらい
# 
# ## パラメータ
# 
# - C: 正則化パラメータ
# 
#     線形クラス分類と同じパラメータ。C:大→より正しくクラス分類されるように決定境界を曲げる(モデルが複雑)
# 
# - カーネルの選択
# 
#     今回はRBF(radical basis function)カーネルだけ見たが、他にもいろいろある: 多項式カーネル(polynomial kernel)など
# 
# - カーネル固有のパラメータ(RBFの場合 gamma)
# 
#     ガウシアンカーネルの幅の逆数。gamma:大→個々のデータポイントを重視するように(モデルが複雑)
#     
#     カーネル幅が大きくなる(gamma: 小)と、ある点が近いとみなす範囲が広くなり決定境界が緩くなる
# 
# `C`と`gamma`は共にモデルの複雑さを制御するパラメータで、大きくするとより複雑なモデルとなる。
# 2つのパラメータ設定は強く相関するため、同時に調整すること。
# 
# 本ではデフォルトC:1, gamma=1/n_featuresってことになってるけど、どうやらこのバージョンではgammaのデフォルトは違うっぽい？
