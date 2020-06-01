# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 22:34:26 2020

@author: lilliloo
"""

#-------------------------------------------#

#1.初期化：入力層・隠れ層・出力層の各層のノード数を設定する
#2.学習：与えられた訓練データから重みを調整する
#3.答えの出力：入力層で入力情報を受け取り、出力層から答えを返す

#-------------------------------------------#

# -------  Library  -------------#
import numpy as np
import scipy.special


# -------  Function  -------------#

class neuralNetwork:
    
    # ニューラルネットワークの初期化
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 入力層・隠れ層・出力層のノード数の設定
        self.inodes = inputnodes # 入力層
        self.hnodes = hiddennodes # 隠れ層
        self.onodes = outputnodes # 出力層
        
        #学習率の設定
        self.lr = learningrate
        
        #重みの初期値の決め方は、平均値が0であり、
        # ノードに入るリンクの数の平方根の逆数を標準偏差に持つ正規分布からサンプリングしてくる
        # リンクの重み行列 wih, who
        # 行列内の重み Wij = ノードiから次の層のノードjへのリンクの重み        
        # 行列
        # W11 W21
        # W12 W22
        # などを作成する         
        # 入力層→隠れ層へのリンクの重み（wih）
        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        # 隠れ層→出力層へのリンクの重み（who）
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
        #numpy.random.normal(av, sd, (n, m))
        #平均av、標準偏差sdの正規分布にしたがう乱数を(n, m)の行列で出力します。
        
        
        # 活性化関数はシグモイド関数
        self.activation_function = lambda x: scipy.special.expit(x)
        pass
 
    # ニューラルネットワークの学習
    def train(self, inputs_list, targets_list):
        # 入力リストを行列に変換
        inputs = np.array(inputs_list, ndmin=2).T
        # 真値リストを行列に変換
        targets = np.array(targets_list, ndmin=2).T
     
        # 隠れ層に入ってくる信号の計算
        hidden_inputs = np.dot(self.wih, inputs)
        # 隠れ層から出る信号に活性化関数を作用させる
        hidden_outputs = self.activation_function(hidden_inputs)
     
        # 出力層に入ってくる信号の計算
        final_inputs = np.dot(self.who, hidden_outputs)
        # 出力層から出る最終的な出力に活性化関数を作用させる
        final_outputs = self.activation_function(final_inputs)
        
        # 出力の誤差 = 真値 - 最終的な出力
        output_errors = targets - final_outputs
        # 隠れ層に伝播された誤差はリンクの重みづけをして結合
        hidden_errors = np.dot(self.who.T, output_errors)
     
     
        # 隠れ層と出力層間のリンクの重みの更新
        self.who += self.lr * np.dot( \
          (output_errors * final_outputs * (1.0 - final_outputs)), \
          np.transpose(hidden_outputs))
     
        # 入力層と隠れ層間のリンクの重みの更新
        self.wih += self.lr * np.dot( \
          (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), \
          np.transpose(inputs))
        pass
 
    # ニューラルネットワークの答えの出力
    def query(self, inputs_list):
        # 入力リストを行列に変換(.Tを使って転置している)
        inputs = np.array(inputs_list, ndmin=2).T
 
        # 入力信号が隠れ層に入ってくるまでの計算
        hidden_inputs = np.dot(self.wih, inputs)
        # 隠れ層からの信号にシグモイド関数を作用させて出力層へ渡す
        hidden_outputs = self.activation_function(hidden_inputs)
 
        # 出力層に入ってくる信号の計算
        final_inputs = np.dot(self.who, hidden_outputs)
        # 出力層からの信号にシグモイド関数を作用させて最終的な出力とする
        final_outputs = self.activation_function(final_inputs)
 
        # 関数の戻り値として最終的な出力結果を返す
        return final_outputs



# --- ユーザー設定 ---
# 入力層・隠れ層・出力層のノード数の設定
input_nodes = 4
hidden_nodes = 4
output_nodes = 4
 
# 学習率の設定
learning_rate = 0.3
# ---
 
 
# ニューラルネットワークのインスタンスの生成
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)



print(n.query([1.0, 0.5, -0.5, 0.5]))























