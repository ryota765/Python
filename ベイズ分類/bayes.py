import math, sys
from janome.tokenizer import Tokenizer
import pickle

class BayesianFilter:
    def __init__(self):
        self.words = set()  #出現した単語を全て記録
        self.word_dict = {}  #カテゴリごとの単語出現回数を記録
        self.category_dict = {}  #カテゴリの出現回数を記録

    #形態素解析を行う
    def split(self, text):
        result = []
        t = Tokenizer()
        malist = t.tokenize(text)
        for w in malist:
            sf = w.surface  #単語そのまま
            bf = w.base_form  #単語の基本形
            if bf == '' or bf == '*': bf = sf
            result.append(bf)
        return result

    #単語とカテゴリをカウント
    def inc_word(self, word, category):
        if not category in self.word_dict:
            self.word_dict[category] = {}
        if not word in self.word_dict[category]:
            self.word_dict[category][word] = 0
        self.word_dict[category][word] += 1
        self.words.add(word)
    def inc_category(self, category):
        if not category in self.category_dict:
            self.category_dict[category] = 0
        self.category_dict[category] += 1

    #テキストを学習する
    def fit(self, text, category):
        word_list = self.split(text)
        for word in word_list:
            self.inc_word(word, category)
        self.inc_category(category)
    
    def score(self, words, category):
        score = math.log(self.category_prob(category))
        for word in words:
            score += math.log(self.word_prob(word, category))
        return score
    
    def predict(self, texts, labels):

        #ファイルから辞書を取得
        with open('category_dict', mode='rb') as f1:
            self.category_dict = pickle.load(f1)
        with open('word_dict', mode='rb') as f2:
            self.word_dict = pickle.load(f2)

        base_category = None
        max_score = -sys.maxsize
        pred_list = []
        for text in texts:
            words = self.split(text)
            score_dict = {}
            for category in self.category_dict.keys():
                score = self.score(words, category)
                score_dict[category] = score 
                #score_list.append((category, score))
                #if score > max_score:
                    #max_score = score
                    #best_category = category
            print(len(score_dict))
            best_category = max(score_dict)
            del score_dict[best_category]
            next_category = max(score_dict)
            pred_list.append([best_category, next_category])
        print('pred_list={}'.format(pred_list))

        #dictionaryを保存
        '''f1 = open('category_dict','wb')
        pickle.dump(self.category_dict,f1)
        f1.close
        f2 = open('word_dict','wb')
        pickle.dump(self.word_dict,f2)
        f2.close'''

        #return pred_list

        count = 0
        count2 = 0
        for i in range(len(labels)):
            if labels[i] == pred_list[i][0]:
                count += 1
            if labels[i] == pred_list[i][0] or labels[i] == pred_list[i][1]:
                count2 += 1
        acc = count / len(labels)
        acc2 = count2 / len(labels)
        return acc, acc2
    
    def get_word_count(self, word, category):
        if word in self.word_dict[category]:
            return self.word_dict[category][word]
        else:
            return 0
    
    #カテゴリ/総カテゴリを計算
    def category_prob(self, category):
        sum_categories = sum(self.category_dict.values())
        category_v = self.category_dict[category]
        return category_v / sum_categories
    
    #カテゴリ内の単語の出現率を計算
    def word_prob(self, word, category):
        n = self.get_word_count(word, category) + 1 
        d = sum(self.word_dict[category].values()) + len(self.words)
        return n / d 

