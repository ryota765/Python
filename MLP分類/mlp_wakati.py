from janome.tokenizer import Tokenizer
import pandas as pd

t = Tokenizer()

class Preprocessing:

    def __init__(self):
        self.word_dic = {'_MAX': 0} 

    #日本語を分かち書き
    def ja_tokenize(self,text_list):
        res_list = []
        for text in text_list:
            res = []
            lines = text.split("\n")
            for line in lines:
                malist = t.tokenize(line)
                for tok in malist:
                    ps = tok.part_of_speech.split(',')[0]
                    if not ps in ['名詞', '動詞', '形容詞']: continue
                    w = tok.base_form
                    if w == '*' or w == '': w = tok.surface
                    if w == '' or w == '\n': continue
                    res.append(w)
                #res.append('\n')
            res_list.append(res)
        return res_list


    #語句を区切ってIDに変換
    #単語に対して順にインデックス（ID）を振る→IDのリストをreturn
    def text_to_ids(self,text_list):
        result_list = []
        for words in text_list:
            result = []
            for n in words:
                n = n.strip()
                if n == '': continue
                if not n in self.word_dic:
                    wid = self.word_dic[n] = self.word_dic['_MAX']
                    self.word_dic['_MAX'] += 1
                    #print(wid, n)
                else:
                    wid = self.word_dic[n]
                result.append(wid)
            result_list.append(result)
        return result_list

    #IDのリストを出現回数のリストに変換
    def id_count(self,id_list):
        cnt_list = []
        for ids in id_list:
            cnt = [0 for n in range(self.word_dic['_MAX'])]
            for wid in ids:
                cnt[wid] += 1
            cnt_list.append(cnt)
        return cnt_list

    def text_to_id(self,text_list):
        #text_list = self.csv_opener('onayami.csv')
        #return text_list[0]
        return self.id_count(self.text_to_ids(self.ja_tokenize(text_list)))

    



