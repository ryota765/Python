from bayes import BayesianFilter
import csv
import pandas as pd
from sklearn.model_selection import train_test_split

bf = BayesianFilter()

#chat_csvをリストに格納
def csv_opener(csv_name):
    text = []
    with open(csv_name, 'r', encoding="utf_8_sig") as f:
        reader = csv.reader(f)
        for row in reader:
            text.append(row)
        return pd.DataFrame(text).T.values.tolist()

text = csv_opener('onayami.csv')
#text = [text[0][0:50],text[1][0:50]]

#(x_train, x_test, y_train, y_test) = train_test_split(text[0], text[1], test_size=0.3)

#すでにファイルに出力済みなら不要
#for i in range(len(text[0])):
    #bf.fit(text[0][i],text[1][i])

acc, acc2 = bf.predict(text[0][0:10], text[1][0:10])

print('結果1=', acc)
print('結果2=', acc2)

#pred = bf.predict(text[0][0:3], text[1][0:3])
#print(pred)

'''
bf.fit('今なら50%OFF','広告')
bf.fit('激安セール - 今日だけ三割引','広告')
bf.fit('クーポンプレゼント & 送料無料','広告')
bf.fit('店内改装セール実施中','広告')
bf.fit('美味しくなって再登場','広告')
bf.fit('早めに応募された方にスペシャルオファー','広告')

bf.fit('本日の予定の確認です','重要')
bf.fit('お世話になっております','重要')
bf.fit('プロジェクトの進捗確認をお願いいたします','重要')
bf.fit('打ち合わせよろしくお願いいたします','重要')
bf.fit('会議の議事録です','重要')
bf.fit('本日の会食のリマインドです','重要')

pre, scorelist = bf.predict('在庫一掃セール実施')
'''
