
from rouge import Rouge
import jieba
from predict import Predict

import config

class RougeEval():
    def __init__(self, path):
        self.path = path
        self.scores = None
        self.rouge = Rouge()
        self.sources = []
        self.hypos = []
        self.refs = []
        self.process()

    def process(self):
        print('Reading from ', self.path)
        with open(self.path, 'r',encoding='utf-8') as test:
            n=0
            for line in test:
                n+=1
                if n<5:
                    source, ref = line.strip().split('<sep>')
                    ref = ''.join(list(jieba.cut(ref))).replace('。', '.')
                    self.sources.append(source)
                    self.refs.append(ref)
        print(f'Test set contains {len(self.sources)} samples.')

    def build_hypos(self, predict):
        """Generate hypos for the dataset.

        Args:
            predict (predict.Predict()): The predictor instance.
        """
        print('Building hypotheses.')
        count = 0
        for source in self.sources:
            count += 1
            if count % 100 == 0:
                print(count)
            self.hypos.append(predict.predict(source.split(),beam_search=True))

    def get_average(self):
        assert len(self.hypos) > 0, 'Build hypotheses first!'
        print('Calculating average rouge scores.')
        return self.rouge.get_scores(self.hypos, self.refs, avg=True)

    def one_sample(self, hypo, ref):
        return self.rouge.get_scores(hypo, ref)[0]

rouge_eval = RougeEval(config.test_data_path)
predict = Predict()
rouge_eval.build_hypos(predict)
result = rouge_eval.get_average()
print('rouge1: ', result['rouge-1'])
print('rouge2: ', result['rouge-2'])
print('rougeL: ', result['rouge-l'])
with open('./data/rouge_result.txt', 'a',encoding='utf-8') as file:
    for r, metrics in result.items():
        file.write(r+'\n')
        for metric, value in metrics.items():
            file.write(metric+': '+str(value*100))
            file.write('\n')