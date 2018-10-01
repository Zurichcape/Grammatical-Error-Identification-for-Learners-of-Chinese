import xml.etree.ElementTree as ET
import numpy as np
import pycrfsuite
import re
tree = ET.parse('2017_clean.xml')
# tree = ET.parse('testfile')
root = tree.getroot()
# find the DOC element
doc = root.findall('./DOC')
ls_sentence = []
for sentence in root.iter('TEXT'):
    sentence_strip = sentence.text.strip('\n')
    ls_sentence.append(sentence_strip)
# print(len(ls_sentence))
ls_error = []
for i in doc:
    ls_error.append(i.findall('./ERROR'))
# print(ls_error)
error_index = []
for i in range(0, len(ls_sentence)):
    x = []
    for j in range(0, len(ls_error[i])):
        start_off = int(ls_error[i][j].attrib['start_off'])
        end_off = int(ls_error[i][j].attrib['end_off'])+1
        for m in range(start_off, end_off):
            x.append(m)
    error_index.append(x)
bad_index = []
for i in range(0, len(error_index)):
    if len(error_index[i]) != len(set(error_index[i])):
        bad_index.append(i)
# print(bad_index)
# print(len(bad_index))
# print(error_index)

error_list = []
for i in range(0, len(ls_sentence)):
    error_sent = []
    for j in range(0, len(ls_error[i])):
        start_off = int(ls_error[i][j].attrib['start_off'])
        end_off = int(ls_error[i][j].attrib['end_off'])+1
        label = ls_error[i][j].attrib['type']
        error_sent.append(([start_off, end_off], label))
    error_list.append(error_sent)
# print(error_list)


word_label = []
for i in range(0, len(ls_sentence)):
    sentence_label = []
    for index in range(0, len(ls_sentence[i])):
        if index+1 not in error_index[i]:
            sentence_label.append((ls_sentence[i][index: index + 1], 'O'))
        else:
            for m in range(0, len(error_list[i])):
                start = error_list[i][m][0][0]
                end = error_list[i][m][0][1]
                for k in range(start, end):
                    if index + 1 == k:
                            sentence_label.append((ls_sentence[i][index:index + 1], error_list[i][m][1]))
    word_label.append(sentence_label)
word_label_modi = np.delete(word_label, bad_index).tolist()
# print(len(word_label_modi))
# print(word_label[0])

def pos_charater_produce(file):
    with open(file) as f:
        lst_n = []
        for line in f:
            x = line.strip('\n')
            y = x.split(' ')
            lst = []
            for item in y:
                m = item.split('#')
                lst.append(m)
            lst_n.append(lst)
    # print(lst_n)

    pos_character = []
    for i in range(0, len(lst_n)):
        pos_sentence = []
        for j in range(0, len(lst_n[i])):
            if len(lst_n[i][j][0]) > 1:
                pos_sentence.append((lst_n[i][j][0][0], 'B-' + lst_n[i][j][1]))
                for k in range(1, len(lst_n[i][j][0])):
                    pos_sentence.append((lst_n[i][j][0][k:k+1], 'I-' + lst_n[i][j][1]))
            else:
                pos_sentence.append((lst_n[i][j][0], lst_n[i][j][1]))
        pos_character.append(pos_sentence)
    return pos_character
pos_character_modi = np.delete(pos_charater_produce('sentences_2017_cleantagged.txt'), bad_index).tolist()

# print(pos_character_modi)

c_pos_label = []
for i in range(0, len(word_label_modi)):
    s_pos_label = []
    for j in range(0, len(word_label_modi[i])):
        x = list(word_label_modi[i][j])
        x.insert(1, pos_character_modi[i][j][1])
        y = tuple(x)
        s_pos_label.append(y)
    # print(s_pos_label)
    # print(i)

    c_pos_label.append(s_pos_label)
# print(c_pos_label[0])

# change postagging according to stanford corenlp

with open('sentences_2017_clean_modi.txt.conll') as fin:
# with open('test.txt.conll') as fin:
    lst = []
    for f in fin:
        if f != '\n':
            lst.append((f.split('\t')[1], f.split('\t')[3], f.split('\t')[6].strip('\n')))

    # print(lst)
    finalst = []
    for i in lst:
        if len(i[0]) > 1:
            finalst.append((i[0][0], 'B-' + i[1], 'B-' + i[2]))
            for j in range(1, len(i[0])):
                finalst.append((i[0][j], 'I-' + i[1], 'I-' + i[2]))
        else:
            finalst.append(i)

num = 0
for sentence in c_pos_label:
# for sentence in c_p_l:
    for i in range(0, len(sentence)):
        if sentence[i][0] == finalst[i+num][0]:
            sentence[i] = list(sentence[i])
            sentence[i][1] = finalst[i+num][1]
            sentence[i] = tuple(sentence[i])
    # num += len(sentence)
        else:
            print('error')
    num += len(sentence)


def character2feature(doc, i):
    character = doc[i][0]
    postag = doc[i][1]
    # common features for all characters
    features = ['character=' + character, 'postag=' + postag]
    if i > 0:
        character1 = doc[i-1][0]
        postag1 = doc[i-1][1]
        features.extend(['-1:character=' +character1, '-1:postag=' + postag1])
        if i > 1:
            character2 = doc[i - 2][0]
            postag2 = doc[i - 2][1]
            features.extend(['-2:character=' + character2, '-2:postag=' + postag2])
            if i > 2:
                character3 = doc[i-3][0]
                postag3 = doc[i-3][1]
                features.extend(['-3:character=' + character3, '-3:postag=' + postag3])
                if i > 3:
                    character4 = doc[i - 4][0]
                    postag4 = doc[i - 4][1]
                    features.extend(['-4:character=' + character4, '-4:postag=' + postag4])
                    if i > 4:
                        character5 = doc[i - 5][0]
                        postag5 = doc[i - 5][1]
                        features.extend(['-5:character=' + character5, '-5:postag=' + postag5])
                        if i > 5:
                            character6 = doc[i - 6][0]
                            postag6 = doc[i - 6][1]
                            features.extend(['-6:character=' + character6, '-6:postag=' + postag6])
                        else:
                            features.append('SIBOS')
                    else:
                        features.append('FIBOS')
                else:
                    features.append('FBOS')
            else:
                features.append('TBOS')
        else:
            # Indicate that it is the 'second beginning of a document'
            features.append('SBOS')
    else:
        # Indicate that it is the 'beginning of a document'
        features.append('BOS')
    if i < len(doc)-1:
        character1 = doc[i+1][0]
        postag1 = doc[i+1][1]
        features.extend(['+1:character=' +character1, '+1:postag=' + postag1])
        if i < len(doc)-2:
            character2 = doc[i + 2][0]
            postag2 = doc[i + 2][1]
            features.extend(['+2:character=' +character2, '+2:postag=' + postag2])
            if i < len(doc)-3:
                character3 = doc[i+3][0]
                postag3 = doc[i+3][1]
                features.extend(['+3:character=' +character3, '+3:postag=' + postag3])
                if i < len(doc) - 4:
                    character4 = doc[i + 4][0]
                    postag4 = doc[i + 4][1]
                    features.extend(['+4:character=' + character4, '+4:postag=' + postag4])
                    if i < len(doc) - 5:
                        character5 = doc[i + 5][0]
                        postag5 = doc[i + 5][1]
                        features.extend(['+5:character=' + character5, '+5:postag=' + postag5])
                        if i < len(doc) - 6:
                            character6 = doc[i + 6][0]
                            postag6 = doc[i + 6][1]
                            features.extend(['+6:character=' + character6, '+6:postag=' + postag6])
                        else:
                            features.append('SIEOS')
                    else:
                        features.append('FIEOS')
                else:
                    features.append('FEOS')
            else:
                features.append('TEOS')
        else:
            features.append('SEOS')
    else:
        # Indicate that it is the 'end of a document'
        features.append('EOS')
    return features

# A function for extracting features in documents
def extract_features(doc):
    return [character2feature(doc, i) for i in range(len(doc))]

# A function fo generating the list of labels for each document
def get_labels(doc):
    return [label for (c, postag, label) in doc]


X_train = [extract_features(doc) for doc in c_pos_label]
y_train = [get_labels(doc) for doc in c_pos_label]

# print(len(X_train))


trainer = pycrfsuite.Trainer(verbose=True)

# Submit training data to the trainer
for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)

#Set the parameters of the model
trainer.set_params({
    # coefficient for L1 penalty
    'c1': 0.1,

    # coefficient for L2 penalty
    'c2': 0.01,

    # maximum number of iterations
    'max_iterations': 200,

    # whether to include transitions that
    # are possible, but not observed
    'feature.possible_transitions': True
})

# Provide a file name as a parameter to the train function, such that
# the model will be saved to the file when training is finished
trainer.train('crf.model')

pos_character_test = pos_charater_produce('2017_testtagged.txt')

with open('2017_test.txt.conll') as fin:
    lst = []
    for f in fin:
        if f != '\n':
            lst.append((f.split('\t')[1], f.split('\t')[3], f.split('\t')[6].strip('\n')))

    # print(lst)
    finalst = []
    for i in lst:
        if len(i[0]) > 1:
            finalst.append((i[0][0], 'B-' + i[1], 'B-' + i[2]))
            for j in range(1, len(i[0])):
                finalst.append((i[0][j], 'I-' + i[1], 'I-' + i[2]))
        else:
            finalst.append(i)
    # print(finalst)

n = 0
for sentence in pos_character_test:
    for i in range(0, len(sentence)):
        if sentence[i][0] == finalst[i+n][0]:
            sentence[i] = list(sentence[i])
            sentence[i][1] = finalst[i+n][1]
            sentence[i] = tuple(sentence[i])
        else:
            print('error')
    n += len(sentence)
# print(pos_character_test)

X_test = [extract_features(doc) for doc in pos_character_test]
# print(len(X_test))


tagger = pycrfsuite.Tagger()
tagger.open('crf.model')
y_pred = [tagger.tag(xseq) for xseq in X_test]
# print(y_pred)

reGex = '\(([^)]+)\)'
with open('test.HSK.Input.txt') as fn:
    num_lst = []
    for line in fn:
        x = re.search(reGex, line).group()
        sentenceNum = x.split('=')[1].strip(')')
        num_lst.append(sentenceNum)
    # print(len(num_lst))

predList = []
for i in y_pred:
    errorLs = []
    correctLs = []
    for j in range(0, len(i)):
        if i[j] != 'O':
            errorLs.append((j, i[j]))
        else:
            correctLs.append((j, i[j]))
    if len(correctLs) == len(i):
        predList.append('correct')
    else:
        predList.append(errorLs)
# print(predList)

lst_f = []
for s in predList:
    if s != 'correct':
        dict = {}
        for n in range(0, len(s)):
            if s[n][1] not in dict:
                dict[s[n][1]] = [s[n][0]]
            else:
                dict[s[n][1]].append(s[n][0])
        lst_f.append(dict)
    else:
        lst_f.append(s)
# print(lst_f)

lst_final = []
for i in lst_f:
    if i != 'correct':
        keys = list(i.keys())
        lst_new = []
        for key in keys:
            if len(i[key]) == 1:
                lst_new.append((key, (i[key][0]+1, i[key][0]+1)))
            elif i[key][-1] - i[key][0] == len(i[key]) - 1:
                lst_new.append((key, (i[key][0] + 1, i[key][-1] + 1)))
            elif i[key][-1] - i[key][0] != len(i[key]) - 1:
                sequences = np.split(i[key], np.array(np.where(np.diff(i[key]) > 1)[0]) + 1)
                lst_group = []
                for s in sequences:
                    if len(s) > 1:
                        lst_group.append((np.min(s), np.max(s)))
                    else:
                        lst_group.append(s)
                for l1 in lst_group:
                    if len(l1) == 1:
                        lst_new.append((key, (l1[0] + 1, l1[0] + 1)))
                    else:
                        lst_new.append((key, (l1[0] + 1, l1[1] + 1)))
        lst_final.append(lst_new)
    else:
        lst_final.append(i)
# print(lst_final[2896])

with open('test_input6_edited.txt', 'w') as fn:
    for i in range(0, len(num_lst)):
        if lst_final[i] != 'correct':
            if len(lst_final[i]) == 1:
                ln = num_lst[i] + ', ' + str(lst_final[i][0][1][0]) + ', ' \
                     + str(lst_final[i][0][1][1]) + ', ' + str(lst_final[i][0][0]) + '\n'
                fn.write(ln)
            else:
                for j in range(0, len(lst_final[i])):
                    l0 = num_lst[i] + ', ' + str(lst_final[i][j][1][0]) + ', ' \
                         + str(lst_final[i][j][1][1]) + ', ' + str(lst_final[i][j][0]) + '\n'
                    fn.write(l0)
        else:
            l1 = num_lst[i] + ', ' + lst_final[i] + '\n'
            fn.write(l1)