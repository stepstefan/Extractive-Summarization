import pickle
import os, sys
import time
sys.setrecursionlimit(2500)

dir_name = sys.argv[1]
end_name = sys.argv[2]
alpha = 0

for idx, pic_name in enumerate(os.listdir(dir_name)):
    print('Normalizing: {} ({}/{})'.format(pic_name, idx+1, len(os.listdir(dir_name))))
    file_list = pickle.load(open(dir_name 
        + pic_name, 'rb'))

    reference = []
    for s in os.listdir('summeries/'):
        if s.startswith(pic_name.split('.')[0]):
            reference.append('summeries/' + s)

    tf         = 0
    idf        = 0
    cf         = 0
    sl         = 0
    stf        = 0
    stf        = 0
    sidf       = 0
    ss         = 0
    sd         = 0

    position   = 0
    length     = 0
    ss         = 0 
    depth      = 0
    atf        = 0
    acf        = 0
    aidf       = 0
    neR        = 0
    numR       = 0
    stopwordR  = 0

    for ltree in file_list:
        for tree in ltree:
            sen_ftrs = tree.sentence_features
            position   = max(position, sen_ftrs[0])
            length     = max(length, sen_ftrs[1])
            ss         = max(ss, sen_ftrs[2])
            depth      = max(depth, sen_ftrs[3])
            atf        = max(atf, sen_ftrs[4])
            acf        = max(acf, sen_ftrs[5])
            aidf       = max(aidf, sen_ftrs[6])
            neR        = max(neR, sen_ftrs[8])
            numR       = max(numR, sen_ftrs[9])
            stopwordR  = max(stopwordR, sen_ftrs[10])
            

            for node in tree.getTerminals():
                ftrs = node.feature
                tf = max(tf, ftrs[0])
                idf = max(idf, ftrs[1])
                cf = max(cf, ftrs[2])
                sl = max(sl, ftrs[6])
                stf = max(stf, ftrs[7])
                scf = max(stf, ftrs[8])
                sidf = max(sidf, ftrs[9])
                ss = max(ss, ftrs[10])
                sd = max(sd, ftrs[11])

    for ltree in file_list:
        for tree in ltree:
            tree.addSalience(reference, alpha)
            tree.sentence_features[0] /=  position  + (0 == position)
            tree.sentence_features[1] /=  length    + (0 == length)
            tree.sentence_features[2] /=  ss        + (0 == ss)
            tree.sentence_features[3] /=  depth     + (0 == depth)
            tree.sentence_features[4] /=  atf       + (0 == atf)
            tree.sentence_features[5] /=  acf       + (0 == acf)
            tree.sentence_features[6] /=  aidf      + (0 == aidf)
            tree.sentence_features[8] /=  neR       + (0 == neR)
            tree.sentence_features[9] /=  numR      + (0 == numR)
            tree.sentence_features[10] /= stopwordR + (0 == stopwordR)

            for node in tree.getTerminals():
                node.feature[0] /= tf + (0 == tf)
                node.feature[1] /= idf + (0 == idf)
                node.feature[2] /= cf + (0 == cf)
                node.feature[6] /= sl + (0 == sl)
                node.feature[7] /= stf + (0 == stf)
                node.feature[8] /= scf + (0 == scf)
                node.feature[9] /= sidf + (0 == sidf)
                node.feature[10] /= ss + (0 == ss)
                node.feature[11] /= sd + (0 == sd)

    pickle.dump(file_list, open(end_name 
        + pic_name, 'wb'))
