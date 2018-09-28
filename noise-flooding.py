import numpy as np
from scipy.io import wavfile
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier
import os
import progressbar
import pydub
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier


cnt = 0

key = ["silence", "background", "yes", "no", "up", "down", "left", "right", "on",
       "off", "stop", "go"]

def load_graph(filename):
    with tf.gfile.FastGFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

def load_audiofile(filename):
    with open(filename, 'rb') as fh:
        return fh.read()


def noisify(sound, eps):
    noise = np.asarray([np.random.randint(-1*eps, eps) for _ in sound])
    return(sound + noise)

def noisify_band(sound, eps, min, max, fps):
    noise = np.asarray([np.random.randint(-1*eps, eps) for _ in sound])
    wavfile.write("noise.wav", fps, noise.astype('int16'))
    array = pydub.audio_segment.AudioSegment.from_wav("noise.wav")
    if(min > 0):
        array = pydub.effects.high_pass_filter(array, min)
        array = pydub.effects.high_pass_filter(array, min)
        array = pydub.effects.high_pass_filter(array, min)
    if(max < fps/2):
        array = pydub.effects.low_pass_filter(array, max)
        array = pydub.effects.low_pass_filter(array, max)
        array = pydub.effects.low_pass_filter(array, max)
    array.export("noise.wav", format="wav")
    fps2, noise = wavfile.read("noise.wav")
    return(sound + noise)


def pred_sd(sess, output_node, filename):
    sound = load_audiofile(filename)
    preds = sess.run(output_node, feed_dict = {
                     'wav_data:0': sound
                     })
    return(np.argmax(preds[0]))

def simple_sd_value(sess, output_node, filename, bar):
    global cnt
    step_size = 50
    orig_pred = pred_sd(sess, output_node, filename)
    fps, sound = wavfile.read(filename)
    eps = 0
    new_pred = orig_pred
    while(new_pred == orig_pred and eps < 100000):
        eps+=step_size
        new_sound = noisify(sound, eps)
        wavfile.write("temp.wav",fps,new_sound.astype('int16'))
        new_pred = pred_sd(sess, output_node, "temp.wav")
    cnt+=1; bar.update(cnt)
    return(eps)

def band_sd_value(sess, output_node, filename, bar):
    global cnt
    step_size = 500
    orig_pred = pred_sd(sess, output_node, filename)
    fps , sound = wavfile.read(filename)
    score = []
    for i in range(4): # 4 frequency bands
        eps = 0
        new_pred = orig_pred
        while(new_pred == orig_pred and eps < 20000):
            eps+=step_size
            new_sound = noisify_band(sound, eps, i*(fps/8), (i+1)*(fps/8), fps)
            wavfile.write("temp2.wav", fps, new_sound.astype('int16'))
            new_pred = pred_sd(sess, output_node, "temp2.wav")
        score+=[eps]
    cnt+=1; bar.update(cnt)
    return(score)
        

def simple_sd_train(sess, output_node):
    global cnt
    bar = progressbar.ProgressBar(max_value=1800)
    adv = []
    for src in range(2,12):
        for trgt in range(2,12):
            case_dir = format("../success_output/result/%s/%s" %(key[trgt], key[src]))
            if os.path.exists(case_dir):
               num_files = len(os.listdir(case_dir))
               breakpoint = int(num_files/2)
               adv+=([format('%s/%s' %(case_dir, f)) for f in os.listdir(case_dir) if f.endswith('.wav')][0:breakpoint])
    ben = []
    for src in range(2,12):
        case_dir = format("data/%s" %(key[src]))
        if os.path.exists(case_dir):
            ben+=([format('%s/%s' %(case_dir, f)) for f in os.listdir(case_dir) if f.endswith('.wav')][0:90])
    cnt = 0
    advscores = [[simple_sd_value(sess, output_node, a, bar)] for a in adv]
    benscores = [[simple_sd_value(sess, output_node, b, bar)] for b in ben]
    labels = [1 for _ in adv] + [0 for _ in ben] 
    scores = advscores + benscores
    clf = DecisionTreeClassifier(max_depth=1, criterion="entropy")
    clf = clf.fit(np.asarray(scores), np.asarray(labels))
    print("Number of training adversarial examples:", len(adv))
    print("Number of training benign examples:", len(ben))
    return(clf)

def simple_sd_scores(sess, output_node, train):
    global cnt
    bar = progressbar.ProgressBar(max_value=1800)
    adv = []
    for src in range(2,12):
        for trgt in range(2,12):
            case_dir = format("../success_output/result/%s/%s" %(key[trgt], key[src]))
            if os.path.exists(case_dir):
               num_files = len(os.listdir(case_dir))
               breakpoint = int(num_files/2)
               if(train):
                   adv+=([format('%s/%s' %(case_dir, f)) for f in os.listdir(case_dir) if f.endswith('.wav')][0:breakpoint])
               else:
                   adv+=([format('%s/%s' %(case_dir, f)) for f in os.listdir(case_dir) if f.endswith('.wav')][breakpoint::])
    ben = []
    for src in range(2,12):
        case_dir = format("data/%s" %(key[src]))
        if os.path.exists(case_dir):
            if(train):
                ben+=([format('%s/%s' %(case_dir, f)) for f in os.listdir(case_dir) if f.endswith('.wav')][0:90])
            else:
                ben+=([format('%s/%s' %(case_dir, f)) for f in os.listdir(case_dir) if f.endswith('.wav')][90:180])
    cnt = 0
    advscores = [[simple_sd_value(sess, output_node, a, bar)] for a in adv]
    benscores = [[simple_sd_value(sess, output_node, b, bar)] for b in ben]
    labels = [1 for _ in adv] + [0 for _ in ben]
    scores = advscores + benscores
    print("Number of training adversarial examples:", len(adv))
    print("Number of training benign examples:", len(ben))
    return(scores)

def band_sd_scores(sess, output_node, train):
    global cnt
    bar = progressbar.ProgressBar(max_value=1800)
    adv = []
    for src in range(2,12):
        for trgt in range(2,12):
            case_dir = format("../success_output/result/%s/%s" %(key[trgt], key[src]))
            if os.path.exists(case_dir):
               num_files = len(os.listdir(case_dir))
               breakpoint = int(num_files/2)
               if(train):
                   adv+=([format('%s/%s' %(case_dir, f)) for f in os.listdir(case_dir) if f.endswith('.wav')][0:breakpoint])
               else:
                   adv+=([format('%s/%s' %(case_dir, f)) for f in os.listdir(case_dir) if f.endswith('.wav')][breakpoint::])
    ben = []
    for src in range(2,12):
        case_dir = format("data/%s" %(key[src]))
        if os.path.exists(case_dir):
            if(train):
                ben+=([format('%s/%s' %(case_dir, f)) for f in os.listdir(case_dir) if f.endswith('.wav')][0:90])
            else:
                ben+=([format('%s/%s' %(case_dir, f)) for f in os.listdir(case_dir) if f.endswith('.wav')][90:180])
    cnt = 0
    advscores = [band_sd_value(sess, output_node, a, bar) for a in adv]
    benscores = [band_sd_value(sess, output_node, b, bar) for b in ben]
    labels = [1 for _ in adv] + [0 for _ in ben]
    scores = advscores + benscores
    print("Number of adversarial examples:", len(adv))
    print("Number of benign examples:", len(ben))
    return(scores)

def index_to_src_trgt():
    global cnt
    srctrgts = []
    for src in range(2,12):
        for trgt in range(2,12):
            case_dir = format("../success_output/result/%s/%s" %(key[trgt], key[src]))
            if os.path.exists(case_dir):
               num_files = len(os.listdir(case_dir))
               breakpoint = int(num_files/2)
               files =([format('%s/%s' %(case_dir, f)) for f in os.listdir(case_dir) if f.endswith('.wav')][breakpoint::])
               srctrgts+=[(src,trgt) for _ in files]  
    return(srctrgts)

def simple_sd_test(sess, output_node, clf):
    global cnt
    bar = progressbar.ProgressBar(max_value=1800)
    adv = []
    cnt = 0
    for src in range(2,12):
        for trgt in range(2,12):
            case_dir = format("../success_output/result/%s/%s" %(key[trgt], key[src]))
            if os.path.exists(case_dir):
                num_files = len(os.listdir(case_dir))
                breakpoint = int(num_files/2)
                adv+=([format('%s/%s' %(case_dir, f)) for f in os.listdir(case_dir) if f.endswith('.wav')][breakpoint::])
    ben = []
    for src in range(2,12):
        case_dir = format("data/%s" %(key[src]))
        if os.path.exists(case_dir):
            ben+=([format('%s/%s' %(case_dir, f)) for f in os.listdir(case_dir) if f.endswith('.wav')][90:180])
    ac = 0; ai = 0
    for a in adv:
        score = simple_sd_value(sess, output_node, a, bar)
        if(clf.predict([[score]]) == [[1]]):
            ac+=1
        else:
            ai+=1
    bc = 0; bi = 0
    for b in ben:
        score = simple_sd_value(sess, output_node, b, bar)
        if(clf.predict([[score]]) == [[0]]):
            bc+=1
        else:
            bi+=1
    print("Tested adversarial examples:", len(adv))
    print("Tested benign examples:", len(ben))
    print("Total Accuracy: ", (ac+bc)/(ac+ai+bc+bi))
    p = ac/(ac+bi); r = ac/(ac+ai)
    f1 = 2*p*r/(p + r)
    print("Precision: ", p)
    print("Recall: ", r)
    print("F1 Score: ", f1)


def filt(sess, output_node):
    s = 0; r = 0
    for src in range(2,12):
        for trgt in range(2,12):
            case_dir = format("../success_output/result/%s/%s" %(key[trgt], key[src]))
            if os.path.exists(case_dir):
                for f in os.listdir(case_dir): 
                    if not f.endswith('.wav'):
                        continue
                    path = format('%s/%s' %(case_dir, f))
                    pred = pred_sd(sess, output_node, path)
                    if(pred == trgt):
                       s+=1
                    elif(pred == src):
                       os.system("rm %s" %(path))
                       r+=1
    print("Successful Adversarial Examples:", s)
    print("Removed", r, "adversarial examples")



def save_band_scores():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    load_graph('ckpts/conv_actions_frozen.pb')
    with tf.Session(config=config) as sess:
        output_node = sess.graph.get_tensor_by_name('labels_softmax:0')
        print("Training...") # 816 adversarial examples and 900 benign examples
        trainscores = band_sd_scores(sess, output_node, True)
        np.save("bandtrainscores.npy", trainscores) 
        print("Testing...") # 856 adversarial examples and 900 benign examples
        testscores = band_sd_scores(sess, output_node, False)
        np.save("bandtestscores.npy", testscores)
       # val1 = band_sd_value(sess, output_node, "../output/data/down/10627519_nohash_0.wav")
       # print("Benign Eps: ", val1) 
       # val2 = band_sd_value(sess, output_node, "../output/result/up/down/10627519_nohash_0.wav")
       # print("Adversarial Eps: ", val2)            

def simple_noise_flood_main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    load_graph('ckpts/conv_actions_frozen.pb')
    with tf.Session(config=config) as sess:
        output_node = sess.graph.get_tensor_by_name('labels_softmax:0')
        print("Training...") # 816 adversarial examples and 900 benign examples
        clf = simple_sd_train(sess, output_node)
        print("Testing...") # 856 adversarial examples and 900 benign examples
        simple_sd_test(sess, output_node, clf)

def save_simple_noise_flood_scores():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    load_graph('ckpts/conv_actions_frozen.pb')
    with tf.Session(config=config) as sess:
        output_node = sess.graph.get_tensor_by_name('labels_softmax:0')
        print("Training...") # 816 adversarial examples and 900 benign examples
        trainscores = simple_sd_scores(sess, output_node, True)
        np.save("simpletrainscores.npy", trainscores)
        print("Testing...") # 856 adversarial examples and 900 benign examples
        testscores = simple_sd_scores(sess, output_node, False)
        np.save("simpletestscores.npy", testscores)

def isolated_band_flood(band):
    clf = DecisionTreeClassifier(max_depth=1, criterion="entropy")
    scores = np.load("bandtrainscores.npy")
    scores = [[i[band]] for i in scores]
    labels = [1 for _ in range(816)] + [0 for _ in range(900)]
    clf = clf.fit(scores, np.asarray(labels))
    testscores = np.load("bandtestscores.npy")
    testscores = [[i[band]] for i in testscores]
    preds = clf.predict(testscores)
    ac = 0; ai = 0; bc = 0; bi = 0
    for a in preds[0:856]:
        if(a == 1):
            ac+=1
        else:
            ai+=1
    for b in preds[856::]:
        if(b == 0):
            bc+=1
        else:
            bi+=1
    print("Total Accuracy: ", (ac+bc)/(ac+ai+bc+bi))
    p = ac/(ac+bi); r = ac/(ac+ai)
    f1 = 2*p*r/(p + r)
    print("Precision: ", p)
    print("Recall: ", r)
    print("F1 Score: ", f1)
       
def test_ensemble_classification():
    clf = XGBClassifier()
    simplescores = np.load("simpletrainscores.npy")
    bandscores = np.load("bandtrainscores.npy")
    scores = [[s for s in simplescores[i]] + [b for b in bandscores[i]] for i in range(1716)] 
    labels = [1 for _ in range(816)] + [0 for _ in range(900)]
    clf = clf.fit(np.asarray(scores), np.asarray(labels))
    simplescores = np.load("simpletestscores.npy")
    bandscores = np.load("bandtestscores.npy")
    testscores = [[s for s in simplescores[i]] + [b for b in bandscores[i]] for i in range(1756)] 
    preds = clf.predict(np.asarray(testscores))
    srctrgts = index_to_src_trgt()
    recalls = np.asarray([[0 for _ in range(10)] for _ in range(10)])
    totals = np.asarray([[0 for _ in range(10)] for _ in range(10)])
    ac = 0; ai = 0; bc = 0; bi = 0
    for i in range(856):
        a = preds[i]
        if(a == 1):
            ac+=1
            recalls[srctrgts[i][0]-2][srctrgts[i][1]-2] += 1
        else:
            ai+=1
        totals[srctrgts[i][0]-2][srctrgts[i][1]-2] += 1
    for b in preds[856::]:
        if(b == 0):
            bc+=1
        else:
            bi+=1
    print("Total Accuracy: ", (ac+bc)/(ac+ai+bc+bi))
    p = ac/(ac+bi); r = ac/(ac+ai)
    f1 = 2*p*r/(p + r)
    print("Precision: ", p)
    print("Recall: ", r)
    print("F1 Score: ", f1)
    for i in range(10):
        totals[i][i]+=1
    return(recalls/totals)


def test_voting_ensembles():
    threshold = 4
    clfs = [DecisionTreeClassifier(max_depth=1, criterion="entropy") for _ in range(5)]
    simplescores = np.load("simpletrainscores.npy")
    bandscores = np.load("bandtrainscores.npy")
    labels = [1 for _ in range(816)] + [0 for _ in range(900)]
    clfs[0] = clfs[0].fit(np.asarray(simplescores), np.asarray(labels))
    for i in range(4):
        scores = [[j[i]] for j in bandscores]
        clfs[i+1] = clfs[i+1].fit(scores, np.asarray(labels))
    simplescores = np.load("simpletestscores.npy")
    bandscores = np.load("bandtestscores.npy")
    testscores = [[s for s in simplescores[i]] + [b for b in bandscores[i]] for i in range(1756)]
    ai = 0; ac = 0; bi = 0; bc = 0
    for i in range(1756):
        votes = 0
        for j in range(5):
            if(clfs[j].predict([[testscores[i][j]]]) == [[1]]):
                votes+=1
        if(votes >= threshold): # adversarial prediction
            if(i < 856):
                ac+=1
            else:
                bi+=1
        else:
            if(i < 856):
                ai+=1
            else:
                bc+=1
    print("Total Accuracy: ", (ac+bc)/(ac+ai+bc+bi))
    p = ac/(ac+bi); r = ac/(ac+ai)
    f1 = 2*p*r/(p + r)
    print("Precision: ", p)
    print("Recall: ", r)
    print("F1 Score: ", f1)
            
recalls = test_ensemble_classification()

