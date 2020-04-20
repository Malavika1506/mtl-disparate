import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import tensorflow as tf
import numpy as np

from constants import ABBC,AFCK,BOVE,CHCT,CLCK,FAAN,FALY,FANI,FARG,GOOP,HOER,HUCA,MPWS,OBRY,\
    PARA,PECK,POMT,POSE,RANZ,SNES,THAL,THET,TRON,VEES,VOGO,WAST,\
        ABBC_LABELS,AFCK_LABELS,BOVE_LABELS,CHCT_LABELS,CLCK_LABELS,FAAN_LABELS,FALY_LABELS,FANI_LABELS,FARG_LABELS, \
        GOOP_LABELS,HOER_LABELS,HUCA_LABELS,MPWS_LABELS,OBRY_LABELS,PARA_LABELS,PECK_LABELS,POMT_LABELS,POSE_LABELS, \
            RANZ_LABELS,SNES_LABELS,THAL_LABELS,THET_LABELS,TRON_LABELS,VEES_LABELS,VOGO_LABELS,WAST_LABELS, TASKS,\
                SIM, DIV, RNN_CELL_TYPES

def task2labels(task):
    if task == ABBC:
        return ABBC_LABELS
    if task == AFCK:
        return AFCK_LABELS
    if task == BOVE:
        return BOVE_LABELS
    if task == CHCT:
        return CHCT_LABELS
    if task == CLCK:
        return CLCK_LABELS
    if task == FAAN:
        return FAAN_LABELS
    if task == FALY:
        return FALY_LABELS
    if task == FANI:
        return FANI_LABELS
    if task == FARG:
        return FARG_LABELS
    if task == GOOP:
        return GOOP_LABELS
    if task == HOER:
        return HOER_LABELS
    if task == HUCA:
        return HUCA_LABELS
    if task == MPWS:
        return MPWS_LABELS
    if task == OBRY:
        return OBRY_LABELS
    if task == PARA:
        return PARA_LABELS
    if task == PECK:
        return PECK_LABELS
    if task == POMT:
        return POMT_LABELS
    if task == POSE:
        return POSE_LABELS
    if task == RANZ:
        return RANZ_LABELS
    if task == SNES:
        return SNES_LABELS
    if task == THAL:
        return THAL_LABELS
    if task == THET:
        return THET_LABELS
    if task == TRON:
        return TRON_LABELS
    if task == VEES:
        return VEES_LABELS
    if task == VOGO:
        return VOGO_LABELS
    if task == WAST:
        return WAST_LABELS
    raise ValueError('No labels available for task %s.' % task)


def task2display_name(task):
    if task == ABBC:
        return "abbc"
    if task == AFCK:
        return "afck"
    if task == BOVE:
        return "bove"
    if task == CHCT:
        return "chct"
    if task == CLCK:
        return "clck"
    if task == FAAN:
        return "faan"
    if task == FALY:
        return "faly"
    if task == FANI:
        return "fani"
    if task == FARG:
        return "farg"
    if task == GOOP:
        return "goop"
    if task == HOER:
        return "hoer"
    if task == HUCA:
        return "huca"
    if task == MPWS:
        return "mpws"
    if task == OBRY:
        return "obry"
    if task == PARA:
        return "para"
    if task == PECK:
        return "peck"
    if task == POMT:
        return "pomt"
    if task == POSE:
        return "pose"
    if task == RANZ:
        return "ranz"
    if task == SNES:
        return "snes"
    if task == THAL:
        return "thal"
    if task == THET:
        return "thet"
    if task == TRON:
        return "tron"
    if task == VEES:
        return "vees"
    if task == VOGO:
        return "vogo"
    if task == WAST:
        return "wast"
    raise ValueError('%s is not a valid task.' % task)


def task2color(task):
    return 'midnightblue'
    #raise ValueError('%s is not available.' % task)


def label2display_name(label):
    
    return label


def plot_label_embeddings(sess, tasks, label_vocab):
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "label_embeddings/label_embeddings")
    assert len(var_list) > 0, 'Error: Label embeddings have not been saved.'
    assert len(var_list) == 1

    label_embeddings = sess.run(var_list[0])
    print('Loaded label embeddings of shape:', label_embeddings.shape)

    assert label_vocab is not None

    # remove the UNK label of the label embeddings
    label_embeddings = label_embeddings[1:, :]

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'olive', 'cyan', 'brown']

    # tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
    pca = PCA(n_components=2)

    label_embeddings_tsne = pca.fit_transform(label_embeddings)
    label_names = []
    task_names = []
    for i, task in enumerate(tasks):
        task_labels = task2labels(task)
        label_names += task_labels
        task_names += [task] * len(task_labels)
    # as a sanity check, make sure that the labels correspond with those in the
    # label vocab; +1 because the labels start at 1 (0 is UNK)
    for i in range(label_embeddings.shape[0]):
        label_id = "%s_%s" % (task_names[i], str(label_names[i]))
        # print(i+1, label_id, label_vocab.sym2id[label_id])
        assert i+1 == label_vocab.sym2id[label_id],\
            'Error: Id %d != label id %d for %s.' % (i+1, label_id, task_names[i])

    file_name = 'label_embeddings.png'
    plot_embedding(label_embeddings_tsne, label_names, task_names, file_name=file_name)


def plot_embedding(X, y, tasks, title=None, file_name=None):
    """Plot an embedding X with the label y colored by colors."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # we can increase the resolution by increasing the figure size
    plt.figure(figsize=(5,5))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        #if tasks[i] == STANCE:
            # skip stance and plot later
            #continue
        plt.text(X[i, 0], X[i, 1], label2display_name(str(y[i])),
                 color=task2color(tasks[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    '''
    for i in range(X.shape[0]):
        if tasks[i] == STANCE:
            plt.text(X[i, 0], X[i, 1], label2display_name(str(y[i])),
                     color=task2color(tasks[i]),
                     fontdict={'weight': 'bold', 'size': 9})
    '''
    
    # create patches for the legend
    patches = []
    for task in sorted(list(set(tasks))):
        patches.append(mpatches.Patch(color=task2color(task), label=task2display_name(task)))
    lgd = plt.legend(handles=patches, loc='upper left', bbox_to_anchor=(1, 1),
                     edgecolor='black')

    # plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    # plt.show()
    plt.savefig(file_name, bbox_extra_artists=(lgd,), bbox_inches='tight')
