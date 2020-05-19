import sys
import tensorflow as tf
import os
import numpy as np
import argparse
import copy
from sklearn.metrics import classification_report
from mtl.tensoriser import load_data
from mtl.training import train, restore_trained_model, get_preds_for_ltn
from preproc.log_utils import log_results, task2score1, task2score2

from constants import ABBC,AFCK,BOVE,CHCT,CLCK,FAAN,FALY,FANI,FARG,GOOP,HOER,HUCA,MPWS,OBRY,\
    PARA,PECK,POMT,POSE,RANZ,SNES,THAL,THET,TRON,VEES,VOGO,WAST,\
        ABBC_LABELS,AFCK_LABELS,BOVE_LABELS,CHCT_LABELS,CLCK_LABELS,FAAN_LABELS,FALY_LABELS,FANI_LABELS,FARG_LABELS, \
        GOOP_LABELS,HOER_LABELS,HUCA_LABELS,MPWS_LABELS,OBRY_LABELS,PARA_LABELS,PECK_LABELS,POMT_LABELS,POSE_LABELS, \
            RANZ_LABELS,SNES_LABELS,THAL_LABELS,THET_LABELS,TRON_LABELS,VEES_LABELS,VOGO_LABELS,WAST_LABELS, TASKS,\
                SIM, DIV, RNN_CELL_TYPES

from preproc.plot_utils import plot_label_embeddings


def main(**options):

    # create the log directory if it does not exist
    log_dir = os.path.dirname(args.log_file)
    if not os.path.exists(log_dir):
        print('Creating %s...' % log_dir)
        os.makedirs(log_dir)

    train_feed_dicts, dev_feed_dicts, test_feed_dicts, vocab, label_vocab, ltn_sizes, label_to_labelvocab = load_data(placeholders, target_labels, target_sizes, **options)

    # remove tasks from target_sizes if not used
    for task in copy.deepcopy(set(target_sizes.keys())):
        if not task in options["tasks"]:
            target_sizes.pop(task)

    print("Data loaded and tensorised. Training model with settings: " + str(options))

    if options['model_type'] != 'hard-sharing' and options["feature_sets"] != "predsonly":
        ex1 = train_feed_dicts[options["main_task"]][0]
        ex1feats = ex1[placeholders["features"]]
        input_size_preds = len(ex1feats[0])
    else:
        input_size_preds = 0

    if label_vocab == None:
        label_vocab_len = 0
    else:
        label_vocab_len = len(label_vocab)

    # Do not take up all the GPU memory all the time.
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    #print('***************** Start of tf.session ***************************')
    with tf.Session(config=sess_config) as sess:
        print(options["plot_embeddings"])
        if options["plot_embeddings"]:
            print('Loading the model for plotting label embeddings...')
            _, _, _, _, _, _, _ = restore_trained_model(
                placeholders, target_sizes, train_feed_dicts, vocab,
                label_vocab_len, label_to_labelvocab, input_size_preds, ltn_sizes, sess=sess,
                **options)
            plot_label_embeddings(sess, args.tasks, label_vocab)
            sys.exit(0)
        elif options["apply_existing_model"] == False:
            #print('***************** Start of train ***************************')
            logits, loss, preds, logits_dict_ltn, loss_dict_ltn, preds_dict_ltn, predict_main_dict = train(placeholders, target_sizes, train_feed_dicts, dev_feed_dicts, vocab, label_vocab, input_size_preds, ltn_sizes, label_to_labelvocab, sess=sess, **options)
            #print('***************** End of train ***************************')
        else:
            #print('***************** Start of restore_trained_model ***************************')
            logits, loss, preds, logits_dict_ltn, loss_dict_ltn, preds_dict_ltn, predict_main_dict = restore_trained_model(placeholders, target_sizes, train_feed_dicts, vocab, label_vocab_len, label_to_labelvocab, input_size_preds, ltn_sizes, sess=sess, **options)
            #print('***************** End of restore_trained_model ***************************')
        print('============')
        # Test on test data
        for task in target_sizes.keys():
            correct_test_all, total_test, correct_test_all_ltn, total_test_ltn = 0.0, 0.0, 0.0, 0.0
            p_inds, g_inds, p_inds_ltn, topics = [], [], [], []
            for j, batch in enumerate(test_feed_dicts[task]):
                p = sess.run(preds[task], feed_dict=batch)
                pred_inds = [np.argmax(pp) for pp in p]
                p_inds.extend(pred_inds)
                gold_inds = [np.argmax(batch[placeholders["targets"]][i]) for i, targ in enumerate(batch[placeholders["targets"]])]
                g_inds.extend(gold_inds)
                hits = [pp for i, pp in enumerate(p) if np.argmax(pp) == np.argmax(batch[placeholders["targets"]][i])]
                correct_test_all += len(hits)
                total_test += len(batch[placeholders["targets"]])

                # keep track of the targets for topic-based scores
                topics += [t for t in batch[placeholders["seq1"]]]

                if options["model_type"] == "semi-supervised" or options["model_type"] == "label-transfer":
                    #print('***************** Start of get_preds_for_ltn ***************************')
                    batch_test = get_preds_for_ltn(sess, batch, placeholders, target_sizes, task, options["main_task"], preds,
                                                   options["ltn_pred_type"], label_to_labelvocab, options["lab_emb_dim"], options["model_type"])

                    p_ltn = sess.run(preds_dict_ltn[task], feed_dict=batch_test)
                    pred_inds_ltn = [np.argmax(pp_dev) for pp_dev in p_ltn]
                    p_inds_ltn.extend(pred_inds_ltn)

                    hits_ltn = [pp for i, pp in enumerate(p_ltn) if np.argmax(pp) == np.argmax(batch_test[placeholders["targets"]][i])]
                    correct_test_all_ltn += len(hits_ltn)
                    total_test_ltn += len(batch_test[placeholders["targets"]])

            with open(options['log_file'].replace('.txt', '_inds.txt'), 'a') as f:
                f.write(task + "\tMain model\t" + str(p_inds).replace("[", "").replace("]", "").replace(",", "") + "\n")
                f.write(task + "\tRelabel model\t" + str(p_inds_ltn).replace("[", "").replace("]", "").replace(",", "") + "\n")
                f.write(task + "\tGold\t" + str(g_inds).replace("[", "").replace("]", "").replace(",", "") + "\n")

            
            acc_test = correct_test_all/total_test
            #print('Test performance :', "Task: " + task, "Acc: ", acc_test)
            
            macro_score = task2score1(task, g_inds, p_inds, topics)
            micro_score = task2score2(task, g_inds, p_inds, topics)

            #print('macro_score:', macro_score)
            #print('micro_score:', micro_score)

            #print(target_labels[task])
            #print(classification_report(g_inds, p_inds))
            
            #log_results(options, micro_score, macro_score, acc_test, task)

            if options["model_type"] == "semi-supervised" or options["model_type"] == "label-transfer":
                acc_test_ltn = correct_test_all_ltn / total_test_ltn
                macro_score_ltn = task2score1(task, g_inds, p_inds_ltn, topics)
                micro_score_ltn = task2score2(task, g_inds, p_inds_ltn, topics)
                #task_score_ltn = task2score1(task, g_inds, p_inds_ltn, topics)
                #print('Test performance LTN:', "Task: " + task, "Acc: ", acc_test_ltn)
                #print('macro_score LTN:', macro_score_ltn)
                #print('micro_score LTN:', micro_score_ltn)
                #print(target_labels[task])
                #print(classification_report(g_inds, p_inds_ltn))
                log_results(options, micro_score_ltn, macro_score_ltn, acc_test_ltn, task)
            else:
                log_results(options, micro_score, macro_score, acc_test, task)
        
        
                
for i in TASKS:

    tf.reset_default_graph()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    seq1 = tf.placeholder(tf.int32, [None, None], name="seq1")
    seq1_lengths = tf.placeholder(tf.int32, [None], name="seq1_lengths")
    seq2 = tf.placeholder(tf.int32, [None, None], name="seq2")
    seq2_lengths = tf.placeholder(tf.int32, [None], name="seq2_lengths")
    seq3 = tf.placeholder(tf.int32, [None, None], name="seq3")
    seq3_lengths = tf.placeholder(tf.int32, [None], name="seq3_lengths")
    seq4 = tf.placeholder(tf.int32, [None, None], name="seq4")
    seq4_lengths = tf.placeholder(tf.int32, [None], name="seq4_lengths")
    seq5 = tf.placeholder(tf.int32, [None, None], name="seq5")
    seq5_lengths = tf.placeholder(tf.int32, [None], name="seq5_lengths")
    targets = tf.placeholder(tf.int32, [None, None], name="targets")
    targets_main = tf.placeholder(tf.int32, [None, None], name="targets_main")  # targets for main task
    features = tf.placeholder(tf.float32, [None, None], name="features")
    preds_for_ltn = tf.placeholder(tf.float32, [None, None], name="preds_for_ltn") # this is set to 0 initially and constantly updated during training
    label_vocab_inds = tf.placeholder(tf.int32, [None, None], name="label_vocab_inds")
    label_vocab_inds_main = tf.placeholder(tf.int32, [None, None], name="label_vocab_inds_main")  # label target for main task


    # This dictionary determines which tasks are used. By default, it contains
    # all existing tasks and is then modified during setup accordingly.

    target_sizes = {ABBC:3,AFCK:7,BOVE:2,CHCT:4,CLCK:3,FAAN:3,FALY:5,FANI:3,FARG:11,GOOP:6,HOER:7,HUCA:3,MPWS:3,OBRY:4,PARA:7,PECK:3,POMT:10,POSE:13,RANZ:2,SNES:12,THAL:6,THET:6,TRON:27,VEES:4,VOGO:8,WAST:15}

    target_labels = {ABBC:[],AFCK:[],BOVE:[],CHCT:[],CLCK:[],FAAN:[],FALY:[],FANI:[],FARG:[],GOOP:[],HOER:[],HUCA:[], MPWS:[],OBRY:[],PARA:[],PECK:[],POMT:[],POSE:[],RANZ:[],SNES:[],THAL:[],THET:[],TRON:[],VEES:[],VOGO:[],WAST:[]}

    placeholders = {"seq1": seq1, "seq1_lengths": seq1_lengths, "seq2": seq2,
                    "seq2_lengths": seq2_lengths, "seq3": seq3,
                    "seq3_lengths": seq3_lengths, "seq4": seq4,
                    "seq4_lengths": seq4_lengths, "seq5": seq5,
                    "seq5_lengths": seq5_lengths, "targets": targets, "targets_main": targets_main,
                    "features": features, "preds_for_ltn": preds_for_ltn,
                    "label_vocab_inds": label_vocab_inds, "label_vocab_inds_main": label_vocab_inds_main}


    if __name__ == "__main__":

        #i = TASKS[8]

        

        parser = argparse.ArgumentParser(description='Train and Evaluate a MTL model with incompatible outputs')
        parser.add_argument('--debug', default=False, action='store_true', help="Debug mode -- for this, only a small portion of the data is used to test code functionality")
        parser.add_argument('--dev_res_during_training', default=False, action='store_true', help="If true, computes results on dev set during training")
        parser.add_argument('--num_instances', type=int, default=128, help="What is the maximum number of instances to use per task")
        parser.add_argument('--apply_existing_model', default=False, action='store_true', help="If set to True, doesn't train model but only applies trained model to test data")
        parser.add_argument('--tasks', nargs='+', default=TASKS, help="Tasks to train on. If this is the same as the main task, a single-task model is trained. Options:" + str(TASKS))
        parser.add_argument('--main_task', type=str, default=i, help="The main task.")
        parser.add_argument('--feature_sets', nargs='+', help='data feature sets. In the paper, only diversity features are tested.', default=DIV)
        parser.add_argument('--ltn_pred_type', type=str, help='Whether to use hard or soft predictions as input to LTN model. In the experiments described in the paper, only soft predictions are used.', default='soft')
        parser.add_argument('--main_num_layers', type=int, help='If > 1, number of hidden layer for main model.', default=2)
        parser.add_argument('--lel_hid_size', type=int, help='If > 0, size of hidden layer for label embedding layer, as described in Section 3.2 of the paper.', default=2)
        parser.add_argument('--model_type', default='label-transfer', choices={'hard-sharing', 'label-transfer', 'semi-supervised'}, help="What model variant to use: "
                                                                                            "'hard-sharing' is the MTL with hard parameter sharing model (Section 3.1), "
                                                                                            "'label-transfer' is the label transfer network (Section 3.3), "
                                                                                            "'semi-supervised' is the semi-supervised MTL (Section 3.4)")
        parser.add_argument('--relabel_with_ltn', default=False, action='store_true', help="Only relevant for semi-supervised model: do we actually use it to relabel data or not. The latter can be used for debugging purposes. "
                                                                                        "Otherwise, this is the semi-supervised variant of the LTN described in Section 3.4 of the paper")
        parser.add_argument('--task_specific_layer_size', type=int, default=2, help="If >0, adds a task-specific hidden layer with that size and skip-connections")
        parser.add_argument('--batch_size', type=int, default=32, help="What batch size should be used")
        parser.add_argument('--max_epochs', type=int, default=1, help="What is the maximum number of epochs to train main model for")
        parser.add_argument('--max_epochs_ltn', type=int, default=2, help="What is the maximum number of epochs to train LTN model for")
        parser.add_argument('--max_epochs_after_ltn', type=int, default=0, help="After we've trained the relabelling function, how many epochs should we train for with augmented data.")
        parser.add_argument('--early_stopping', type=float, default=3.0, help="Threshold for early stopping on dev set of main task. If 1.0, there is no early stopping.")
        parser.add_argument('--emb_dim', type=int, default=128, help="What embedding size should be used")
        parser.add_argument('--lab_emb_dim', type=int, default=16, help='What embedding size should be used for the label embeddings. If 0, no label embeddings are used.')
        parser.add_argument('--lab_embs_for_ltn', default=False, action='store_true', help='Whether to use label embeddings for relabelling function or not.')
        parser.add_argument('--skip_connections', default=False, action='store_true', help='Skip connections for the RNN or not')
        parser.add_argument('--learning_rate', type=float, default=0.01, help="What initial learning rate should be used")
        parser.add_argument('--dropout_rate', type=float, default=1.0, help="What rate of dropout should be used. 1.0 -> no dropout")
        parser.add_argument('--l1_rate_main', type=float, default=1.0, help="What rate of l1 regularisation should be used for main model. 1.0 -> no l1")
        parser.add_argument('--l2_rate_main', type=float, default=1.0, help="What rate of l2 regularisation should be used for main model. 1.0 -> no l2")
        parser.add_argument('--l1_rate_ltn', type=float, default=1.0, help="What rate of l1 regularisation should be used for em model. 1.0 -> no l1")
        parser.add_argument('--l2_rate_ltn', type=float, default=1.0, help="What rate of l2 regularisation should be used for em model. 1.0 -> no l2")
        parser.add_argument('--rnn_cell_type', type=str, help='RNN cell type. Options:' + str(RNN_CELL_TYPES), default="lstm")
        parser.add_argument('--attention', default=False, action='store_true', help='Word by word attention mechanism')
        parser.add_argument('--save_model', default=False
        , action='store_true', help="Save model after end of training")
        parser.add_argument('--exp_id', type=str, default="run1", help="Experiment ID. In case the same experiment with the same configurations needs to be run more than once.")
        parser.add_argument('--features-path', type=str, default='./results/meta_claim_rvi/'+i+'/saved_features_new', help='the directory where the computed features are saved')
        parser.add_argument('--log_file', type=str, default="./results/meta_claim_evi/"+i+"/log.txt", help='the path to which results should be logged')
        parser.add_argument('--alternate_batches', default=True, action='store_true', help='alternate tasks between batches instead of between epochs during training') 
        parser.add_argument('--plot_embeddings', action='store_true', help='plot label embeddings of trained model')


        args = parser.parse_args()
        if args.debug:
            print('Debugging is switched on. Only a small portion of data is used.')
        if args.apply_existing_model:
            args.save_model = False
        if args.alternate_batches:
            print('Alternating tasks between batches...')
        else:
            print('Alternating tasks between epochs...')
        if args.feature_sets == 'predsonly' and args.model_type == 'semi-supervised':
            print("The model type 'label-transfer' needs to be used for this to work. Changing it to that setting.")
            args.model_type = 'label-transfer'
        main(**vars(args))
