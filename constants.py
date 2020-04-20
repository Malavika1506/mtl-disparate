"""
Constants shared across modules.
"""


ABBC= "abbc"
AFCK= "afck"
BOVE= "bove"
CHCT= "chct"
CLCK= "clck"
FAAN= "faan"
FALY= "faly"
FANI= "fani"
FARG= "farg"
GOOP= "goop" 
HOER= "hoer"
HUCA= "huca" 
MPWS= "mpws" 
OBRY= "obry"
PARA= "para"
PECK= "peck"
POMT= "pomt"
POSE= "pose" 
RANZ= "ranz" 
SNES= "snes" 
THAL= "thal" 
THET= "thet"
TRON= "tron" 
VEES= "vees" 
VOGO= "vogo" 
WAST= "wast"


TASK_NAMES_SHORT = {"abbc":	"ABBC", "afck":	"AFCK", "bove":	"BOVE",  "chct":"CHCT" , "clck":"CLCK", "faan":	"FAAN", "faly":	"FALY", "fani":"FANI", "farg":	"FARG", "goop":	"GOOP", "hoer":	"HOER", "huca":	"HUCA", "mpws":	"MPWS", "obry":	"OBRY", "para":	"PARA", "peck":	"PECK", "pomt":	"POMT", "pose":	"POSE", "ranz":	"RANZ", "snes":	"SNES", "thal":	"THAL", "thet":	"THET", "tron":	"TRON", "vees":	"VEES", "vogo":	"VOGO", "wast": "WAST"}


RNN_CELL_TYPES = ["lstm", "phased_lstm", "layer_norm", "nas"]  # LSTM, plus the RNN cell types in Tensorflow interchangable with it


TASKS = [ABBC, AFCK,BOVE,CHCT,CLCK,FAAN,FALY,FANI,FARG,GOOP,HOER,HUCA,MPWS,OBRY,PARA,PECK,POMT,POSE,RANZ,SNES,THAL,THET,TRON,VEES,VOGO,WAST]

ABBC_LABELS =['in-the-green', 'in-the-red', 'in-between']
AFCK_LABELS =['incorrect', 'understated', 'exaggerated', 'misleading', 'correct', 'unproven', 'mostly-correct']
BOVE_LABELS =['none', 'rating: false']
CHCT_LABELS =['none', 'verdict: true', 'verdict: unsubstantiated', 'verdict: false']
CLCK_LABELS =['incorrect', 'unsupported', 'misleading']
FAAN_LABELS =['factscan score: misleading', 'factscan score: false', 'factscan score: true']
FALY_LABELS =['partly true', 'true', 'none', 'unverified', 'false']
FANI_LABELS =['conclusion: false', 'conclusion: unclear', 'conclusion: accurate']
FARG_LABELS =['spins the facts', 'unsupported', 'no evidence', 'cherry picks', 'misleading', 'distorts the facts', 'none', 'not the whole story', 'false', 'pants on fire!', 'exaggerates']
GOOP_LABELS =['1', '4', '2', '3', '0', '10']
HOER_LABELS =['fake news', 'statirical reports', 'true messages', 'misleading recommendations', 'facebook scams', 'unsubstantiated messages', 'bogus warning']
HUCA_LABELS =['a lot of baloney', 'some baloney', 'a little baloney']
MPWS_LABELS =['accurate', 'false', 'misleading']
OBRY_LABELS =['verified', 'mostly_false', 'unobservable', 'mostly_true']
PARA_LABELS =['half-true', 'true', 'mostly false', 'pants on fire!', 'mostly true', 'false', 'half flip']
PECK_LABELS =['false', 'partially true', 'true']
POMT_LABELS =['half-true', 'no flip', 'true', 'mostly false', 'None', 'full flop', 'mostly true', 'false', 'pants on fire!', 'half flip']
POSE_LABELS =['promise kept', 'half-true', 'true', 'compromise', 'stalled', 'mostly false', 'promise broken', 'in the works', 'not yet rated', 'mostly true', 'false', 'pants on fire!', 'None']
RANZ_LABELS =['fiction', 'fact']
SNES_LABELS =['mixture', 'misattributed', 'outdated', 'miscaptioned', 'true', 'correct attribution', 'mostly false', 'scam', 'mostly true', 'false', 'legend', 'unproven']
THAL_LABELS =['half true', 'we rate this claim false', 'none', 'mostly false', 'mostly true', 'unproven']
THET_LABELS =['half true', 'true', 'none', 'mostly false', 'mostly true', 'false']
TRON_LABELS =['scam!', 'truth! & misleading!', 'correct attribution!', 'truth! & unproven!', 'outdated!', 'mostly fiction!', 'truth! & disputed!', 'truth!', 'truth! & outdated!', 'investigation pending!', 'fiction! & satire!', 'inaccurate attribution!', 'disputed!', 'incorrect attribution!', 'previously truth! now resolved!', 'unproven!', 'misleading!', 'authorship confirmed!', 'mostly truth!', 'confirmed authorship!', 'none', 'fiction!', 'truth! & fiction!', 'opinion!', 'virus!', 'commentary!', 'grass roots movement!']
VEES_LABELS =['none', 'false', 'misleading', 'fake']
VOGO_LABELS =['determination: huckster propaganda', 'determination: a stretch', 'none', 'determination: true', 'determination: barely true', 'determination: mostly true', 'determination: misleading', 'determination: false']
WAST_LABELS =['half-true', 'promise kept', 'true', '4 pinnochios', 'none', 'mostly false', 'stalled', '2 pinnochios', 'in-between', 'mostly true', '3 pinnochios', 'false', 'pants on fire!', 'not the whole story', 'needs context']


SIM = 'similarity'
DIV = 'diversity'
NONE = 'predsonly'
SIMILARITY_FEATURES = ['jensen-shannon', 'renyi', 'cosine', 'euclidean',
                       'variational', 'bhattacharyya']
DIVERSITY_FEATURES = ['num_word_types', 'type_token_ratio', 'entropy',
                      'simpsons_index', 'renyi_entropy']
# we don't use 'quadratic_entropy' at the moment, as it requires word vectors
