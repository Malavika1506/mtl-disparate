
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
AFCK_LABELS =['mostly-correct', 'incorrect', 'understated', 'exaggerated', 'correct', 'misleading', 'unproven']
BOVE_LABELS =['rating: false', 'none']
CHCT_LABELS =['verdict: unsubstantiated', 'verdict: true', 'verdict: false', 'none']
CLCK_LABELS =['unsupported', 'misleading', 'incorrect']
FAAN_LABELS =['factscan score: misleading', 'factscan score: true', 'factscan score: false']
FALY_LABELS =['false', 'partly true', 'none', 'unverified', 'true']
FANI_LABELS =['conclusion: accurate', 'conclusion: false', 'conclusion: unclear']
FARG_LABELS =['false', 'pants on fire!', 'unsupported', 'distorts the facts', 'exaggerates', 'none', 'spins the facts', 'no evidence', 'misleading', 'not the whole story', 'cherry picks']
GOOP_LABELS =['0', '10', '3', '2', '4', '1']
HOER_LABELS =['fake news', 'unsubstantiated messages', 'true messages', 'bogus warning', 'misleading recommendations', 'facebook scams', 'statirical reports']
HUCA_LABELS =['some baloney', 'a little baloney', 'a lot of baloney']
MPWS_LABELS =['false', 'misleading', 'accurate']
OBRY_LABELS =['unobservable', 'verified', 'mostly_false', 'mostly_true']
PARA_LABELS =['false', 'half flip', 'pants on fire!', 'mostly true', 'half-true', 'mostly false', 'true']
PECK_LABELS =['false', 'partially true', 'true']
POMT_LABELS =['false', 'pants on fire!', 'half flip', 'mostly true', 'no flip', 'None', 'full flop', 'half-true', 'mostly false', 'true']
POSE_LABELS =['false', 'pants on fire!', 'mostly true', 'in the works', 'promise kept', 'compromise', 'promise broken', 'None', 'half-true', 'mostly false', 'true', 'not yet rated', 'stalled']
RANZ_LABELS =['fact', 'fiction']
SNES_LABELS =['false', 'miscaptioned', 'outdated', 'mostly true', 'legend', 'scam', 'mostly false', 'unproven', 'mixture', 'true', 'misattributed', 'correct attribution']
THAL_LABELS =['mostly true', 'half true', 'none', 'mostly false', 'unproven', 'we rate this claim false']
THET_LABELS =['false', 'mostly true', 'half true', 'none', 'mostly false', 'true']
TRON_LABELS =['unproven!', 'outdated!', 'fiction! & satire!', 'investigation pending!', 'grass roots movement!', 'misleading!', 'truth! & outdated!', 'truth! & disputed!', 'mostly fiction!', 'truth! & misleading!', 'none', 'truth!', 'fiction!', 'previously truth! now resolved!', 'disputed!', 'correct attribution!', 'commentary!', 'confirmed authorship!', 'mostly truth!', 'incorrect attribution!', 'authorship confirmed!', 'virus!', 'truth! & unproven!', 'inaccurate attribution!', 'opinion!', 'truth! & fiction!', 'scam!']
VEES_LABELS =['false', 'fake', 'misleading', 'none']
VOGO_LABELS =['determination: misleading', 'determination: true', 'none', 'determination: a stretch', 'determination: false', 'determination: mostly true', 'determination: huckster propaganda', 'determination: barely true']
WAST_LABELS =['false', 'pants on fire!', 'mostly true', 'in-between', '3 pinnochios', 'promise kept', '2 pinnochios', 'none', 'not the whole story', 'half-true', 'mostly false', '4 pinnochios', 'needs context', 'true', 'stalled']


SIM = 'similarity'
DIV = 'diversity'
NONE = 'predsonly'
SIMILARITY_FEATURES = ['jensen-shannon', 'renyi', 'cosine', 'euclidean',
                       'variational', 'bhattacharyya']
DIVERSITY_FEATURES = ['num_word_types', 'type_token_ratio', 'entropy',
                      'simpsons_index', 'renyi_entropy']
# we don't use 'quadratic_entropy' at the moment, as it requires word vectors
