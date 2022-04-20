SAVE_MDETR_PREDICTIONS = False
USE_MDETR_PREDICTIONS_AS_GROUNDTRUTHS = False
MDETR_PREDICTION_PATH = 'processed_mdetr_predictions_training_set.csv'

REPLACE_ARM_WITH_EYE_TO_FINGERTIP = True
EYE_TO_FINGERTIP_ANNOTATION_TRAIN_PATH = 'yourefit/eye_to_fingertip/eye_to_fingertip_annotations_train.csv'
EYE_TO_FINGERTIP_ANNOTATION_VALID_PATH = 'yourefit/eye_to_fingertip/eye_to_fingertip_annotations_valid.csv'

ARM_LOSS_COEF = 3
ARM_SCORE_LOSS_COEF = 1.5
ARM_BOX_ALIGN_LOSS_COEF = 100
USE_GT__ARM_FOR_ARM_BOX_ALIGN_LOSS = True
ARM_BOX_ALIGN_OFFSET_BY_GT = True # if fasle, use ARM_BOX_ALIGH_FIXED_OFFSET
ARM_BOX_ALIGH_FIXED_OFFSET = 1

DEACTIVATE_EXTRA_TRANSFORMS = False

REPLACE_IMAGES_WITH_INPAINT = False
# Inpaint dir, relative to data_root
#inpaint_dir = 'inpaint'
#INPAINT_DIR = 'inpaint_Place'
INPAINT_DIR = 'inpaint_Place_using_expanded_masks'

SAVE_EVALUATION_PREDICTIONS = False
prediction_dir = 'predictions'
prediction_file_name = 'debug.csv'


EVAL_EARLY_STOP = False
PRINT_PREDICTIONS_AT_BREAKPOINT = False
EVAL_EARLY_STOP_COUNT = 30

CHECKPOINT_FREQUENCY = 3

PREDICT_POSE_USING_A_DIFFERENT_MODEL = False
POSE_MLP_NUM_LAYERS = 3

ARGS_POSE = None # will be updated by the program