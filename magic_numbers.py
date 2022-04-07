SAVE_MDETR_PREDICTIONS = False
USE_MDETR_PREDICTIONS_AS_GROUNDTRUTHS = False
MDETR_PREDICTION_PATH = 'processed_mdetr_predictions_training_set.csv'

REPLACE_ARM_WITH_EYE_TO_FINGERTIP = True
EYE_TO_FINGERTIP_ANNOTATION_TRAIN_PATH = 'yourefit/eye_to_fingertip/eye_to_fingertip_annotations_train.csv'
EYE_TO_FINGERTIP_ANNOTATION_VALID_PATH = 'yourefit/eye_to_fingertip/eye_to_fingertip_annotations_valid.csv'

ARM_LOSS_COEF = 3
ARM_SCORE_LOSS_COEF = 1.5

DEACTIVATE_EXTRA_TRANSFORMS = False

REPLACE_IMAGES_WITH_INPAINT = False
# Inpaint dir, relative to data_root
#inpaint_dir = 'inpaint'
#INPAINT_DIR = 'inpaint_Place'
INPAINT_DIR = 'inpaint_Place_using_expanded_masks'