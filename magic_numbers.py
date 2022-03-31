SAVE_MDETR_PREDICTIONS = False
USE_MDETR_PREDICTIONS_AS_GROUNDTRUTHS = False
MDETR_PREDICTION_PATH = 'processed_mdetr_predictions_training_set.csv'

REPLACE_ARM_WITH_EYE_TO_FINGERTIP = True
EYE_TO_FINGERTIP_ANNOTATION_TRAIN_PATH = 'yourefit/eye_to_fingertip/eye_to_fingertip_annotations_train.csv'
EYE_TO_FINGERTIP_ANNOTATION_VALID_PATH = 'yourefit/eye_to_fingertip/eye_to_fingertip_annotations_valid.csv'

ARM_LOSS_COEF = 3
ARM_SCORE_LOSS_COEF = 1.5