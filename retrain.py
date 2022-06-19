import os
import shutil
import logging
from dl_retrain import RetrainModel
from dl_evaluate import EvaluateModel

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(filename='logs/dl_retrain.log', filemode='w', format='%(asctime)s - %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S')

logging.warning("<<< Starting >>>")

if os.path.exists("tmp"):
    shutil.rmtree('tmp', ignore_errors=True)
    logging.warning('tmp folder deleted')

os.mkdir("tmp")
os.mkdir("tmp/datasets")
os.mkdir("tmp/model")

retrain = RetrainModel()
retrain.fetch_tr_data() # fetch training data
retrain.fetch_val_data() # fetch valiadtion data
retrain.fetch_te_data() # fetch testing data
retrain.train()

dl_dir = './backups/dl'
next_num = len([name for name in os.listdir(dl_dir) if os.path.isfile(os.path.join(dl_dir, name))])

source = "./tmp/model/moraphishdet.h5"
destination = "."
backup_destination = "./backups/dl/moraphishdet" + str(next_num) + ".h5"

evaluate = EvaluateModel()
acc_now, acc_new, loss_now, loss_new = evaluate.get_accuracy()

if (acc_new > acc_now):
    shutil.copy("./moraphishdet.h5", backup_destination)
    shutil.copy(source, destination)
    logging.warning("The new model accuracy has been improved. Copied to the main location.")

elif (acc_new == acc_now):
    if (loss_new < loss_now):
        shutil.copy("./moraphishdet.h5", backup_destination)
        shutil.copy(source, destination)
        logging.warning("The new model accuracy is same as the current model. However, the new model loss is less compared to the current model. Copied to the main location.")
    else:
        logging.warning("The new model accuracy is same as the current model and the new model loss is high compared to the current model. Nothing copied.")
else:
    logging.warning("The new model accuracy is less compared to the current model. Nothing copied.")
