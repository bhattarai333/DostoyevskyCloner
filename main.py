from train_model import main as train_model
from generate import main as generate

import sys




if __name__ == "__main__":
    try:
        if sys.argv[1] == 'train':
            train_model()
    except:
        generate()
