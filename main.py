from train_model import main as train_model
from generate import main as generate

import sys




if __name__ == "__main__":
    try:
        if sys.argv[1] == 'train':
            print('training...')
            train_model()
            print('generating...')
            generate()
    except Exception as e:
        print('only generating...')
        generate()
