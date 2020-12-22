from src.preproc_image import get_train_image, get_test_image, normalize_image
from src.bayesian_model import bayesian
from src.train_and_evalutate import make_model, make_graph
import time
'''
1. GLCM numeric feature 및 GLCM 특징을 이용하여 Bayesian 분류기 개발
2. GLCM numeric feature 및 GLCM 특징을 이용하여 MLP 분류기 개발
3. raw image data를 입력으로 MLP를 이용한 텍스처 분류기 개발
4. raw image data를 입력으로 CNN을 이용한 텍스처 분류기 개발
'''

from torchvision.datasets import CIFAR10
from torchvision import transforms

if __name__ == "__main__":
    make_bayesian = False
    test = False
    model = 4
    
    if test:
        train_dir = './texture_data/train'
        test_dir = './texture_data/test'
        classes = ['brick', 'grass', 'ground']
    else:
        train_dir = './archive/seg_train/seg_train'
        test_dir = './archive/seg_test/seg_test'
        classes = ['buildings', 'forest', 'mountain', 'sea']
        # classes = ['buildings']

    start_time = time.time()

    if make_bayesian:
        X_train, Y_train = get_train_image(train_dir, classes, using_glcm=True)
        X_test, Y_test = get_test_image(test_dir, classes, using_glcm=True)

        bayesian(X_train, Y_train, X_test, Y_test, classes)
    elif model == 1:
        X_train, Y_train = get_train_image(train_dir, classes, PATCH_SIZE=30, using_glcm=True)
        X_test, Y_test = get_test_image(test_dir, classes, using_glcm=True)

        train_losses, test_losses, train_accs, test_accs = make_model(X_train, Y_train, X_test, Y_test, 300, 1, PATCH_SIZE=30)
        make_graph(train_losses, test_losses, train_accs, test_accs, 1)

    elif model == 2:
        X_train, Y_train = get_train_image(train_dir, classes, PATCH_SIZE=None, using_glcm=False)
        X_test, Y_test = get_test_image(test_dir, classes, using_glcm=False)
        X_train, Y_train, X_test, Y_test = normalize_image(X_train, Y_train, X_test, Y_test)

        train_losses, test_losses, train_accs, test_accs = make_model(X_train, Y_train, X_test, Y_test, 100, 2)
        make_graph(train_losses, test_losses, train_accs, test_accs, 2)
    
    elif model == 3:
        X_train, Y_train = get_train_image(train_dir, classes, PATCH_SIZE=None, using_glcm=False)
        X_test, Y_test = get_test_image(test_dir, classes, using_glcm=False)
        X_train, Y_train, X_test, Y_test = normalize_image(X_train, Y_train, X_test, Y_test)

        train_losses, test_losses, train_accs, test_accs = make_model(X_train, Y_train, X_test, Y_test, 100, 3)
        make_graph(train_losses, test_losses, train_accs, test_accs, 3)

    elif model == 4:
        X_train, Y_train = get_train_image(train_dir, classes, PATCH_SIZE=None, using_glcm=False)
        X_test, Y_test = get_test_image(test_dir, classes, using_glcm=False)
        # X_train, Y_train, X_test, Y_test = normalize_image(X_train, Y_train, X_test, Y_test)

        make_graph(train_losses, test_losses, train_accs, test_accs, 4)

    end_time = time.time()

    print("elapsed time = ", end_time - start_time)
    