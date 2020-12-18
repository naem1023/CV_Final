from src.preproc_image import get_train_image, get_test_image
from src.bayesian_model import bayesian
from src.train_and_evalutate import make_model, make_graph
'''
1. GLCM numeric feature 및 GLCM 특징을 이용하여 Bayesian 분류기 개발
2. GLCM numeric feature 및 GLCM 특징을 이용하여 MLP 분류기 개발
3. raw image data를 입력으로 MLP를 이용한 텍스처 분류기 개발
4. raw image data를 입력으로 CNN을 이용한 텍스처 분류기 개발
'''

if __name__ == "__main__":
    make_bayesian = False
    test = True
    model = 2

    if test:
        train_dir = './texture_data/train'
        test_dir = './texture_data/test'
        classes = ['brick', 'grass', 'ground']
    else:
        train_dir = './texture_data/train'
        test_dir = './texuture_data/test'
        classes = ['brick', 'grass', 'ground']

    if model == 1:
        X_train, Y_train = get_train_image(train_dir, classes, PATCH_SIZE=30, using_glcm=True)
        X_test, Y_test = get_test_image(test_dir, classes, using_glcm=True)

        train_losses, test_losses, train_accs, test_accs = make_model(X_train, Y_train, X_test, Y_test, 500, 1)
        make_graph(train_losses, test_losses, train_accs, test_accs, 1)

    if model == 2:
        X_train, Y_train = get_train_image(train_dir, classes, PATCH_SIZE=32, using_glcm=False)
        X_test, Y_test = get_test_image(test_dir, classes, using_glcm=True)

        train_losses, test_losses, train_accs, test_accs = make_model(X_train, Y_train, X_test, Y_test, 500, 2)
        make_graph(train_losses, test_losses, train_accs, test_accs, 1)

    elif make_bayesian:
        X_train, Y_train = get_train_image(train_dir, classes, using_glcm=True)
        X_test, Y_test = get_test_image(test_dir, classes, using_glcm=True)

        bayesian(X_train, Y_train, X_test, Y_test, classes)
    