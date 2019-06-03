import matplotlib.pyplot as plt
import cnn_model
import utils

model = cnn_model.create_model()

cnn_model.restore_model(model, 'saved_models/checkpoint1')

test_images = utils.load_images_from_dir('dataset/test_images');

for image in test_images:        
    plt.imshow(image)
    plt.show()    
    
    result = model.predict(utils.prepare_image(image));    
    
    print('Dog: %f' % (result[0][0]))
    print('Cat: %f' % (1 - result[0][0]))