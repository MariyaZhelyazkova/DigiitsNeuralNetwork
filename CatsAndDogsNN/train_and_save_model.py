import matplotlib.pyplot as plt
import cnn_model
import utils

model = cnn_model.create_model()

cnn_model.train_model(model, 'dataset/training_set', 'dataset/test_set')
cnn_model.save_model(model, 'saved_models/%s/checkpoint' % (utils.get_current_date_time()))

labels = cnn_model.get_labels('dataset/training_set')
test_images = utils.load_images_from_dir('dataset/test_images')

for image in test_images:        
    plt.imshow(image)
    plt.show()    
    
    result = model.predict(utils.prepare_image(image))   
    
    for i, label in enumerate(labels):
        print('Species type %s - possibility = %f' % (label, result[0][i]))        