import matplotlib.pyplot as plt
import cnn_model
import utils

saved_model_folder_name = input('Set saved model folder name: ')

if len(saved_model_folder_name) <= 0:    
    raise SystemExit('The given folder name is empty. Exiting...')

model = cnn_model.create_model()

cnn_model.restore_model(model, 'saved_models/%s/checkpoint' % (saved_model_folder_name))

test_images = utils.load_images_from_dir('dataset/test_images');

for image in test_images:        
    plt.imshow(image)
    plt.show()    
    
    result = model.predict(utils.prepare_image(image));    
    
    print('Dog: %f' % (result[0][0]))
    print('Cat: %f' % (1 - result[0][0]))