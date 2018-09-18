# Create a generator for prediction
validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(image_size, image_size),
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=False)
 
# Get the filenames from the generator
fnames = validation_generator.filenames
 
# Get the ground truth from generator
ground_truth = validation_generator.classes
 
# Get the label to class mapping from the generator
label2index = validation_generator.class_indices
 
# Getting the mapping from class index to class label
idx2label = dict((v,k) for k,v in label2index.items())
 
# Get the predictions from the model using the generator
predictions = model.predict_generator(validation_generator, steps=validation_generator.samples/validation_generator.batch_size,verbose=1)
predicted_classes = np.argmax(predictions,axis=1)
 
errors = np.where(predicted_classes != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors),validation_generator.samples))
 
# Show the errors
for i in range(len(errors)):
    pred_class = np.argmax(predictions[errors[i]])
    pred_label = idx2label[pred_class]
     
    title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
        fnames[errors[i]].split('/')[0],
        pred_label,
        predictions[errors[i]][pred_class])
     
    original = load_img('{}/{}'.format(validation_dir,fnames[errors[i]]))
    plt.figure(figsize=[7,7])
    plt.axis('off')
    plt.title(title)
    plt.imshow(original)
    plt.show()












 fpath = 'loss-' + loss_function + '-' + str(num_classes)
    # datagen.fit(X_train)
    
    # model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size, shuffle=True, save_to_dir='./datagen/', save_prefix='datagen-',save_format='png'), # To save the images created by the generator
                # samples_per_epoch=num_samples, nb_epoch=nb_epoch,
                # verbose=1, validation_data=(X_test,Y_test),
                # callbacks=[Plotter(show_regressions=False, save_to_filepath=fpath, show_plot_window=False)])

    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1, validation_data=(X_test, Y_test),
              callbacks=[Plotter(show_regressions=False, save_to_filepath=fpath, show_plot_window=False)])



    score = model.evaluate(X_test, Y_test, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    pathWeights='model'+loss_function+'.h5'
    model.save_weights(pathWeights)


    '''
    If key_input == True:
        save_weight()
    '''


    '''
    score = model.evaluate(X_valid, Y_valid, verbose=1)
    print('Valid score:', score[0])
    print('Valid accuracy:', score[1])
    '''
