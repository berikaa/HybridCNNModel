tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/fit/".format(NAME))

model.compile(loss='mean_squared_error',optimizer='rmsprop',metrics=['accuracy'])

model.summary()
#(learning_rate=0.001, rho=0.9) #...
model.fit(train_filenames, train_labels,
          batch_size=32,
          epochs=10,
          verbose=1,
          validation_data=(val_filenames, val_labels),
          callbacks=[tensorboard])
