import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

def model_to_pb(model, output_dir="./", output_graph_name='model.pb', output_names=["dense_1/Softmax"]):
    # convert variables in the model graph to constants
    ksess = tf.keras.backend.get_session()
    constants_graph = tf.graph_util.convert_variables_to_constants(ksess, ksess.graph.as_graph_def(), output_names)

    # save the model in .pb
    tf.io.write_graph(constants_graph, output_dir, output_graph_name, as_text=False)

    print(output_names)

model_to_pb(model)