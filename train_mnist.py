import argparse


import tensorflow as tf
import keras
from keras import backend as K

import cloudpickle
import pandas as pd
import mlflow
import mlflow.keras
import mlflow.pyfunc
from mlflow.pyfunc import PythonModel
from mlflow.utils.environment import _mlflow_conda_env
import json
#configure mlflow
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/var/www/project/mlflow/project/mlflow_test_2")
# Not worked with this configuration.

# mlflow.keras.autolog()
# define a set of arguments
parser = argparse.ArgumentParser(
    description='MNIST Classification')
parser.add_argument('--batch-size', '-b', type=int, default=128)
parser.add_argument('--epochs', '-e', type=int, default=4)
parser.add_argument('--learning-rate', '-l', type=float, default=0.05)
parser.add_argument('--num-hidden-units', '-n', type=int, default=512)
parser.add_argument('--dropout', '-d', type=float, default=0.25)
parser.add_argument('--momentum', '-m', type=float, default=0.85)
args = parser.parse_args()
with mlflow.start_run():
    mlflow.log_param('batch_size', args.batch_size)
    mlflow.log_param('epochs', args.epochs)
    mlflow.log_param('learning_rate', args.learning_rate)




# load mnist digit classification datasets
mnist = keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# define feed-forward nn using keras sequential api 
model = keras.models.Sequential([
  keras.layers.Flatten(input_shape=x_train[0].shape),
  keras.layers.Dense(args.num_hidden_units, activation=tf.nn.relu),
  keras.layers.Dropout(args.dropout),
  keras.layers.Dense(10, activation=tf.nn.softmax)
])
print(x_test)
with open('result.json', '+w') as f:
    df = pd.DataFrame(data=x_test[0])
    f.write(df.to_json(orient='split'))
f.close()
print(y_test)
#
optimizer = keras.optimizers.SGD(lr=args.learning_rate,
                                 momentum=args.momentum,
                                 nesterov=True)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

class MLflowLogClassifier(keras.callbacks.Callback):
    def on_epoch_end(self, epoch,logs={}):
        print(logs.keys())
        mlflow.log_metric('training_loss', logs['loss'],epoch)
        mlflow.log_metric('training_accuracy', logs['accuracy'],epoch)



# Fits the model in training data
model.fit(x_train, y_train,
          epochs=args.epochs,
          batch_size=args.batch_size,
          callbacks=[MLflowLogClassifier()])

# Evaluate by test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
mlflow.log_metric('test_loss', test_loss)
mlflow.log_metric('test_accuracy', test_acc)

conda_env = _mlflow_conda_env(
    additional_conda_deps=[
        "keras=={}".format(keras.__version__),
        "tensorflow=={}".format(tf.__version__),
    ],
    additional_pip_deps=[
        "cloudpickle=={}".format(cloudpickle.__version__),
    ])

# class KerasMnistCNN(PythonModel):

#     def load_context(self, context):
#         import tensorflow as tf
#         self.graph = tf.Graph()
#         with self.graph.as_default():
#             K.set_learning_phase(0)
#             self.model = mlflow.keras.load_model(context.artifacts["keras-model"])

#     def predict(self, context, input_df):
#         with self.graph.as_default():
#             return self.model.predict(input_df.values.reshape(-1, 28, 28))

# mlflow.pyfunc.log_model(
#     artifact_path="keras-model",
#     python_model=KerasMnistCNN(),
#     conda_env=conda_env)

mlflow.keras.log_model(model,artifact_path='keras-model')

mlflow.end_run()
