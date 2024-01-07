01/01:Exploración de datos de la competición
02/01:Investigación e interpretación del problema. Investifación de Imbalanced set.
03/01: Implementación e investigación de Embeddings y bag of words. StopWords y Dictionary. Generación de datos con ChatGPT3.
04/01 Normalización de Embeddings, entrenamiento de la primera red NN con metrica ROC AUC. Primera subida de notebook a la competición. Puntuación 5.0. Muy malo
05/01 Generación de nuevos datos con LLM. Probando Api openai, limitación de pago. Instalando local llama 2 7b,limitado por GPU, cambiando a version con solo CPU.lama 2 7B nesesita 30 gb de memoria RAM.
No usar Llama2. Evaluando si hacerlo a mano o pagar quota.
Decido generar un poco más de datos de manera manual y luego centrarme en el notebook que habla sobre tokenization.
06/01 Utilizando ChatGPT3.5,Llama 70b y Mixtral8x4 para hacer unos ejemplos más.
Creando un tokenizador con la libreria tokenaizers de huggingface.
Subimos de 0.5 a 0.56 esto gracias al tokenizador y evitar overfitting.
1- Faltan datos en mi dataset.
2- Podemos probar con transformers.
3- Podemos probar con RNN.
07/01 Descargando el dataset de , publicado en la discussion.
https://www.kaggle.com/datasets/thedrcat/daigt-v3-train-dataset/?select=train_v3_drcat_02.csv

Separar la prueba en test del entrenamiento
Re ordenar directorios.
Crear un notebook donde se carguen todos los pesos de NN,word2vec y tokenizer.
Inferir vectores en vez de entrenar con los tests?