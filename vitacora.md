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
Separacion de notebook de prediccion y entrenamiento
Re estructuración de carpetas del proyecto
Almacenamiento y carga de modelos y pesos para NN, word2vec y tikenizer
Inferir vectores en vez de entrenar con los tests.
Explroación de hyperprametros fallida
08/01
Probar la rquitectura más grande sin overfitting
Comenzar con otra arquitectura
09/01

AI pipeline for text
1. Tokenized
2. Embedding or bag of words
3. Model
4. Train

1. Tokenizar
    1. Codear un Tokenizer (NO)
    2. nltk.word_tokenizer
    3. tokenizer.tokenizer (Entrenar un tokenizer)
    4. transformers.BertTokenizer (Entrenar bert para tokenizar)
(Quitar stopwords)
2. Embeddings
    1. gensim.word2vec
    2. gensim.doc2vec 
    3. GloVE
    4. fastVec
    5. transformer.Bert
3. Models
    1. NN
    2. CNN
    3. RNN
    4. BERT
4. Training using keras.

Que hemos probado:
1. score=0.5
    1. Word_tokenizer
    2. doc2vec
    3. NN
2. score=0.7
    1. tokenizer.tokenizer
    2. doc2vec
    3. NN

Puedo probar:
1. 
    1. tokenizer.tokenizer
    2. word2vec
    3. CNN
2. 
    1. BERTTokenizer
    2. word2vec
    3. CNN    
3. 
    1. tokenizer.tokenizer
    2. word2vec
    3. RNN
4. 
    1. BERTTokenizer
    2. word2vec
    3. RNN    
5. 
    1. BERTokenizer
    2. BERT
6. OTROS TRANSFORMERS