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
Investigue que otros modelos podia usar.
Separación del notebook en diferentes partes.
La utilizar word2vec, no se pueden inferir nuevas palabras.
Que solución hay? cambio de generador de embeddings? Voy directo a bert?
10/01
word2vec no acepta palabras fuera del vocabulario. Cambio a fastText de gesim.
Existe la posibildad de entrenar con un embbeding por palabra.
11/01 NADA
12/01 NADA
13/01 
Tokenización y utilziación del modelo BERT
Entrenamiento con batch 20 (maximo) en colab tarda 50 minutos por epoca.
30 minutos en hace predicción de todo el dataset.
Demasiados parametros, mucho entramiento poco veneficio.
14/01 Trabajar con la NN, Overfitting, y despues metodos de regularización.
https://www.kaggle.com/code/residentmario/full-batch-mini-batch-and-online-learning
Pobrando un full-batch con la NN detectamos que no overfitea ni con 50 epocas.
Con un online batch es decir de 1, en 25 epocas logra aprender casi todos los ejemplos
Según lo leido esto se debe a la cantidad de veces que se actualizan los pesos.
Basicamente interpreto que al utilizar un full-batch training cae en un optimo local.
Porbando superregularizada, solo con dropout 
15/01 Obtención del posible solucíon provista por ususarios de kaggle
16/01 Nada
17/01 Nada
18/01 Dataset de datos más grande. Prueba de solución propuesta. No funcional debido a falta de hardware para computo. 
Abandono el concurso. Tomo el dataset de doctor cat y lo utilizo para hacer todos los experimentos.
22/01 Realize la pruebas sobre el dataset unicamente y los resultados son buenos. Claramente 
los resultados en la pagina de kaggle tienen que ver con que los datos que uso en el modelo
no son suficientenmente parecidos a los del dataset dde test de la competencia y ademas
de que las NN no generalizan suficientemente bien.
23/01 Redactando informe
24/01 nada
25/01 Escribiendo informe
26/01 Redactando informe

@TODO
SOLUCIONAR PROBLEMA
4. Realizar una corrida del NN
5. Realizar inferencia y ver el resultado de subirlo a la pagina kaggle
6. Tomar las screenshots y graficos generados para el informe.
7. pegar las screenshots.
8. Colocar citas de libros y marco teorico.
9. Enviar el borrador con 1 experimento al profesor.
10. Reunirme para saber cambios. Hago más experimentos? 
11. ???
12. Realizar una presentación para resumir el informe. Que sea corto.


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
    2. doc2vec
    3. CNN
2. 
    1. BERTTokenizer
    2. doc2vec
    3. CNN    
3. 
    1. tokenizer.tokenizer
    2. doc2vec
    3. RNN
4. 
    1. BERTTokenizer
    2. doc2vec
    3. RNN    
5. 
    1. BERTokenizer
    2. BERT
6. OTROS TRANSFORMERS