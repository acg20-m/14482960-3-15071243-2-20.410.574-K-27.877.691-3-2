# 14482960-3-15071243-2-20.410.574-K-27.877.691-3-2
Repositorio de comparación de modelos de ML para asignación precisa de los Grupos Relacionados por el Diagnóstico (GRD), lo cual es un cuello de botella crítico en la administración moderna de la salud y la auditoría clínica. 

Resumen (Abstract)— La asignación precisa de los Grupos Relacionados por el Diagnóstico (GRD) es un cuello de botella crítico en la administración moderna de la salud y la auditoría clínica. Este estudio propone un enfoque de Aprendizaje Profundo (Deep Learning) comparando arquitecturas de Redes Neuronales Recurrentes, específicamente LSTM y GRU, para predecir automáticamente los GRD basándose en registros médicos secuenciales. Para abordar el severo desbalance de clases inherente en los datos clínicos, se implementó la Técnica de Sobremuestreo Sintético de Minorías (SMOTE) adaptada para espacios de características discretas (tokens). Entrenado con una cohorte de 1.220 instancias clínicas, el modelo candidato LSTM con SMOTE superó a la arquitectura GRU al demostrar una recuperación (Recall) superior en las clases minoritarias, resolviendo el colapso predictivo del modelo base. Se emplearon técnicas de explicabilidad global y local (SHAP) para validar empíricamente que la red pondera adecuadamente las complicaciones clínicas secundarias por sobre el diagnóstico principal. El trabajo futuro propone arquitecturas multimodales y mecanismos de Atención (Transformers) para la escalabilidad del modelo.

Palabras Clave (Index Terms)— Grupos Relacionados por el Diagnóstico, Aprendizaje Profundo, LSTM, GRU, SMOTE, Registros Médicos Electrónicos (EHR), SHAP.


I. INTRODUCCIÓN
La optimización de la auditoría clínica y los procesos automatizados de facturación médica es esencial para la eficiencia institucional. El sistema de Grupos Relacionados por el Diagnóstico (GRD) agrupa a pacientes con perfiles clínicos y consumo de recursos similares. Sin embargo, la asignación manual de los GRD depende de codificadores humanos que revisan extensos Registros Médicos Electrónicos (EHR), un proceso propenso a errores y retrasos.

El objetivo de este estudio es desarrollar y comparar modelos de aprendizaje automático capaces de predecir automáticamente el GRD de un paciente. Al tratar los diagnósticos y procedimientos del paciente como secuencias lógicas y temporales, se busca aislar la trayectoria clínica para automatizar la clasificación con alta precisión y explicabilidad. 


II. REVISIÓN BIBLIOGRÁFICA
La literatura reciente se ha centrado en la aplicación del Aprendizaje Profundo a los datos de EHR. Los modelos tabulares tradicionales tienen dificultades con la extrema dispersión (sparsity) de los códigos médicos. En consecuencia, las Redes Neuronales Recurrentes (RNN), como LSTM (Long Short-Term Memory) y GRU (Gated Recurrent Unit), se han convertido en el estándar para el procesamiento de secuencias clínicas. Estas arquitecturas capturan eficazmente la progresión lógica de los diagnósticos principales seguidos de los procedimientos. No obstante, los estudios enfrentan frecuentemente el desafío del desbalance de clases clínico, requiriendo técnicas avanzadas de sobremuestreo para evitar sesgos algorítmicos hacia las patologías mayoritarias.


III. ANÁLISIS DE DATOS
El conjunto de datos utilizado (dataset_elpino.csv) contenía inicialmente 14.561 registros de pacientes. 

A. Completitud, Correctitud y Filtrado La evaluación de la calidad reveló valores nulos en la característica 'Edad', imputados utilizando la mediana. Para reducir el ruido y enfocar el alcance predictivo en un escenario de alta complejidad, el conjunto de datos se filtró para incluir los tres GRD más frecuentes (146101, 146102 y 146103), resultando en una cohorte refinada de 1.220 instancias. Las tablas de contingencia confirmaron que los diagnósticos exhiben distribuciones distintas dependiendo del GRD objetivo, asegurando la correctitud de los datos. 

<img width="921" height="455" alt="image" src="https://github.com/user-attachments/assets/e83df002-e02b-4ed4-a4d4-32d9fdc96f52" />

Figura 1: Histograma que ilustra la distribución de edades en la cohorte clínica analizada, demostrando la imputación de valores faltantes.

B. Estadísticas Descriptivas y Desbalance de Clases El Análisis Exploratorio de Datos (EDA) demostró un severo desbalance. La clase mayoritaria (GRD 146101) representa 813 casos, mientras que las clases minoritarias (146102 y 146103) contienen 244 y 163 casos, respectivamente. 

 <img width="921" height="456" alt="image" src="https://github.com/user-attachments/assets/195f4b3f-acdd-4576-9138-0465b4a42ca8" />

Figura 2: Frecuencia de los Grupos Relacionados por el Diagnóstico (GRD), evidenciando el desbalance natural hacia la clase 146101.

IV. METODOLOGÍA
A. Secuenciación y Selección de Características Las características de entrada se limitaron estrictamente a códigos de diagnóstico y procedimiento. Se excluyeron variables demográficas estáticas (Edad, Sexo) para aislar y medir la eficacia del procesamiento puramente secuencial. Se construyó un vocabulario unificado de 533 tokens. Los registros se transformaron en secuencias de enteros y, utilizando la técnica pad_sequences, se estandarizaron a una longitud máxima de 65 tokens. 

B. Balanceo de Datos con SMOTE
Para mitigar el desbalance, se aplicó la Técnica de Sobremuestreo Sintético de Minorías (SMOTE) exclusivamente sobre el conjunto de entrenamiento. Dado que las secuencias alimentan una capa de Embedding, la interpolación generada por SMOTE se transformó aplicando una función de redondeo numérico (np.round().astype(int)), asegurando que las muestras sintéticas mantuvieran la naturaleza discreta de los tokens del vocabulario médico.

C. Métricas de Evaluación Para evaluar los modelos candidatos en un entorno multiclase desbalanceado, se utilizaron las métricas de Precisión, Sensibilidad (Recall), F1-Score (Macro) y curvas ROC-AUC bajo una estrategia One-vs-Rest. 


V. ARQUITECTURAS CANDIDATAS Y EXPERIMENTOS
Se diseñaron dos modelos secuenciales bajo condiciones idénticas (Ceteris Paribus) para su comparación. Ambos reciben secuencias estandarizadas que ingresan a una Capa de Embedding (dimensión 64), mapeando los códigos en un espacio vectorial denso.


•	Candidato 1 (LSTM): Capa oculta con 32 unidades de Memoria a Corto y Largo Plazo, diseñada para retener el contexto clínico prolongado.

•	Candidato 2 (GRU): Capa oculta con 32 unidades recurrentes cerradas, evaluada por su eficiencia computacional y menor riesgo de sobreajuste.

Ambas arquitecturas finalizan en una Capa Densa con activación softmax y fueron entrenadas durante 16 épocas utilizando el optimizador Adam y la función de pérdida categorical cross-entropy, con Early Stopping para evitar sobreajuste. 


VI. RESULTADOS Y SELECCIÓN DEL MODELO
La evaluación en el conjunto de prueba independiente, fijando una semilla determinista para garantizar reproducibilidad científica, arrojó los siguientes resultados comparativos: 

Métrica (Macro Avg)	Modelo LSTM + SMOTE	Modelo GRU + SMOTE
Accuracy	0.889	0.885
Precision	0.881	0.861
Recall	0.858	0.849
F1-Score	0.865	0.854

Justificación de Selección: Se seleccionó empíricamente la arquitectura LSTM como el modelo final. La red LSTM superó a la arquitectura GRU en todas las métricas de evaluación macro (Accuracy, Precision, Recall y F1-Score), demostrando una mayor estabilidad y una capacidad superior de recuperación para los GRD minoritarios tras la integración de las muestras sintéticas. Esto eliminó por completo el colapso predictivo (Recall 0.0) observado en las pruebas preliminares sin balanceo.


 <img width="851" height="735" alt="image" src="https://github.com/user-attachments/assets/ee758af2-bf22-44d2-b995-c69a3e20085b" />

Figura 3: Matriz de confusión multietiqueta del modelo LSTM ganador, evidenciando la recuperación exitosa de las clases minoritarias 146102 y 146103 tras aplicar SMOTE.


 <img width="921" height="688" alt="image" src="https://github.com/user-attachments/assets/117a88e2-7c61-46c4-8fab-d6dc901ef566" />

Figura 4: Curvas ROC One-vs-Rest para el modelo LSTM, demostrando un área bajo la curva (AUC) superior a 0.90 en todas las clases.

VII. ANÁLISIS DE DESEMPEÑO Y EXPLICABILIDAD
Se empleó la librería SHAP (SHapley Additive exPlanations) para analizar la toma de decisiones del modelo LSTM ganador, garantizando la transparencia de la auditoría.

A. Explicabilidad Global
El gráfico de resumen de SHAP reveló empíricamente que la posición cronológica del tercer y cuarto diagnóstico (Diag03, Diag04) posee un impacto predictivo promedio superior al diagnóstico principal de ingreso (Diag01). Esto confirma que el modelo logró inferir la lógica médica subyacente: el GRD suele estar determinado por las comorbilidades y complicaciones secundarias más que por la causa inicial de admisión.


<img width="921" height="1095" alt="image" src="https://github.com/user-attachments/assets/2de1748a-ad16-4dac-859d-989789dec2a3" />

 
Figura 5: Resumen global SHAP (Summary Plot) que clasifica la importancia de las posiciones secuenciales (características) en la predicción del modelo LSTM.

B. Explicabilidad Local (Análisis de Falsos Negativos)
Los gráficos de cascada (Waterfall plots) demostraron cómo el modelo pondera hallazgos contradictorios. En los casos donde la predicción falla, se observó que la influencia negativa de códigos ambiguos tempranos supera el peso clínico de las intervenciones posteriores, lo que señala áreas específicas para mejorar el entrenamiento en futuras iteraciones.


<img width="921" height="339" alt="image" src="https://github.com/user-attachments/assets/a64ff5f8-bafc-4b7c-b533-d3bb1326a8ef" />

 
Figura 6: Análisis local SHAP (Waterfall plot) demostrando la ponderación de atributos en una clasificación.

VIII. CONCLUSIONES Y TRABAJO FUTURO
La integración de SMOTE con la arquitectura secuencial LSTM solucionó el problema crítico del desbalance de clases, permitiendo predecir el Grupo Relacionado por el Diagnóstico con un alto nivel de Exactitud y Sensibilidad. El modelo valida que las RNN son idóneas para extraer el contexto de historiales médicos escuetos. 

Propuestas de Estudios Futuros: Para superar las limitaciones actuales, se proponen dos vías de desarrollo: 


1.	Arquitectura Multimodal: Implementar redes que fusionen la salida secuencial LSTM con capas densas dedicadas al procesamiento simultáneo de variables tabulares estáticas, como la Edad y el Sexo.

2.	Mecanismos de Atención: Si el volumen de datos clínicos a nivel institucional aumenta significativamente en el futuro, se recomienda migrar a arquitecturas basadas en Transformers (ej. ClinicalBERT), las cuales permitirían evaluar dependencias no secuenciales en historias clínicas de longitud masiva, superando las limitaciones inherentes de las redes recurrentes clásicas.

