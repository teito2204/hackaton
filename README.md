#  Modelo Predictivo Empresarial de Fallos en Producción

##  Descripción del Proyecto

Este proyecto implementa un sistema de **machine learning** para predecir fallos en procesos de producción industrial. Utiliza dos algoritmos principales: **Random Forest** y **Regresión Logística**, permitiendo a las empresas anticipar problemas y optimizar sus operaciones de mantenimiento.

## Características Principales

- ✅ **Análisis exploratorio de datos** con visualizaciones interactivas
- ✅ **Dos modelos predictivos** (Random Forest y Regresión Logística)
- ✅ **Interfaz web intuitiva** desarrollada con Streamlit
- ✅ **Preprocesamiento automático** de datos
- ✅ **Métricas de evaluación** completas
- ✅ **Exportación de reportes** en PDF y Excel
- ✅ **Chatbot integrado** para consultas sobre el modelo
- ✅ **Análisis de importancia** de variables

## Requisitos del Sistema

### Requisitos de Software
- Python 3.8 o superior
- Navegador web moderno (Chrome, Firefox, Edge)
- Windows/Linux/macOS

### Dependencias de Python
```
streamlit
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
imbalanced-learn
fpdf2
```

## 🛠️ Instalación Paso a Paso

### 1. Clonar o Descargar el Proyecto
```bash
# Si tienes git instalado
git clone <url-del-repositorio>

# O descarga manualmente el archivo ZIP y extráelo
```

### 2. Navegar al Directorio del Proyecto
```bash
cd Hackaton
```

### 3. Crear un Entorno Virtual (Recomendado)
```bash
# En Windows
python -m venv .venv
.venv\Scripts\activate

# En Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Instalar Dependencias
```bash
pip install streamlit pandas numpy matplotlib seaborn plotly scikit-learn imbalanced-learn fpdf2
```

### 5. Verificar la Instalación
```bash
# Verificar que Streamlit está instalado correctamente
streamlit --version
```

## Estructura del Dataset

El modelo espera un archivo CSV con las siguientes columnas:

### Columnas Requeridas:
- **timestamp**: Fecha y hora del registro (opcional)
- **turno**: Turno de trabajo (mañana, tarde, noche)
- **operador_id**: Identificador del operador
- **maquina_id**: Identificador de la máquina
- **producto_id**: Identificador del producto
- **temperatura**: Temperatura en grados Celsius
- **vibración**: Nivel de vibración
- **humedad**: Porcentaje de humedad
- **tiempo_ciclo**: Tiempo de ciclo en minutos
- **cantidad_producida**: Unidades producidas
- **unidades_defectuosas**: Unidades defectuosas
- **eficiencia_porcentual**: Porcentaje de eficiencia
- **consumo_energia**: Consumo de energía
- **paradas_programadas**: Número de paradas programadas
- **paradas_imprevistas**: Número de paradas imprevistas
- **fallo_detectado**: Variable objetivo ('Sí' o 'No')

### Ejemplo de Formato CSV:
```csv
timestamp,turno,operador_id,maquina_id,producto_id,temperatura,vibración,humedad,tiempo_ciclo,cantidad_producida,unidades_defectuosas,eficiencia_porcentual,consumo_energia,paradas_programadas,paradas_imprevistas,fallo_detectado
2024-01-01 08:00:00,mañana,OP001,MAQ001,PROD001,75.5,2.3,45.2,12.5,100,2,98.0,150.5,0,1,No
2024-01-01 08:15:00,mañana,OP001,MAQ001,PROD001,78.2,3.1,46.8,13.2,95,5,94.7,155.2,0,0,Sí
```

##  Guía de Uso

### 1. Ejecutar la Aplicación

#### Opción A: Ejecutar desde el directorio del proyecto
```bash
# Navegar al directorio .venv/project
cd .venv/project

# Ejecutar Streamlit
streamlit run modelo_predictivo.py
```

#### Opción B: Ejecutar desde el directorio raíz
```bash
# Desde el directorio Hackaton
streamlit run .venv/project/modelo_predictivo.py
```

### 2. Acceder a la Aplicación
- Se abrirá automáticamente en tu navegador
- URL por defecto: `http://localhost:8501`
- Si no se abre automáticamente, copia la URL que aparece en la terminal

### 3. Cargar Datos

#### Usando Datos por Defecto:
- La aplicación incluye un dataset de ejemplo (`reto_dataset.csv`)
- Si no cargas un archivo, usará estos datos automáticamente

#### Cargando tus Propios Datos:
1. En la **barra lateral izquierda**, encontrarás una zona de carga de archivos
2. **Arrastra y suelta** tu archivo CSV o **haz clic para buscar**
3. El archivo debe cumplir con el formato especificado arriba
4. Si hay errores, la aplicación te notificará y usará los datos por defecto

### 4. Seleccionar Modelo
En la barra lateral, elige entre:
- **Random Forest**: Ideal para patrones complejos y datos ruidosos
- **Regresión Logística**: Mejor para interpretabilidad y explicabilidad

##  Entrenamiento del Modelo

### Proceso Automático de Entrenamiento

El sistema ejecuta automáticamente los siguientes pasos:

#### 1. Preprocesamiento de Datos
```python
# Elimina columnas irrelevantes
# Convierte variables categóricas a numéricas
# Imputa valores faltantes
# Normaliza las variables numéricas
```

#### 2. División de Datos
- **80%** para entrenamiento
- **20%** para prueba
- División aleatoria con semilla fija para reproducibilidad

#### 3. Balanceo de Clases
- Utiliza **SMOTE** (Synthetic Minority Oversampling Technique)
- Genera muestras sintéticas para balancear las clases
- Mejora el rendimiento en casos de desbalance

#### 4. Entrenamiento del Modelo

##### Random Forest:
```python
model = RandomForestClassifier(random_state=42)
model.fit(X_train_balanced, y_train_balanced)
```

##### Regresión Logística:
```python
model = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
model.fit(X_train_balanced, y_train_balanced)
```

#### 5. Evaluación Automática
- Matriz de confusión
- Precisión, recall, F1-score
- Error absoluto medio (MAE)
- Coeficiente de determinación (R²)

##  Navegación por Pestañas

###  Pestaña 1: "Análisis de datos"
**Qué muestra:**
- Datos originales sin procesar
- Resumen estadístico completo
- Conteo de valores nulos
- Distribuciones de variables numéricas
- Gráficos de variables categóricas

**Cómo usar:**
1. Revisa la calidad de tus datos
2. Identifica patrones y anomalías
3. Verifica la distribución de la variable objetivo

###  Pestaña 2: "Resultados del modelo"
**Qué muestra:**
- Datos después del preprocesamiento
- Métricas de rendimiento del modelo
- Matriz de confusión interactiva
- Reporte de clasificación detallado
- Informe ejecutivo con recomendaciones

**Cómo interpretar:**
- **Accuracy > 0.85**: Excelente rendimiento
- **Accuracy 0.70-0.85**: Buen rendimiento
- **Accuracy < 0.70**: Requiere mejoras

**Acciones disponibles:**
- Descargar datos procesados en Excel
- Generar reporte ejecutivo en PDF

###  Pestaña 3: "Importancia de variables"
**Qué muestra:**
- Ranking de variables más importantes
- Gráfico de barras interactivo
- Tabla con valores numéricos de importancia

**Cómo usar:**
1. Identifica qué variables más influyen en las predicciones
2. Enfoca los esfuerzos de monitoreo en estas variables
3. Considera recopilar más datos de variables importantes

###  Pestaña 4: "Asistente"
**Qué incluye:**
- Chatbot interactivo para consultas
- Respuestas sobre el funcionamiento del modelo
- Explicaciones de métricas y resultados
- Guía para interpretación de resultados

##  Interpretación de Resultados

### Matriz de Confusión
```
                Predicción
Real        No Fallo    Fallo
No Fallo       TN        FP
Fallo          FN        TP
```

- **TN (True Negative)**: Casos sin fallo predichos correctamente
- **TP (True Positive)**: Fallos detectados correctamente
- **FP (False Positive)**: Falsa alarma (predice fallo pero no hay)
- **FN (False Negative)**: Fallo no detectado (más crítico)

### Métricas Clave

#### Precision (Precisión)
```
Precision = TP / (TP + FP)
```
- Proporción de predicciones de fallo que fueron correctas
- **Alta precisión**: Pocas falsas alarmas

#### Recall (Sensibilidad)
```
Recall = TP / (TP + FN)
```
- Proporción de fallos reales que fueron detectados
- **Alto recall**: Pocos fallos pasan desapercibidos

#### F1-Score
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```
- Media armónica entre precisión y recall
- **F1 alto**: Buen balance entre precisión y recall

### Recomendaciones por Métrica

#### Si Precision es Baja (< 0.7):
- Demasiadas falsas alarmas
- Revisar umbral de decisión
- Mejorar calidad de datos

#### Si Recall es Bajo (< 0.7):
- Muchos fallos no detectados
- Crítico para seguridad
- Considerar ajustar modelo o recopilar más datos

#### Si F1-Score es Bajo (< 0.7):
- Modelo desbalanceado
- Revisar preprocesamiento
- Considerar técnicas de balanceo adicionales

##  Exportación de Resultados

###  Excel - Datos Procesados
**Contenido:**
- Dataset limpio y preprocesado
- Variables codificadas
- Valores imputados
- Listo para análisis adicional

**Cómo descargar:**
1. Ve a la pestaña "Resultados del modelo"
2. Haz clic en "Descargar datos procesados en Excel"
3. El archivo se guardará en tu carpeta de Descargas

###  PDF - Reporte Ejecutivo
**Contenido:**
- Resumen ejecutivo no técnico
- Métricas principales
- Variables más importantes
- Diagnóstico del modelo
- Recomendaciones empresariales
- Visualizaciones clave

**Cómo generar:**
1. Entrena el modelo en la pestaña "Resultados del modelo"
2. Haz clic en "Descargar reporte del modelo en PDF"
3. Espera unos segundos mientras se genera
4. El PDF se descargará automáticamente

## 🛠️ Solución de Problemas

### Problema: "ModuleNotFoundError"
**Solución:**
```bash
pip install [nombre_del_módulo_faltante]
```

### Problema: "Error al leer el archivo CSV"
**Causas comunes:**
- Formato incorrecto del CSV
- Codificación de caracteres
- Columnas faltantes o mal nombradas

**Solución:**
1. Verifica que el CSV tenga todas las columnas requeridas
2. Asegúrate de que use separador de coma (,)
3. Guarda el archivo con codificación UTF-8

### Problema: "La aplicación no carga"
**Solución:**
```bash
# Reinstalar Streamlit
pip uninstall streamlit
pip install streamlit

# Verificar versión de Python
python --version  # Debe ser 3.8+
```

### Problema: "Rendimiento del modelo muy bajo"
**Posibles causas:**
- Datos de baja calidad
- Variables irrelevantes
- Desbalance extremo de clases
- Datos insuficientes

**Soluciones:**
1. Revisar calidad de los datos originales
2. Aumentar la cantidad de datos
3. Considerar ingeniería de características adicional
4. Probar con el otro modelo disponible

### Problema: "Página en blanco en el navegador"
**Solución:**
1. Verifica la URL: `http://localhost:8501`
2. Prueba en modo incógnito
3. Reinicia la aplicación:
   ```bash
   Ctrl+C  # Detener la aplicación
   streamlit run modelo_predictivo.py  # Reiniciar
   ```

## 🔧 Personalización Avanzada

### Modificar Parámetros del Modelo

#### Random Forest:
```python
# En la función entrenar_y_evaluar(), línea ~75
model = RandomForestClassifier(
    n_estimators=100,        # Número de árboles
    max_depth=None,          # Profundidad máxima
    min_samples_split=2,     # Mínimo de muestras para dividir
    random_state=42
)
```

#### Regresión Logística:
```python
# En la función entrenar_y_evaluar(), línea ~81
model = LogisticRegression(
    max_iter=1000,           # Máximo de iteraciones
    C=1.0,                   # Regularización
    solver='lbfgs',          # Algoritmo de optimización
    random_state=42
)
```

### Agregar Nuevas Variables
1. Modifica la función `preprocesar_datos()`
2. Añade el procesamiento para nuevas columnas
3. Actualiza la documentación del formato CSV

### Cambiar el Split de Datos
```python
# En la función entrenar_y_evaluar(), línea ~70
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, 
    test_size=0.3,    # Cambiar proporción de prueba
    random_state=42
)
```

## 📚 Recursos Adicionales

### Documentación de Librerías
- [Streamlit](https://docs.streamlit.io/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [Pandas](https://pandas.pydata.org/docs/)
- [Plotly](https://plotly.com/python/)

### Conceptos de Machine Learning
- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#random-forests)
- [Regresión Logística](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)

##  Soporte y Contribuciones

### Reportar Problemas
1. Describe el problema detalladamente
2. Incluye el mensaje de error completo
3. Especifica tu versión de Python y SO
4. Adjunta una muestra de tus datos (sin información sensible)

### Mejoras Sugeridas
- [ ] Soporte para más formatos de archivo (Excel, JSON)
- [ ] Más algoritmos de machine learning
- [ ] Validación cruzada automática
- [ ] Optimización de hiperparámetros
- [ ] Predicciones en tiempo real
- [ ] Dashboard de monitoreo continuo

##  Licencia

Este proyecto está desarrollado para fines educativos y empresariales. 

##  Créditos

Desarrollado como parte del proyecto de hackathon para soluciones empresariales de machine learning.

---

**¿Necesitas ayuda?** Utiliza el chatbot integrado en la pestaña "Asistente" para consultas específicas sobre el modelo y sus resultados.
