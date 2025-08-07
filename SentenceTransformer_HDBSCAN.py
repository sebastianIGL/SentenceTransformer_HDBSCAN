import pandas as pd
from sentence_transformers import SentenceTransformer
from collections import Counter


# --- CONFIGURACIÓN DE RUTAS DINÁMICAS ---
import os
print("\n=== CATEGORIZADOR SEMÁNTICO DE MUTUAL DE SEGURIDAD ===")
# --- CONFIGURA AQUÍ LAS RUTAS DE TUS ARCHIVOS ---
ruta_archivo = "data_limpia.csv"  # Cambia por el path de tu archivo de datos
ruta_categorias = "categorias_nuevas.csv"  # Cambia por el path de tu archivo de categorías

# --- NO MODIFICAR DESDE AQUÍ ---
if not os.path.isfile(ruta_archivo):
    raise FileNotFoundError(f"No se encontró el archivo de datos: {ruta_archivo}")
if not os.path.isfile(ruta_categorias):
    raise FileNotFoundError(f"No se encontró el archivo de categorías: {ruta_categorias}")
df = pd.read_csv(ruta_archivo)


# Paso 2: Preparar los textos y limpiar vacíos o irrelevantes
columna_texto = 'MotivoAdherencia'  # ← Asegúrate de que el nombre sea correcto
# Antes de limpiar
total_original = len(df)
def es_texto_valido(texto):
    t = str(texto).strip()
    return bool(t) and not all(c in '. ' for c in t)

# Contar cuántos serán eliminados por caracteres especiales
no_validos = (~df[columna_texto].apply(es_texto_valido)).sum()
print(f"Total de filas originales: {total_original}")
print(f"Filas eliminadas por caracteres especiales/vacíos: {no_validos}")
print(f"Filas que quedan después de limpiar: {total_original - no_validos}")

# Limpiar datos
df = df[df[columna_texto].apply(es_texto_valido)].copy()
textos = df[columna_texto].astype(str).fillna("").tolist()



# Paso 6.1: Crear etiquetas automáticas para cada cluster y permitir iteración interactiva para robustecer categorías
import re

def limpiar_texto(texto):
    # Elimina signos de puntuación y caracteres especiales básicos
    return re.sub(r'[.,;:!¡¿?"\'\(\)\[\]{}<>\-_/\\]', '', texto.lower())

def sugerir_palabras_clave(textos, top_n=10):
    palabras = []
    frases = []
    for texto in textos:
        limpio = limpiar_texto(texto)
        palabras += re.findall(r'\b\w{5,}\b', limpio)
        frases.append(texto.strip())
    counter = Counter(palabras)
    # Devuelve tanto palabras como frases frecuentes
    return [palabra for palabra, _ in counter.most_common(top_n)], frases[:top_n]


# --- Clasificación semántica usando SentenceTransformer ---
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
umbral= 0.5  # Umbral de similitud para asignar categorías
def asignar_etiqueta_semantica(texto, categorias, modelo, umbral=umbral):
    if not texto or not isinstance(texto, str):
        return "Otros"
    emb_texto = modelo.encode([texto])
    mejor_cat = "Otros"
    mejor_sim = umbral
    for categoria, palabras in categorias.items():
        frases = palabras + [categoria] if palabras else [categoria]
        emb_frases = modelo.encode(frases)
        sims = cosine_similarity(emb_texto, emb_frases)[0]
        if sims.max() > mejor_sim:
            mejor_sim = sims.max()
            mejor_cat = categoria
    return mejor_cat


# --- Cargar categorías desde archivo CSV ---
df_categorias = pd.read_csv(ruta_categorias, encoding="utf-8")
categorias = {
    row["categoria"]: [p.strip() for p in str(row["palabrasClave"]).split(";") if p.strip()]
    for _, row in df_categorias.iterrows()
}


# Inicializar modelo SentenceTransformer
print("Cargando modelo SentenceTransformer...")
modelo_st = SentenceTransformer('paraphrase-MiniLM-L6-v2')
print("Modelo cargado.")

# Asignar etiquetas iniciales usando clasificación semántica
df['CategoriaDescriptiva'] = df[columna_texto].apply(lambda x: asignar_etiqueta_semantica(x, categorias, modelo_st))




# Iterar para robustecer categorías con control de ciclos infinitos y nombres representativos
max_iteraciones_sin_cambio = 3
iteraciones_sin_cambio = 0
prev_cantidad_otros = None
while True:
    otros_df = df[df['CategoriaDescriptiva'] == "Otros"]
    cantidad_otros = len(otros_df)
    print(f"\nQuedan {cantidad_otros} textos en 'Otros'.")
    if otros_df.empty:
        print("\nNo quedan textos en 'Otros'.")
        break



    # Sugerir palabras/frases clave y frases frecuentes para robustecer categorías
    sugeridas, frases_sugeridas = sugerir_palabras_clave(otros_df[columna_texto].tolist(), top_n=8)
    print("\nPalabras clave sugeridas:")
    for idx, palabra in enumerate(sugeridas, 1):
        print(f"  {idx}. {palabra}")
    print("\nFrases frecuentes sugeridas para agregar a categorías:")
    for idx, frase in enumerate(frases_sugeridas, 1):
        print(f"  {idx}. {frase}")

    # Asignar automáticamente cada frase frecuente a la categoría más cercana semánticamente
    for frase in frases_sugeridas:
        emb_frase = modelo_st.encode([frase])
        mejor_cat = None
        mejor_sim = 0.0
        for cat, palabras in categorias.items():
            frases_cat = palabras + [cat] if palabras else [cat]
            emb_cat = modelo_st.encode(frases_cat)
            sim = cosine_similarity(emb_frase, emb_cat)[0].max()
            if sim > mejor_sim:
                mejor_sim = sim
                mejor_cat = cat
        # Solo agregar si la similitud es razonable (ej: >0.4) y la frase no está ya en la categoría
        if mejor_cat and mejor_sim > 0.4 and frase not in categorias[mejor_cat]:
            categorias[mejor_cat].append(frase)
            print(f"Frase '{frase}' agregada automáticamente a la categoría '{mejor_cat}' (similitud: {mejor_sim:.2f})")

    # Re-categorizar solo los que siguen en 'Otros' usando razonamiento semántico
    mask_otros = df['CategoriaDescriptiva'] == "Otros"
    df.loc[mask_otros, 'CategoriaDescriptiva'] = df.loc[mask_otros, columna_texto].apply(lambda x: asignar_etiqueta_semantica(x, categorias, modelo_st))

    # Control de ciclos infinitos
    if prev_cantidad_otros is not None and cantidad_otros == prev_cantidad_otros:
        iteraciones_sin_cambio += 1
    else:
        iteraciones_sin_cambio = 0
    prev_cantidad_otros = cantidad_otros
    if iteraciones_sin_cambio >= max_iteraciones_sin_cambio:
        print(f"\nNo se logró reducir la cantidad de textos en 'Otros' tras {max_iteraciones_sin_cambio} iteraciones consecutivas. Finalizando para evitar ciclo infinito.")
        break

# Guardar archivo final con todas las filas y todas las categorías

# Guardar archivo final con todas las filas y todas las categorías
nombre_base = os.path.splitext(os.path.basename(ruta_archivo))[0]
nombre_salida = f"categorias_nuevas_umbral_{umbral}.xlsx"
df.to_excel(nombre_salida, index=False)
print(f"\nArchivo guardado como: {nombre_salida}")

# Asegurar que todas las categorías presentes en el DataFrame estén en el archivo de categorías
for cat in df['CategoriaDescriptiva'].unique():
    if cat not in categorias:
        categorias[cat] = []

# Exportar el diccionario de categorías robustecido a CSV solo una vez al final
df_categorias_final = pd.DataFrame([
    {"categoria": cat, "palabrasClave": ";".join(palabras)}
    for cat, palabras in categorias.items()
])
nombre_categorias_out = f"{nombre_base}_categorias_final.csv"
df_categorias_final.to_csv(nombre_categorias_out, index=False, encoding="utf-8")
print(f"\nArchivo de categorías actualizado: {nombre_categorias_out}")