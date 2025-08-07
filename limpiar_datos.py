import pandas as pd

# Ruta del archivo de entrada
archivo_entrada = "Adhesion-6_agosto_2025.csv"

# Leer el archivo CSV y seleccionar solo las columnas necesarias

# Definir las columnas originales y leer solo esas


# Definir las columnas a extraer
columnas_originales = [
    "ID de respuesta",
    "¿Cuales son las tres principales razones por las que te adheriste a Mutual de Seguridad?  Comentalas en orden de importancia"
]

# Leer el archivo CSV de forma robusta (manejo de encoding y separador)
try:
    df = pd.read_csv(archivo_entrada, encoding="utf-8", sep=";", usecols=columnas_originales)
except Exception as e:
    print(f"Error al leer el archivo: {e}")
    exit(1)

# Renombrar columnas de forma eficiente
df = df.rename(columns={
    'ID de respuesta': 'id',
    '¿Cuales son las tres principales razones por las que te adheriste a Mutual de Seguridad?  Comentalas en orden de importancia': 'MotivoAdherencia'
})
print("-------------- DATOS PROCESADOS ----------------")

# Estadísticas iniciales
total = len(df)
print(f"datos totales: {total}")

# Contar y mostrar datos NA
na_count = df['MotivoAdherencia'].isna().sum()
print(f"datos na: {na_count}")

# Contar y mostrar datos con solo espacios o caracteres raros
import re
def es_raro(x):
    if pd.isna(x):
        return False
    t = str(x).strip()
    return t == '' or all(c in '. ,;:-_¡!¿?"' for c in t)

caracteres_count = df['MotivoAdherencia'].apply(es_raro).sum()
print(f"datos con caracteres: {caracteres_count}")

# Limpiar datos: eliminar filas NA y filas con solo espacios/caracteres raros
df_limpio = df.dropna(subset=['MotivoAdherencia']).copy()
df_limpio = df_limpio[~df_limpio['MotivoAdherencia'].apply(es_raro)]
print(f"datos para analisis: {len(df_limpio)}")

# Guardar el archivo limpio
df_limpio.to_csv("data_limpia.csv", index=False)

print("Archivo limpio guardado como 'data_limpia.csv'")
print("-------------- DATOS LIMPIOS ----------------")
