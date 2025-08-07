import pandas as pd
import re
from collections import Counter

def limpiar_texto(texto):
    return re.sub(r"[.,;:!¡¿?\"'()\[\]{}<>\-_/\\]", '', str(texto).lower())

def es_texto_valido(texto):
    t = str(texto).strip()
    return bool(t) and not all(c in '. ' for c in t)

def asignar_etiqueta(texto, categorias):
    texto_l = limpiar_texto(texto)
    for categoria, palabras in categorias.items():
        if any(palabra in texto_l for palabra in palabras):
            return categoria
    return "Otros"

# Diccionario de categorías base
CATEGORIAS_BASE = {
    "Buena experiencia con ejecutivos": ["buena atencion", "buen trato", "ejecutivo", "personal", "profesional"],
    "Mala experiencia con la competencia": ["mala atencion", "competencia", "isl", "otra mutual", "anterior"],
    "Recomendación de terceros": ["recomendacion", "recomendado", "referencia", "prevencionista"],
    "Obligación contractual": ["obligacion", "exige", "requisito", "licitacion", "contrato", "legal"],
    "Cercanía geográfica o conveniencia": ["cercania", "localidad", "sucursal", "faena", "presencia"],
    "Confianza en la mutual": ["confianza", "confiable", "trayectoria", "experiencia", "historia", "todo ok"],
    "Costos o beneficios económicos": ["costo", "beneficio", "arancel", "precio", "economico"],
    "Prestaciones o herramientas destacadas": ["plataforma", "herramienta", "curso", "capacitacion", "web", "documentos", "servicio"],
}

def categorizar_archivo(archivo_entrada, columna_texto='MotivoAdherencia', archivo_salida=None, categorias=None):
    """
    Recibe un archivo .csv o .xlsx, categoriza los textos y guarda un .xlsx con la columna de categoría.
    """
    if categorias is None:
        categorias = CATEGORIAS_BASE.copy()
    # Leer archivo
    if archivo_entrada.endswith('.csv'):
        df = pd.read_csv(archivo_entrada)
    elif archivo_entrada.endswith('.xlsx'):
        df = pd.read_excel(archivo_entrada)
    else:
        raise ValueError("El archivo debe ser .csv o .xlsx")
    # Limpiar textos
    df = df[df[columna_texto].apply(es_texto_valido)].copy()
    # Asignar categoría
    df['CategoriaDescriptiva'] = df[columna_texto].apply(lambda x: asignar_etiqueta(x, categorias))
    # Solo categorías originales
    df_final = df[df['CategoriaDescriptiva'].isin(list(categorias.keys()))].copy()
    # Guardar
    if archivo_salida is None:
        archivo_salida = archivo_entrada.replace('.csv', '_categorizado.xlsx').replace('.xlsx', '_categorizado.xlsx')
    df_final.to_excel(archivo_salida, index=False)
    print(f"Archivo guardado como: {archivo_salida}")
    return df_final

# Ejemplo de uso:
# categorizar_archivo('categoria_otro_filtrado.csv')
# categorizar_archivo('archivo.xlsx', columna_texto='MotivoAdherencia')
