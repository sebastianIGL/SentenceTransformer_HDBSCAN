import pandas as pd
from sentence_transformers import SentenceTransformer
from collections import Counter

# Paso 1: Cargar el archivo CSV
ruta_archivo = "categoria_otro_filtrado.csv"  # ← Cambia esto por la ruta real
df = pd.read_csv(ruta_archivo)


# Paso 2: Preparar los textos y limpiar vacíos o irrelevantes
columna_texto = 'MotivoAdherencia'  # ← Asegúrate de que el nombre sea correcto
def es_texto_valido(texto):
    t = str(texto).strip()
    # Considera inválido si está vacío o solo tiene puntos/espacios
    return bool(t) and not all(c in '. ' for c in t)

df = df[df[columna_texto].apply(es_texto_valido)].copy()
textos = df[columna_texto].astype(str).fillna("").tolist()



# Paso 6.1: Crear etiquetas automáticas para cada cluster y permitir iteración interactiva para robustecer categorías
import re

def limpiar_texto(texto):
    # Elimina signos de puntuación y caracteres especiales básicos
    return re.sub(r'[.,;:!¡¿?"\'\(\)\[\]{}<>\-_/\\]', '', texto.lower())

def sugerir_palabras_clave(textos, top_n=10):
    palabras = []
    for texto in textos:
        limpio = limpiar_texto(texto)
        palabras += re.findall(r'\b\w{5,}\b', limpio)
    counter = Counter(palabras)
    return [palabra for palabra, _ in counter.most_common(top_n)]

def asignar_etiqueta(texto, categorias):
    texto_l = limpiar_texto(texto)
    for categoria, palabras in categorias.items():
        if any(palabra in texto_l for palabra in palabras):
            return categoria
    return "Otros"

# Diccionario inicial de categorías
categorias = {
    "Buena experiencia con ejecutivos": ["buena atencion", "buen trato", "ejecutivo", "personal", "profesional"],
    "Mala experiencia con la competencia": ["mala atencion", "competencia", "isl", "otra mutual", "anterior"],
    "Recomendación de terceros": ["recomendacion", "recomendado", "referencia", "prevencionista"],
    "Obligación contractual": ["obligacion", "exige", "requisito", "licitacion", "contrato", "legal"],
    "Cercanía geográfica o conveniencia": ["cercania", "localidad", "sucursal", "faena", "presencia"],
    "Confianza en la mutual": ["confianza", "confiable", "trayectoria", "experiencia", "historia", "todo ok"],
    "Costos o beneficios económicos": ["costo", "beneficio", "arancel", "precio", "economico"],
    "Prestaciones o herramientas destacadas": ["plataforma", "herramienta", "curso", "capacitacion", "web", "documentos", "servicio"],
}

# Asignar etiquetas iniciales
df['CategoriaDescriptiva'] = df[columna_texto].apply(lambda x: asignar_etiqueta(x, categorias))



# Iterar para robustecer categorías con control de ciclos infinitos y nombres representativos
max_iteraciones_sin_cambio = 3
iteraciones_sin_cambio = 0
prev_cantidad_otros = None
while True:
    otros_df = df[df['CategoriaDescriptiva'] == "Otros"]
    cantidad_otros = len(otros_df)
    if otros_df.empty:
        print("\nNo quedan textos en 'Otros'.")
        print("\nCATEGORÍAS FINALES Y PALABRAS/CLAVES ASOCIADAS:")
        for cat, palabras in categorias.items():
            print(f"- {cat}: {palabras}")
        print("\nVista previa del DataFrame categorizado:")
        print(df[[columna_texto, 'CategoriaDescriptiva']].head(20))

        # Guardar archivo final solo con categorías válidas (solo categorías originales)
        df['CategoriaDescriptiva'] = df[columna_texto].apply(lambda x: asignar_etiqueta(x, categorias))
        df_final = df[df['CategoriaDescriptiva'].isin(list(categorias.keys()))].copy()
        nombre_salida = "categoria_otro_filtrado_categorizado.xlsx"
        df_final.to_excel(nombre_salida, index=False)
        print(f"\nArchivo guardado como: {nombre_salida}")
        break
    print(f"\nQuedan {cantidad_otros} textos en 'Otros'. Mostrando ejemplos:")
    ejemplos = otros_df[columna_texto].sample(n=min(5, cantidad_otros), random_state=42).tolist()
    for i, texto in enumerate(ejemplos, 1):
        print(f"  {i}. {texto}")

    sugeridas = sugerir_palabras_clave(otros_df[columna_texto].tolist(), top_n=8)
    print("\nPalabras/frases clave sugeridas para nuevas categorías o para agregar a existentes:")
    for idx, palabra in enumerate(sugeridas, 1):
        print(f"  {idx}. {palabra}")

    # Lógica automática: agregar sugeridas a la categoría más cercana semánticamente
    posibles_categorias = list(categorias.keys())
    conteo = {cat: 0 for cat in posibles_categorias}
    for texto in otros_df[columna_texto].tolist():
        for cat, palabras in categorias.items():
            if any(palabra in texto.lower() for palabra in palabras):
                conteo[cat] += 1
    if all(v == 0 for v in conteo.values()):
        # Buscar palabras clave que se parezcan a las categorías originales
        palabras_unidas = ' '.join(sugeridas)
        # Elegir la categoría original más cercana por coincidencia de palabras
        import difflib
        mejor_cat = difflib.get_close_matches(palabras_unidas, posibles_categorias, n=1)
        if mejor_cat:
            nombre_nueva = f"Relacionado a {mejor_cat[0]}"
        else:
            nombre_nueva = f"Relacionado a Otros motivos"
        categorias[nombre_nueva] = sugeridas.copy()
        print(f"Se creará la categoría '{nombre_nueva}' y se agregarán las palabras/frases: {', '.join(sugeridas)}")
    else:
        cat_name = max(conteo, key=conteo.get)
        categorias[cat_name].extend([p for p in sugeridas if p not in categorias[cat_name]])
        print(f"Se agregarán las palabras/frases: {', '.join(sugeridas)} a la categoría '{cat_name}'")

    # Re-categorizar solo los que siguen en 'Otros'
    mask_otros = df['CategoriaDescriptiva'] == "Otros"
    df.loc[mask_otros, 'CategoriaDescriptiva'] = df.loc[mask_otros, columna_texto].apply(lambda x: asignar_etiqueta(x, categorias))

    # Control de ciclos infinitos
    if prev_cantidad_otros is not None and cantidad_otros == prev_cantidad_otros:
        iteraciones_sin_cambio += 1
    else:
        iteraciones_sin_cambio = 0
    prev_cantidad_otros = cantidad_otros
    if iteraciones_sin_cambio >= max_iteraciones_sin_cambio:
        print(f"\nNo se logró reducir la cantidad de textos en 'Otros' tras {max_iteraciones_sin_cambio} iteraciones consecutivas. Finalizando para evitar ciclo infinito.")
        print("\nCATEGORÍAS FINALES Y PALABRAS/CLAVES ASOCIADAS:")
        for cat, palabras in categorias.items():
            print(f"- {cat}: {palabras}")
        print("\nVista previa del DataFrame categorizado:")
        print(df[[columna_texto, 'CategoriaDescriptiva']].head(20))
        # Guardar archivo final solo con categorías válidas (solo categorías originales)
        df['CategoriaDescriptiva'] = df[columna_texto].apply(lambda x: asignar_etiqueta(x, categorias))
        df_final = df[df['CategoriaDescriptiva'].isin(list(categorias.keys()))].copy()
        nombre_salida = "categoria_otro_filtrado_categorizado.xlsx"
        df_final.to_excel(nombre_salida, index=False)
        print(f"\nArchivo guardado como: {nombre_salida}")
        break