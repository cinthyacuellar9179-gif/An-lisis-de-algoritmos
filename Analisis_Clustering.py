import umap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.patches as mpatches


# Cargar datos con pandas
def load_fashion_mnist_pandas(file_path):
    print(f"Cargando datos desde: {file_path}")
    df = pd.read_csv(file_path)
    labels = df.iloc[:, 0].values.astype(int)
    images = df.iloc[:, 1:].values.reshape(-1, 28, 28)
    return images, labels, df


# Cargar el dataset
try:
    images, labels, df = load_fashion_mnist_pandas(
       # "C:/Users/Jose-/OneDrive/Desktop/Analisis de Algoritmo/fashion-mnist_test.csv")
        "C:/Users/karen/PyCharmMiscProject/fashion-mnist_test.csv")
    print(f" Datos cargados: {images.shape[0]} imágenes, {images.shape[1]}x{images.shape[2]} píxeles")
    print(f" DataFrame shape: {df.shape}")
except Exception as e:
    print(f" No se pudo cargar el archivo: {e}, usando datos de ejemplo...")
    # Datos de ejemplo si no carga el archivo
    from sklearn.datasets import fetch_openml

    fashion_mnist = fetch_openml('Fashion-MNIST', version=1, as_frame=False, parser='auto')
    images = fashion_mnist.data.reshape(-1, 28, 28)
    labels = fashion_mnist.target.astype(int)
    # Crear DataFrame simulado
    df = pd.DataFrame(fashion_mnist.data)
    df['label'] = fashion_mnist.target.astype(int)

# Nombres de las clases
class_names = {
    0: 'Playera',
    1: 'Pantalon',
    2: 'Jersey',
    3: 'Vestido',
    4: 'Abrigo',
    5: 'Sandalia',
    6: 'Camisa',
    7: 'Tenis',
    8: 'Bolsa',
    9: 'Bota'
}

# Preprocesamiento usando pandas
print("Preprocesando datos...")
X_flat = images.reshape(images.shape[0], -1)
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_flat)

# Crear DataFrame con características normalizadas
features_df = pd.DataFrame(X_normalized, columns=[f'pixel_{i}' for i in range(X_normalized.shape[1])])
features_df['label'] = labels

# Tomar muestra usando pandas
sample_size = min(5000, len(features_df))
sample_df = features_df.sample(n=sample_size, random_state=42)
X_sample = sample_df.drop('label', axis=1).values
labels_sample = sample_df['label'].values
sample_indices = sample_df.index.values

print(f"Tamaño de la muestra para UMAP: {X_sample.shape}")

# === APLICACIÓN DE UMAP SOBRE EL CONJUNTO COMPLETO ===
print("Aplicando UMAP sobre el conjunto completo...")
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
embedding = reducer.fit_transform(X_sample)

# Añadir embedding al DataFrame de muestra
sample_df = sample_df.copy()
sample_df['umap_x'] = embedding[:, 0]
sample_df['umap_y'] = embedding[:, 1]

print("UMAP completado")


# Visualización de UMAP para el conjunto completo
def plot_umap_full(df, class_names, title="UMAP - Fashion MNIST Completo"):
    plt.figure(figsize=(15, 12))

    scatter = plt.scatter(df['umap_x'], df['umap_y'],
                          c=df['label'], cmap='tab10', s=5, alpha=0.7)

    # Crear leyenda
    legend_elements = []
    for class_id, class_name in class_names.items():
        if class_id in df['label'].values:
            legend_elements.append(mpatches.Patch(
                color=plt.cm.tab10(class_id / 10),
                label=f'{class_id}: {class_name}'
            ))

    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.title(title, fontsize=16)
    plt.xlabel('Eje X')
    plt.ylabel('Eje Y')
    plt.tight_layout()
    plt.show()


plot_umap_full(sample_df, class_names)

# === SELECCIÓN DE UN CLUSTER ESPECÍFICO USANDO PANDAS ===
print("\nSeleccionando cluster de calzado...")
shoe_classes = [5, 7, 9]  # Sandal, Sneaker, Ankle boot
shoes_df = sample_df[sample_df['label'].isin(shoe_classes)].copy()
X_shoes = shoes_df.drop(['label', 'umap_x', 'umap_y'], axis=1).values
labels_shoes = shoes_df['label'].values

print(f"Tamaño del cluster de calzado: {shoes_df.shape[0]} imágenes")

# === APLICAR UMAP SOBRE EL SUBSET DE CALZADO ===
print("Aplicando UMAP al cluster de calzado...")
reducer_shoes = umap.UMAP(n_components=2, random_state=42, n_neighbors=10, min_dist=0.1)
embedding_shoes = reducer_shoes.fit_transform(X_shoes)

# Añadir embedding de calzado al DataFrame
shoes_df = shoes_df.copy()
shoes_df['shoe_umap_x'] = embedding_shoes[:, 0]
shoes_df['shoe_umap_y'] = embedding_shoes[:, 1]


# Visualización del cluster de calzado
def plot_shoe_cluster(df, class_names, title="UMAP - Cluster de Calzado"):
    plt.figure(figsize=(12, 10))

    # Mapear colores específicos para calzado
    color_map = {5: 0, 7: 1, 9: 2}
    colors = df['label'].map(color_map)

    scatter = plt.scatter(df['shoe_umap_x'], df['shoe_umap_y'],
                          c=colors, cmap='Set2', s=20, alpha=0.8)

    # Leyenda personalizada para calzado
    shoe_legend_elements = [
        mpatches.Patch(color=plt.cm.Set2(0), label='Sandalia'),
        mpatches.Patch(color=plt.cm.Set2(1), label='Tenis'),
        mpatches.Patch(color=plt.cm.Set2(2), label='Bota')
    ]

    plt.legend(handles=shoe_legend_elements, loc='upper right')
    plt.title(title, fontsize=16)
    plt.xlabel('Eje X')
    plt.ylabel('Eje Y')
    plt.tight_layout()
    plt.show()


plot_shoe_cluster(shoes_df, class_names)

# === IDENTIFICACIÓN DE SUBCLUSTERS ===
print("Identificando subclusters...")
dbscan = DBSCAN(eps=0.5, min_samples=5)
subcluster_labels = dbscan.fit_predict(embedding_shoes)

# Añadir etiquetas de subcluster al DataFrame
shoes_df['subcluster'] = subcluster_labels

n_subclusters = len(shoes_df['subcluster'].unique()) - (1 if -1 in shoes_df['subcluster'].values else 0)
print(f"Número de subclusters identificados: {n_subclusters}")
print(f"Puntos considerados como ruido: {sum(shoes_df['subcluster'] == -1)}")


# === VISUALIZACIÓN DE SUBCLUSTERS ===
def plot_subclusters(df, class_names, title="Subclusters en Calzado"):
    plt.figure(figsize=(14, 10))

    scatter = plt.scatter(df['shoe_umap_x'], df['shoe_umap_y'],
                          c=df['subcluster'], cmap='tab20', s=30, alpha=0.8)

    # Añadir etiquetas de clase original para algunos puntos
    for i, row in df.iloc[::15].iterrows():  # Mostrar solo algunas etiquetas
        plt.annotate(class_names[row['label']],
                     (row['shoe_umap_x'], row['shoe_umap_y']),
                     fontsize=8, alpha=0.7)

    plt.colorbar(scatter, label='Subcluster')
    plt.title(title, fontsize=16)
    plt.xlabel('Eje X')
    plt.ylabel('Eje Y')
    plt.tight_layout()
    plt.show()


plot_subclusters(shoes_df, class_names)


# === ANÁLISIS DE SUBCLUSTERS USANDO PANDAS ===
def analyze_subclusters_pandas(shoes_df, class_names):
    print("\n=== ANÁLISIS DETALLADO DE SUBCLUSTERS ===")

    # Análisis de ruido
    noise_count = (shoes_df['subcluster'] == -1).sum()
    print(f"\n🔍 Puntos de ruido: {noise_count}")

    # Análisis por subcluster
    for cluster_id in sorted(shoes_df['subcluster'].unique()):
        if cluster_id == -1:
            continue

        cluster_data = shoes_df[shoes_df['subcluster'] == cluster_id]
        total_in_cluster = len(cluster_data)

        # Análisis de distribución de clases
        class_distribution = cluster_data['label'].value_counts()
        dominant_class = class_distribution.index[0]
        dominant_count = class_distribution.iloc[0]
        dominant_percentage = (dominant_count / total_in_cluster) * 100

        print(f"\n Subcluster {cluster_id}:")
        print(f"   - Total de imágenes: {total_in_cluster}")
        print(f"   - Clase dominante: {class_names[dominant_class]} ({dominant_percentage:.1f}%)")
        print(f"   - Distribución completa:")

        for class_id, count in class_distribution.items():
            percentage = (count / total_in_cluster) * 100
            print(f"       {class_names[class_id]}: {count} ({percentage:.1f}%)")


analyze_subclusters_pandas(shoes_df, class_names)


# === MOSTRAR IMÁGENES REPRESENTATIVAS ===
def display_representative_images_pandas(images, shoes_df, class_names):
    print("\n=== IMÁGENES REPRESENTATIVAS DE SUBCLUSTERS ===")

    unique_subclusters = shoes_df['subcluster'].unique()
    unique_subclusters = unique_subclusters[unique_subclusters != -1]

    for cluster_id in unique_subclusters:
        cluster_data = shoes_df[shoes_df['subcluster'] == cluster_id]
        cluster_indices = cluster_data.index

        print(f"\n🖼️  Subcluster {cluster_id} ({len(cluster_data)} imágenes):")

        if len(cluster_data) > 0:
            # Encontrar imagen más central usando las características originales
            cluster_features = cluster_data.drop(
                ['label', 'umap_x', 'umap_y', 'shoe_umap_x', 'shoe_umap_y', 'subcluster'],
                axis=1).values
            centroid = np.mean(cluster_features, axis=0)
            distances = np.linalg.norm(cluster_features - centroid, axis=1)
            rep_idx = np.argmin(distances)

            representative_index = cluster_indices[rep_idx]
            rep_class = class_names[cluster_data.iloc[rep_idx]['label']]

            print(f"   - Imagen representativa: {rep_class}")

            # Mostrar distribución usando pandas
            class_distribution = cluster_data['label'].value_counts()
            for class_id, count in class_distribution.items():
                class_name = class_names[class_id]
                percentage = (count / len(cluster_data)) * 100
                print(f"   - {class_name}: {count} ({percentage:.1f}%)")


# Mostrar imágenes representativas
display_representative_images_pandas(images, shoes_df, class_names)

# === ANÁLISIS ADICIONAL CON PANDAS ===
print("\n=== ANÁLISIS ADICIONAL CON PANDAS ===")
print("\nResumen estadístico por clase en todo el dataset:")
class_summary = pd.DataFrame({
    'Clase': [class_names[i] for i in range(10)],
    'Conteo': [sum(labels == i) for i in range(10)],
    'Porcentaje': [f"{(sum(labels == i) / len(labels)) * 100:.1f}%" for i in range(10)]
})
print(class_summary)

print("\nAnálisis completado con UMAP y pandas!")


