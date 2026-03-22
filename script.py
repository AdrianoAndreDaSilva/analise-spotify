# ============================================================
# PRÉ-PROCESSAMENTO DE DADOS - SPOTIFY TRACKS DATASET
# ============================================================

import os
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings('ignore')

os.makedirs('resultados', exist_ok=True)

# Configurações visuais
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 10

# ============================================================
# 1. LEITURA DO DATASET
# ============================================================
print("=" * 60)
print("1. LEITURA E EXPLORAÇÃO DO DATASET")
print("=" * 60)

df = pd.read_csv('dataset.csv')  

print(f"\nShape: {df.shape[0]} registros x {df.shape[1]} atributos")
print("\nPrimeiras linhas:")
print(df.head())

print("\nNomes das colunas:")
print(df.columns.tolist())

# ============================================================
# 2. ANÁLISE INICIAL DOS DADOS
# ============================================================
print("\n" + "=" * 60)
print("2. ANÁLISE INICIAL DOS DADOS")
print("=" * 60)

print("\n--- Tipos de variáveis ---")
print(df.dtypes)

print("\n--- Valores ausentes ---")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'Ausentes': missing, 'Percentual (%)': missing_pct})
print(missing_df[missing_df['Ausentes'] > 0] if missing.sum() > 0 else "Nenhum valor ausente encontrado.")

print("\n--- Duplicatas ---")
duplicatas = df.duplicated(subset=['track_name', 'artists']).sum()
print(f"Registros duplicados: {duplicatas}")

print("\n--- Estatísticas descritivas iniciais ---")
print(df.describe())

fig, ax = plt.subplots(figsize=(10, 4))
missing_plot = missing[missing > 0]
if len(missing_plot) > 0:
    missing_plot.plot(kind='bar', ax=ax, color='salmon', edgecolor='black')
    ax.set_title('Valores Ausentes por Coluna')
    ax.set_ylabel('Quantidade')
    ax.set_xlabel('Coluna')
    plt.xticks(rotation=45, ha='right')
else:
    ax.text(0.5, 0.5, 'Nenhum valor ausente encontrado!',
            ha='center', va='center', fontsize=14, color='green',
            transform=ax.transAxes)
    ax.set_title('Valores Ausentes por Coluna')
plt.tight_layout()
plt.savefig('resultados/01_valores_ausentes.png')
print("Gráfico salvo: 01_valores_ausentes.png")

# ============================================================
# 3. PRÉ-PROCESSAMENTO DOS DADOS
# ============================================================
print("\n" + "=" * 60)
print("3. PRÉ-PROCESSAMENTO DOS DADOS")
print("=" * 60)

df_clean = df.copy()

total_registros_inicial = len(df_clean)

# 3.1 Remover duplicatas
antes = len(df_clean)
df_clean.drop_duplicates(subset=['track_name', 'artists'], inplace=True)
removidos_duplicatas = antes - len(df_clean)
print(f"\n[3.1] Duplicatas removidas: {removidos_duplicatas} registros")

# 3.2 Tratar valores ausentes
print("\n[3.2] Tratamento de valores ausentes:")
num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df_clean.select_dtypes(include=['object']).columns.tolist()

ausentes_por_linha = df_clean.isnull().sum(axis=1)
removidos_ausentes = (ausentes_por_linha >= 3).sum()
df_clean = df_clean[ausentes_por_linha < 3].copy()
print(f"  Registros removidos por conta de valores ausentes: {removidos_ausentes}")

for col in num_cols:
    if df_clean[col].isnull().sum() > 0:
        mediana = df_clean[col].median()
        df_clean[col].fillna(mediana, inplace=True)
        print(f"  - '{col}': preenchido com mediana ({mediana:.4f})")

for col in cat_cols:
    if df_clean[col].isnull().sum() > 0:
        moda = df_clean[col].mode()[0]
        df_clean[col].fillna(moda, inplace=True)
        print(f"  - '{col}': preenchido com moda ('{moda}')")

if df_clean.isnull().sum().sum() == 0:
    print("  ✓ Nenhum valor ausente restante.")

# 3.3 Converter duration_ms para minutos (se existir)
if 'duration_ms' in df_clean.columns:
    df_clean['duration_min'] = df_clean['duration_ms'] / 60000
    print("\n[3.3] 'duration_ms' convertida para 'duration_min' (minutos)")

# 3.4 Remover colunas irrelevantes para análise numérica
cols_to_drop = [c for c in ['Unnamed: 0', 'Unnamed: 0.1', 'track_id'] if c in df_clean.columns]
if cols_to_drop:
    df_clean.drop(columns=cols_to_drop, inplace=True)
    print(f"\n[3.4] Colunas removidas: {cols_to_drop}")

# 3.5 Remoção de faixas com popularity = 0
if 'popularity' in df_clean.columns:
    antes = len(df_clean)
    df_clean = df_clean[df_clean['popularity'] > 0].copy()
    removidos_popularity = antes - len(df_clean)
    print(f"\n[3.5] Registros removidos por popularity = 0: {removidos_popularity}")

# 3.6 Validação de intervalos conhecidos
print("\n[3.6] Validação de intervalos:")
validacoes = {
    'duration_ms':        (lambda df: df['duration_ms'] > 0)                  if 'duration_ms' in df_clean.columns else None,
    'popularity':         (lambda df: df['popularity'].between(0, 100))        if 'popularity' in df_clean.columns else None,
    'danceability':       (lambda df: df['danceability'].between(0.0, 1.0))    if 'danceability' in df_clean.columns else None,
    'energy':             (lambda df: df['energy'].between(0.0, 1.0))          if 'energy' in df_clean.columns else None,
    'loudness':           (lambda df: df['loudness'].between(-60, 0))          if 'loudness' in df_clean.columns else None,
    'speechiness':        (lambda df: df['speechiness'].between(0.0, 1.0))     if 'speechiness' in df_clean.columns else None,
    'acousticness':       (lambda df: df['acousticness'].between(0.0, 1.0))    if 'acousticness' in df_clean.columns else None,
    'instrumentalness':   (lambda df: df['instrumentalness'].between(0.0, 1.0)) if 'instrumentalness' in df_clean.columns else None,
    'liveness':           (lambda df: df['liveness'].between(0.0, 1.0))        if 'liveness' in df_clean.columns else None,
    'valence':            (lambda df: df['valence'].between(0.0, 1.0))         if 'valence' in df_clean.columns else None,
    'tempo':              (lambda df: df['tempo'] > 0)                         if 'tempo' in df_clean.columns else None,
    'key':                (lambda df: df['key'].between(0, 11))                if 'key' in df_clean.columns else None,
    'mode':               (lambda df: df['mode'].isin([0, 1]))                 if 'mode' in df_clean.columns else None,
    'time_signature':     (lambda df: df['time_signature'].between(1, 7))      if 'time_signature' in df_clean.columns else None,
}

mascara_valida = pd.Series(True, index=df_clean.index)
for col, regra in validacoes.items():
    if regra is not None:
        validos = regra(df_clean)
        invalidos = (~validos).sum()
        if invalidos > 0:
            print(f"  - '{col}': {invalidos} registros com valor inválido")
        mascara_valida &= validos

antes = len(df_clean)
df_clean = df_clean[mascara_valida].copy()
removidos_invalidos = antes - len(df_clean)
print(f"  Total de registros removidos por valores inválidos: {removidos_invalidos}")

# 3.7 Tratamento de outliers via clipagem IQR
print("\n[3.7] Clipagem de outliers (IQR):")
outlier_cols = [c for c in [
    'popularity', 'danceability', 'energy', 'loudness',
    'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo', 'duration_min'
] if c in df_clean.columns]

for col in outlier_cols:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()
    df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
    print(f"  - '{col}': {outliers} outliers clipados (limite [{lower:.4f}, {upper:.4f}])")

# 3.8 Normalização (Min-Max) das colunas numéricas principais
features_to_normalize = [c for c in [
    'popularity', 'danceability', 'energy', 'loudness',
    'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo', 'duration_min'
] if c in df_clean.columns]

scaler = MinMaxScaler()
df_normalized = df_clean.copy()
df_normalized[features_to_normalize] = scaler.fit_transform(df_clean[features_to_normalize])

print(f"\n[3.8] Normalização Min-Max aplicada nas colunas: {features_to_normalize}")

# ============================================================
# 4. ANÁLISE ESTATÍSTICA BÁSICA E CORRELAÇÃO
# ============================================================
print("\n" + "=" * 60)
print("4. ANÁLISE ESTATÍSTICA BÁSICA E CORRELAÇÃO")
print("=" * 60)

stats_cols = features_to_normalize
stats = df_clean[stats_cols].agg(['mean', 'median', 'std', 'min', 'max'])
stats.index = ['Média', 'Mediana', 'Desvio Padrão', 'Mínimo', 'Máximo']
print("\n--- Estatísticas descritivas (dados originais) ---")
print(stats.round(4).T.to_string())

# --- GRÁFICOS ---

# Gráfico 2: Distribuição das variáveis numéricas
n_cols_plot = min(len(features_to_normalize), 12)
cols_plot = features_to_normalize[:n_cols_plot]
n_rows = (n_cols_plot + 2) // 3

fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 3.5))
axes = axes.flatten()

for i, col in enumerate(cols_plot):
    axes[i].hist(df_clean[col].dropna(), bins=30, color='steelblue', edgecolor='white', alpha=0.85)
    axes[i].set_title(f'Distribuição: {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequência')
    mean_val = df_clean[col].mean()
    axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=1.2, label=f'Média: {mean_val:.2f}')
    axes[i].legend(fontsize=8)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Distribuição das Variáveis Numéricas', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('resultados/02_distribuicoes.png', bbox_inches='tight')
print("Gráfico salvo: 02_distribuicoes.png")

# Gráfico 3: Boxplots
fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 3.5))
axes = axes.flatten()

for i, col in enumerate(cols_plot):
    axes[i].boxplot(df_clean[col].dropna(), patch_artist=True,
                    boxprops=dict(facecolor='lightblue', color='navy'),
                    medianprops=dict(color='red', linewidth=2))
    axes[i].set_title(f'Boxplot: {col}')
    axes[i].set_ylabel(col)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Boxplots das Variáveis Numéricas', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('resultados/03_boxplots.png', bbox_inches='tight')
print("Gráfico salvo: 03_boxplots.png")

# Gráfico 4: Matriz de correlação
corr_matrix = df_clean[stats_cols].corr()

fig, ax = plt.subplots(figsize=(12, 9))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            mask=mask, ax=ax, linewidths=0.5,
            annot_kws={'size': 8}, vmin=-1, vmax=1)
ax.set_title('Matriz de Correlação entre Variáveis Numéricas', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('resultados/04_correlacao.png', bbox_inches='tight')
print("Gráfico salvo: 04_correlacao.png")

# Gráfico 5: Top correlações com 'popularity'
if 'popularity' in stats_cols:
    corr_pop = corr_matrix['popularity'].drop('popularity').sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['salmon' if v < 0 else 'steelblue' for v in corr_pop]
    corr_pop.plot(kind='barh', ax=ax, color=colors, edgecolor='black')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title('Correlação das Variáveis com Popularity', fontsize=12, fontweight='bold')
    ax.set_xlabel('Coeficiente de Correlação')
    plt.tight_layout()
    plt.savefig('resultados/05_correlacao_popularity.png', bbox_inches='tight')
    print("Gráfico salvo: 05_correlacao_popularity.png")

# Gráfico 6: Top 10 gêneros mais frequentes (se existir)
if 'track_genre' in df_clean.columns:
    top_genres = df_clean['track_genre'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    top_genres.plot(kind='bar', ax=ax, color='mediumpurple', edgecolor='black')
    ax.set_title('Top 10 Gêneros Musicais', fontsize=12, fontweight='bold')
    ax.set_xlabel('Gênero')
    ax.set_ylabel('Quantidade de Faixas')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('resultados/06_top_generos.png', bbox_inches='tight')
    print("Gráfico salvo: 06_top_generos.png")

# Gráfico 7: Scatter - Energy vs Popularity
if 'energy' in df_clean.columns and 'popularity' in df_clean.columns:
    fig, ax = plt.subplots(figsize=(8, 5))
    sample = df_clean.sample(min(2000, len(df_clean)), random_state=42)
    ax.scatter(sample['energy'], sample['popularity'], alpha=0.3, color='teal', edgecolors='none', s=15)
    ax.set_xlabel('Energy')
    ax.set_ylabel('Popularity')
    ax.set_title('Energy vs Popularity', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('resultados/07_energy_vs_popularity.png', bbox_inches='tight')
    print("Gráfico salvo: 07_energy_vs_popularity.png")

# Gráfico 8: Danceability vs Valence
if 'danceability' in df_clean.columns and 'valence' in df_clean.columns:
    fig, ax = plt.subplots(figsize=(8, 5))
    sample = df_clean.sample(min(2000, len(df_clean)), random_state=42)
    scatter = ax.scatter(sample['danceability'], sample['valence'],
                         c=sample['popularity'] if 'popularity' in sample.columns else 'blue',
                         cmap='viridis', alpha=0.4, s=15)
    plt.colorbar(scatter, ax=ax, label='Popularity')
    ax.set_xlabel('Danceability')
    ax.set_ylabel('Valence')
    ax.set_title('Danceability vs Valence (cor = Popularity)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('resultados/08_danceability_vs_valence.png', bbox_inches='tight')
    print("Gráfico salvo: 08_danceability_vs_valence.png")

# ============================================================
# 5. DATASET FINAL
# ============================================================
print("\n" + "=" * 60)
print("5. DATASET FINAL")
print("=" * 60)

print(f"\nShape final (dados limpos): {df_clean.shape}")
print(f"Shape final (dados normalizados): {df_normalized.shape}")

total_removidos = removidos_duplicatas + removidos_ausentes + removidos_popularity + removidos_invalidos
print(f"\n--- Totalização de registros removidos ---")
print(f"  Duplicatas:               {removidos_duplicatas}")
print(f"  Valores ausentes:         {removidos_ausentes}")
print(f"  Popularity = 0:           {removidos_popularity}")
print(f"  Valores inválidos:        {removidos_invalidos}")
print(f"  Total removido:           {total_removidos}")
print(f"  Registros iniciais:       {total_registros_inicial}")
print(f"  Registros finais:         {len(df_clean)}")

print("\n--- Amostra do dataset limpo ---")
print(df_clean.head())

print("\n--- Amostra do dataset normalizado ---")
print(df_normalized[features_to_normalize].head())

# Salvar datasets
df_clean.to_csv('spotify_clean.csv', index=False)
df_normalized.to_csv('spotify_normalized.csv', index=False)

print("\n✓ Dataset limpo salvo em: spotify_clean.csv")
print("✓ Dataset normalizado salvo em: spotify_normalized.csv")

print("\n" + "=" * 60)
print("RESUMO DAS TRANSFORMAÇÕES REALIZADAS")
print("=" * 60)
print("""
1. Remoção de duplicatas (critério: track_name + artists)
2. Tratamento de valores ausentes:
   - Registros com 3 ou mais valores ausentes: removidos
   - Numéricos (1 ou 2 ausentes): preenchidos com a mediana
   - Categóricos (1 ou 2 ausentes): preenchidos com a moda
3. Conversão de 'duration_ms' para 'duration_min' (minutos)
4. Remoção de colunas irrelevantes (Unnamed: 0, track_id)
5. Remoção de faixas com popularity = 0
6. Validação de intervalos conhecidos por coluna
7. Clipagem de outliers via IQR (limites Q1 - 1.5*IQR e Q3 + 1.5*IQR)
8. Normalização Min-Max das variáveis numéricas principais
""")
