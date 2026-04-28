# =============================================================================
# Função utilitária: normalização de tipos do SIH-SUS
#
# Razão: o PySUS converte DBC -> Parquet preservando tipos originais, e o DBC
# guarda muitos campos numéricos como strings de 1 char. Isso causa bugs
# silenciosos (string == int sempre False).
#
# Esta função converte explicitamente cada coluna para o tipo correto,
# tratando valores ausentes/inválidos de forma controlada.
# =============================================================================

import pandas as pd
import numpy as np


def normalizar_tipos_sih(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza tipos das colunas do SIH-SUS após leitura do parquet.
    
    Decisões metodológicas (todas conservadoras):
    - Valores inválidos em campos binários (MORTE) viram 0 (assume vivo).
    - Valores inválidos em IDADE viram NaN (preserva info de "desconhecido").
    - VAL_TOT inválido vira NaN (não inventa zero — falso barato é tão ruim
      quanto falso caro).
    - DT_INTER e DT_SAIDA viram datetime; inválidos viram NaT.
    
    Retorna DataFrame com tipos coerentes, sem alterar o original (cópia).
    """
    df = df.copy()
    
    # === Inteiros binários (0/1) ===
    # MORTE: óbito hospitalar. Conservador: NaN -> 0 (assume vivo)
    if "MORTE" in df.columns:
        df["MORTE"] = (
            pd.to_numeric(df["MORTE"], errors="coerce")
              .fillna(0)
              .astype("int8")
        )
    
    # === Inteiros pequenos ===
    if "SEXO" in df.columns:
        # SEXO: 1=Masculino, 3=Feminino, 0=Ignorado (padrão DataSUS)
        df["SEXO"] = pd.to_numeric(df["SEXO"], errors="coerce").astype("Int8")
    
    if "RACA_COR" in df.columns:
        # 1=Branca, 2=Preta, 3=Parda, 4=Amarela, 5=Indígena, 99=Sem info
        df["RACA_COR"] = pd.to_numeric(df["RACA_COR"], errors="coerce").astype("Int8")
    
    if "COD_IDADE" in df.columns:
        # 2=dias, 3=meses, 4=anos, 5=mais de 100 anos
        df["COD_IDADE"] = pd.to_numeric(df["COD_IDADE"], errors="coerce").astype("Int8")
    
    # === Inteiros maiores ===
    for col in ["IDADE", "DIAS_PERM", "UTI_MES_TO", "ANO_CMPT", "MES_CMPT"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int32")
    
    # === Floats (valores monetários) ===
    if "VAL_TOT" in df.columns:
        df["VAL_TOT"] = pd.to_numeric(df["VAL_TOT"], errors="coerce")
    
    # === Datas ===
    # DataSUS usa formato YYYYMMDD como string
    for col in ["DT_INTER", "DT_SAIDA"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="%Y%m%d", errors="coerce")
    
    # === Strings (mantém, mas garante tipo string) ===
    for col in ["DIAG_PRINC", "DIAG_SECUN", "MUNIC_RES", "UF_ZI", "N_AIH"]:
        if col in df.columns:
            df[col] = df[col].astype("string")
    
    return df


def relatorio_qualidade_tipos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gera relatório de qualidade pós-normalização.
    Para cada coluna importante: tipo, % de nulos, valores únicos (se ≤10).
    
    Use isso para auditar conversões — se uma coluna tem 50% de NaN após
    normalização, alguma coisa está errada (ou o dado original é ruim).
    """
    relatorio = []
    for col in df.columns:
        info = {
            "coluna": col,
            "dtype": str(df[col].dtype),
            "pct_nulo": round(100 * df[col].isna().mean(), 2),
            "n_unicos": df[col].nunique(dropna=True),
        }
        if df[col].nunique(dropna=True) <= 10:
            info["valores"] = sorted(df[col].dropna().unique().tolist())[:10]
        else:
            info["valores"] = "..."
        relatorio.append(info)
    return pd.DataFrame(relatorio)