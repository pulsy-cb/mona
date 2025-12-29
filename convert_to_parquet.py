import pyarrow.csv as pv
import pyarrow.parquet as pq
import time
import os

def convert_csv_to_parquet(csv_file, parquet_file):
    print(f"Conversion de {csv_file} en {parquet_file} via PyArrow...")
    start_time = time.time()
    
    try:
        # Lecture du CSV via PyArrow (beaucoup plus économe en RAM)
        table = pv.read_csv(csv_file)
        
        # Écriture en Parquet
        pq.write_table(table, parquet_file, compression='snappy')
        
        end_time = time.time()
        duration = end_time - start_time
        
        csv_size = os.path.getsize(csv_file) / (1024 * 1024)
        parquet_size = os.path.getsize(parquet_file) / (1024 * 1024)
        
        print(f"Terminé en {duration:.2f} secondes.")
        print(f"Taille CSV : {csv_size:.2f} Mo")
        print(f"Taille Parquet : {parquet_size:.2f} Mo")
        print(f"Gain d'espace : {((csv_size - parquet_size) / csv_size) * 100:.2f}%")
        
    except Exception as e:
        print(f"Erreur lors de la conversion : {e}")

if __name__ == "__main__":
    csv_input = "XAUUSD.csv"
    parquet_output = "XAUUSD.parquet"
    
    if os.path.exists(csv_input):
        convert_csv_to_parquet(csv_input, parquet_output)
    else:
        print(f"Erreur : Le fichier {csv_input} n'existe pas.")
