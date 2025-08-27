# Nama file: prepare_data.py

input_file = "data/ind.txt"
output_en_file = "data/en_corpus.txt"
output_id_file = "data/id_corpus.txt"

# Buka file output untuk ditulis
with open(output_en_file, "w", encoding="utf-8") as f_en, \
     open(output_id_file, "w", encoding="utf-8") as f_id:

    # Buka file input untuk dibaca
    with open(input_file, "r", encoding="utf-8") as f_in:
        for line in f_in:
            # Kolom dipisahkan oleh karakter tab (\t)
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                en_sentence = parts[0]
                id_sentence = parts[1]

                # Tulis kalimat ke file yang sesuai
                f_en.write(en_sentence + "\n")
                f_id.write(id_sentence + "\n")

print(f"Data berhasil dipisah ke '{output_en_file}' dan '{output_id_file}'")