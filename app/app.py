# =[Modules dan Packages]========================
import numpy as np
from flask import Flask, render_template, request, jsonify
# from flask_ngrok import run_with_ngrok  #Karena engga make ngrok diapus aja
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from joblib import load

# =[Variabel Global]=============================

app = Flask(__name__, static_url_path='/static')
model = None

# =[Routing]=====================================


# [Routing untuk Halaman Utama atau Home]
@app.route("/")
def beranda():
  return render_template('index.html')


# [Routing untuk API]
@app.route(
  "/api/prediksi", methods=['POST']
)  ##file api/prediksi liat dimana mat?## api prediksi itu path route nya pal bukan file
def apiPrediksi():
  # float_feature = [float(x) for x in request.form.values()]
  # feature = [np.array(float_feature)]
  # prediksi = model.prediksi(feature)
  # return render_template("test.html", prediksi_text="{}".format(prediksi))

  # POST data dari API
  if request.method == 'POST':
    # Set nilai untuk variabel input atau features (X) berdasarkan input dari pengguna
    # $("#range_sepal_length").val();

    input_Lokasi = str(request.form.get('lokasi'))
    input_Luas_Tanah = str(request.form.get('luas_tanah'))
    input_Luas_Bangunan = str(request.form.get('luas_bangunan'))
    input_KT = str(request.form.get('kamar_tidur'))
    input_KM = str(request.form.get('kamar_mandi'))
    input_Listrik = str(request.form.get('listrik'))
    input_Garasi = str(request.form.get('garasi'))

    print(f'test : {input_KT}')

    # Nilai default untuk variabel input atau features (X) ke model
    if input_Lokasi == "Lokasi":
      input_Lokasi = "3"
    if input_KT == "KT":
      input_KT = "2"
    if input_KM == "KM":
      input_KM = "1"
    if input_Listrik == "listrik":
      input_Listrik = "2200"
    if input_Garasi == "Garasi":
      input_Garasi = "0"

    # Load data ke dalam DataFrame
    data = pd.read_csv('data_rumah_jabodetabek.csv', sep=';')
    # Preprocessing data untuk model rekomendasi
    # Encoder lokasi
    data['lokasi'] = data['lokasi'].map({
      'Kota Jakarta': 0,
      'Kota Bogor': 1,
      'Kabupaten Bogor': 2,
      'Kota Depok': 3,
      'Kota Tangerang': 4,
      'Kota Bekasi': 5,
      'Kabupaten Bekasi': 6
    })

    # Encoder garasi/carport
    data['garasi_carport'] = data['garasi_carport'].map({
      'Ada': 0,
      'Tidak ada': 1
    })

    # Membuat dataframe pandas
    df = pd.DataFrame(
      data={
        "lokasi": [input_Lokasi],
        "LT": [input_Luas_Tanah],
        "LB": [input_Luas_Bangunan],
        "KT": [input_KT],
        "KM": [input_KM],
        "listrik": [input_Listrik],
        "garasi_carport": [input_Garasi],
      })

    #Mendapatkan rekomendasi rumah berdasarkan input pengguna
    X = data.drop('harga', axis=1)
    similarity_matrix = cosine_similarity(df, X)

    # Fungsi untuk mendapatkan 6 rekomendasi berdasarkan rumah yang dipilih
    def get_recommendations(similarity_matrix, data, n=6):
      rumah_indices = similarity_matrix.argsort()[0][
        -n:]  # Ambil indeks rumah yang paling mirip dengan input pengguna
      recommended_rumah = data.iloc[rumah_indices]
      return recommended_rumah

    rekomendasi = get_recommendations(similarity_matrix, data)

    print(f'test sistem rekomendasi :\n {rekomendasi}')
    print(f' Baris 1 Kolom lokasi : \n{rekomendasi.iloc[0,0]}')

    # array_rekomendasi = []
    # for i in range(len(rekomendasi)):
    #   print(rekomendasi.iloc[i])
    #   array_rekomendasi.append(rekomendasi.iloc[i])
    df_rekomendasi = rekomendasi.to_json(orient='records')

    print(df)
    print(df[0:1])

    #Menampilkan Prediksi model adaboost
    hasil_prediksi = model.predict(df[0:1])[0]

    print(f'Hasil prediksi belum dikonversi:{hasil_prediksi}')

    hasil_prediksi_conv = np.exp(hasil_prediksi)

    print(f'Hasil prediksi setelah dikonversi:{hasil_prediksi_conv}')

    hasil_prediksi_conv = hasil_prediksi_conv / 1000000
    hasil_prediksi_conv = int(hasil_prediksi_conv)
    hasil_prediksi_conv = hasil_prediksi_conv * 1000000
    hasil_prediksi_conv = format(hasil_prediksi_conv, ',d')

    hasil_prediksi_conv = f"Rp{hasil_prediksi_conv}"

    # Return hasil prediksi dengan format JSON
    return jsonify({
      "prediksi": hasil_prediksi_conv,
      "rekomendasi": df_rekomendasi
    })


# =[Main]========================================

if __name__ == '__main__':
  # Load model yang telah ditraining
  model = load('alaya.model')

  # Run Flask di Google Colab menggunakan ngrok
  # run_with_ngrok(
  #   app
  # )  # <-- Ngrok susah kalo di replit, jadi harus run ga pake ngrok terus host nya diganti "0.0.0.0"

  app.run(host="0.0.0.0", port=4000, debug=True)  # <-- Harusnya gini
  # app.run()
