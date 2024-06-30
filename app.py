from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Memuat model yang sudah disimpan
model = joblib.load('random_forest_model_new.pkl')

# Data informasi dan gambar tanaman
plant_info = {
    0: {
        'name': 'Apel',
        'description': 'Apel merupakan tanaman buah tahunan yang berasal dari daerah Asia Barat dengan iklim subtropis. Tanaman Apel dapat tumbuh dan berbuah baik pada ketinggian (700-1200 mdpl) Dan dengan ketinggian Optimal (1.000-1.200 mdpl). Di Indonesia, apel telah ditanam sejak tahun 1934.',
        'image': 'static/images/Apel.jpg'
    },
    1: {
        'name': 'Pisang',
        'description': 'Pisang adalah kelompok tanaman herba yang berbuah dan diketahui berasal dari kawasan tropis. Tumbuh dengan baik mulai dari dataran rendah hingga ketinggian 1300 meter dari permukaan laut. Pisang dikenal sebagai buah yang tinggi serat. Kandungan serat di dalamnya dapat memberikan efek kenyang lebih lama dan menahan nafsu makan sehingga mengurangi keinginan untuk makan secara berlebihan.',
        'image': 'static/images/Pisang.jpeg'
    },
    2: {
        'name': 'Lentil Hitam',
        'description': 'Lentil hitam adalah tanaman herbal tahunan dengan daun berbulu dan polong tipis sepanjang 4-6 cm. Seperti kacang-kacangan lain, lentil hitam menyukai iklim sedang dengan kelembapan rendah. Tumbuhan ini terutama tersebar di dataran rendah tetapi dapat ditemukan hingga 1800 mdpl dan tumbuh paling baik selama musim kemarau',
        'image': 'static/images/Lentil hitam.jpg'
    },
    3: {
        'name': 'Kacang Arab',
        'description': 'Kacang arab tumbuh dengan baik di tanah kering dan membutuhkan sedikit air, sehingga dapat ditanam sebagai tanaman tadah hujan. Jika curah hujan tidak mencukupi, irigasi harus dilakukan sebelum bunga dan selama pengembangan polong.',
        'image': 'static/images/Kacang arab.jpg'
    },
    4: {
        'name': 'Kelapa',
        'description': 'Pohon kelapa adalah tanaman asli daerah tropis. Selain tumbuh liar, pohon kelapa juga tumbuh subur dengan melalui pembudidayaan. Tanaman kelapa tumbuh baik didaerah dataran rendah dengan Ketinggian yang optimal 0-450 m dpl.',
        'image': 'static/images/Kelapa.jpg'
    },
    5: {
        'name': 'Kopi',
        'description': 'Kopi idealnya ditanam di daerah pegunungan dengan ketinggian antara 900-1.500 mdpl, namun tidak menutup kemungkinan beberapa jenis kopi dapat tumbuh dengan baik dibawah ketinggian 900 mdpl.',
        'image': 'static/images/Kopi.jpg'
    },
    6: {
        'name': 'Kapas',
        'description': 'Tanaman kapas adalah sejenis semak, tanaman asli daerah tropis dan subtropis di Amerika, Australia, Afrika, dan India. Sebaiknya tanaman kapas ditanam di tanah datar, dan cocok pada ketinggian 10-150 mdpl. Tanaman kapas menyukai iklim hangat dan lembap dengan curah hujan sedang.',
        'image': 'static/images/Kapas.jpg'
    },
    7: {
        'name': 'Anggur',
        'description': 'Anggur adalah buah yang dihasilkan oleh tanaman perdu merambat. Buah ini umumnya bermanfaat sebagai bahan jus anggur, selai, minuman anggur, minyak biji anggur, kismis, atau konsumsi langsung. Anggur paling baik ditanam di daerah beriklim musim dingin ringan dan periode hangat yang panjang.',
        'image': 'static/images/Anggur.jpg'
    },
    8: {
        'name': 'Jute',
        'description': ' Serat jute bisa dimanfaatkan sebagai bahan pembuatan karung dan pembungkus. Serat ini juga digunakan sebagai bahan kerajinan industri tekstil, tali-temali, terpal, isolasi listrik, dan bahan pembuatan atap.Jute tumbuh dalam kondisi yang mirip dengan padi, dan tanaman ini paling cocok di daerah hangat yang memiliki musim hujan tahunan.',
        'image': 'static/images/Jute.jpg'
    },
    9: {
        'name': 'Kacang Merah',
        'description': 'Kacang merah terkenal juga sebagai red bean atau kidney bean menjadi salah satu jenis kacang yang mengandung karbohidrat dan serat yang tinggi. Kacang merah cocok ditanam di daerah yang mempunyai iklim basah dengan ketinggian yang bervariasi. Ketinggian tempat yang cocok adalah 1000-1500 mdpl.',
        'image': 'static/images/Kacang merah.jpg'
    },
    10: {
        'name': 'Lentil',
        'description': 'Lentil merupakan tanaman polong-polongan tertua di dunia yang dibudidayakan. Lebih menyukai iklim yang lebih sejuk tetapi tidak dapat tumbuh dengan baik di cuaca yang lebih dingin.',
        'image': 'static/images/Lentil.jpg'
    },
    11: {
        'name': 'Jagung',
        'description': 'Jagung adalah salah satu tanaman serealia penting di Indonesia, selain sebagai tanaman bahan pangan pokok pengganti beras dalam upaya diversifikasi pangan, jagung juga merupakan pakan ternak. Salah satu karakteristik tanaman jagung adalah mudah tumbuh pada berbagai jenis tanah dan memiliki kemampuan beradaptasi dengan baik, namun paling cocok pada wilayah bersuhu dan bercurah hujan sedang.',
        'image': 'static/images/Jagung.jpg'
    },
    12: {
        'name': 'Mangga',
        'description': 'Mangga merupakan salah satu jenis buah yang mempunyai sumber vitamin dan mineral yang banyak terdapat di Indonesia. Mangga tumbuh dengan baik di sebagian besar wilayah tropis dan sub-tropis tetapi sangat sensitif terhdap panas maupun dingin.',
        'image': 'static/images/Mangga.jpg'
    },
    13: {
        'name': 'Moth Bean',
        'description': 'Moth bean adalah tanaman tahunan yang tumbuh rendah dan dapat mengikat nitrogen, membentuk lapisan di tanah. Kacang ini tahan terhadap kekeringan, bergizi dan dapat dimakan segar atau sebagai kacang kering. Kacang ngengat juga dapat digunakan sebagai tanaman penutup tanah untuk menekan gulma dan memperbaiki kondisi tanah.',
        'image': 'static/images/Moth bean.jpg'
    },
    14: {
        'name': 'Kacang Hijau',
        'description': 'Kacang hijau adalah tanaman sejenis palawija yang dikenal luas di daerah tropis, produksi kacang hijau sebagai bahan olahan bahan pangan berprotein nabati tinggi dan berperan dalam menumbuh kembangkan industri kecil maupun menengah. Kacang hijau dapat tumbuh dan berproduksi dengan baik di daerah pesisir pantai sampai dataran tinggi dengan ketinggian 500 mdpl',
        'image': 'static/images/Kacang hijau.jpg'
    },
    15: {
        'name': 'Melon',
        'description': 'Melon berasal dari Afrika Selatan dan merupakan buah gurun yang mengandung 92% air.Sebagai tanaman musim panas, melon membutuhkan sinar matahari yang cukup dan cuaca kering. Tempat ideal untuk budidaya melon berada pada kisaran ketinggian 250-700 mdpl.',
        'image': 'static/images/Melon.jpg'
    },
    16: {
        'name': 'Orange/Jeruk',
        'description': 'Apel merupakan tanaman buah tahunan yang berasal dari daerah Asia Barat dengan iklim subtropis. Tanaman Apel dapat tumbuh dan berbuah baik pada ketinggian (700-1200 mdpl) Dan dengan ketinggian Optimal (1.000-1.200 mdpl). Di Indonesia, apel telah ditanam sejak tahun 1934.',
        'image': 'static/images/Orange.jpg'
    },
    17: {
        'name': 'Pepaya',
        'description': 'Pepaya adalah buah tropis yang kaya akan nutrisi seperti vitamin C. Budidaya pepaya cocok di daerah tropis dan sub-tropis di ketinggian 600 mdpl.',
        'image': 'static/images/Pepaya.jpg'
    },
    18: {
        'name': 'Kacang gude',
        'description': 'Kacang gude telah dibudidayakan selama ribuan tahun dan berfungsi sebagai sumber protein. Kacang gude tahan kekeringan dan dapat tumbuh di daerah dengan curah hujan tahunan kurang dari 650 mm.',
        'image': 'static/images/Kacang gude.jpg'
    },
    19: {
        'name': 'Delima/Pomegranate',
        'description': 'Delima adalah buah komersial yang dapat dikonsumsi secara langsung maupun diolah menjadi jus dan selai. Delima dapat tumbuh di daerah beriklim sedang, semi kering, dan subtropis.',
        'image': 'static/images/Delima.jpeg'
    },
    20: {
        'name': 'Padi',
        'description': 'Padi merupakan tanaman pangan berupa rumput berumpun yang berasal dari dua benua yaitu Asia dan Afrika Barat tropis dan subtropis. Padi sebagian besar ditanam sebagai tanaman musiman. Tanaman padi sangat cocok di daerah yang mempunyai iklim yang berhawa panas dan banyak mengandung uap air.',
        'image': 'static/images/Padi.jpg'
    },
    21: {
        'name': 'Semangka',
        'description': ' Semangka merupakan tanaman semusim yang masih mempunyai hubungan kekerabatan dengan melon.Tanaman semangka dapat tumbuh dengan baik di dataran rendah hingga dataran tinggi 0-550 meter diatas permukaan laut. Daerah yang berkapur dan mengandung banyak bahan organik (subur) dengan iklim yang relatif kering lebih disenangi.',
        'image': 'static/images/Semangka.jpeg'
    }
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_features = np.array([data['features']])
    prediction = model.predict(input_features)
    plant_id = prediction[0]

    return jsonify({
        'name': plant_info[plant_id]['name'],
        'description': plant_info[plant_id]['description'],
        'image': plant_info[plant_id]['image']
    })

if __name__ == '__main__':
    app.run(debug=True)
