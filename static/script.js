async function getRecommendation() {
    // Ambil nilai dari input fields
    const field1 = document.getElementById('field1').value;
    const field2 = document.getElementById('field2').value;
    const field3 = document.getElementById('field3').value;
    const field4 = document.getElementById('field4').value;
    const field5 = document.getElementById('field5').value;
    const field6 = document.getElementById('field6').value;
    const field7 = document.getElementById('field7').value;

    // Format data untuk dikirim ke API
    const data = {
        features: [field1, field2, field3, field4, field5, field6, field7]
    };

    try {
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        const result = await response.json();

        // Tampilkan hasil rekomendasi
        const recommendation = `Berdasarkan input Anda, kami merekomendasikan tanaman: ${result.name}`;
        const description = result.description;
        const image = result.image;
        
        document.getElementById('recommendationText').innerText = recommendation;
        document.getElementById('descriptionText').innerText = description;
        document.getElementById('plantImage').src = image;

        document.getElementById('inputForm').classList.add('hidden');
        document.getElementById('result').classList.remove('hidden');
    } catch (error) {
        console.error('Error:', error);
    }
}

function resetForm() {
    document.getElementById('inputForm').reset();
    document.getElementById('inputForm').classList.remove('hidden');
    document.getElementById('result').classList.add('hidden');
}