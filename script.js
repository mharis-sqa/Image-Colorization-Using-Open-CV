// # This code is for my future plan about this project curently I failed to exeute it with User Interface



// document.getElementById('imageInput').addEventListener('change', handleImage);

// function handleImage() {
//     const input = document.getElementById('imageInput');
//     const originalImage = document.getElementById('originalImage');
//     originalImage.src = URL.createObjectURL(input.files[0]);
// }

// function colorizeImage() {
//     const originalImage = document.getElementById('originalImage');
//     const colorizedImage = document.getElementById('colorizedImage');
    
//     // Create a FormData object to send the image to the server
//     const formData = new FormData();
//     formData.append('image', originalImage.src);

//     // Send the image to the server using fetch
//     fetch('/colorize', {
//         method: 'POST',
//         body: formData
//     })
//     .then(response => response.json())
//     .then(data => {
//         colorizedImage.src = `data:image/png;base64, ${data.colorized}`;
//     })
//     .catch(error => console.error('Error:', error));
// }
