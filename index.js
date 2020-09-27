var fr; // Variable to store the file reader
var is_img_ready = false;
var name;
//Function to load the image from local path to img and canvas
function loadImage() {
    img_src = document.getElementById('img_src');
    if(!img_src.files[0]) {
        alert('Please select an Image first!')
        return;
    }
    var fsize=img_src.files[0].size;
    var size=Math.round((fsize / 1024));
    if(size>5000){
        alert('Please select an image less than 5mb')
        return;
    }
    fr = new FileReader();
    fr.onload = updateImage;
    fr.readAsDataURL(img_src.files[0])
    name=img_src.files[0].name
}

function updateImage() {
    img = new Image();

    img.onload = function() {
        var canvas = document.getElementById("local_canvas")
        canvas.width = img.width;
        canvas.height = img.height;
        var ctx = canvas.getContext("2d");
        ctx.drawImage(img,0,0);
        //alert(canvas.toDataURL("image/png"));
    };
    img.src = fr.result;
    is_img_ready = true;
}
function loadtext(face , emotions)
{
    if (face>0){
        var str1="Number of faces detected : "+face;
        var str2="Emotions detected are : "+emotions;
        document.getElementById("face_text").innerHTML = " "+str1;
        document.getElementById("emotion_text").innerHTML = " "+str2;

    }
    else{
        document.getElementById("face_text").innerHTML = "No face detected";
    }
}
function loadProcessedImage(data) {
    img = new Image();

    img.onload = function() {
        var processedCanvas = document.getElementById('processed_canvas');
        var localCanvas = document.getElementById('local_canvas');
        processedCanvas.width = localCanvas.width;
        processedCanvas.height = localCanvas.height;
        ctx = processedCanvas.getContext('2d');
        ctx.drawImage(img, 0, 0);
    };
    console.log(data);
    img.src = 'data:image/jpeg;base64,' + data;
}
function processImage() {
    if (is_img_ready == false) {
        alert('Please load image first!');
        return;
    }

    alert("Please wait Image is loading.")
    //Send the image to the server and wait for a response
    canvas = document.getElementById('local_canvas');
    image_data = canvas.toDataURL('image/jpeg');

    $.ajax({
        url:"http://127.0.0.1:5000/process_image",
        method: "POST",
        contentType: 'application/json',
        crossDomain: true,
        data: JSON.stringify({
            name:name,
            image_data: image_data,
            msg: 'This is image data'
        }),
        success: function(data){
            //loadProcessedImage(data);

            loadProcessedImage(data.image_data);
            loadtext(data.face,data.emotions);
        },
        error: function(err) {
            console.log(err)
        }
    });
}