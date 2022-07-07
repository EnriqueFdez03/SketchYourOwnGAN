site = `${window.location.hostname}:${port}`
//image size que se mostrará como máximo
const canvasWidth = 512
const canvasHeight = 512
const desiredWidth = 256
const desiredHeight = 256

const canvas = document.getElementById("pizarra");
canvas.width = canvasWidth
canvas.height = canvasHeight
const ctx = canvas.getContext('2d');
ctx.fillStyle = "white";
ctx.fillRect(0, 0, canvas.width, canvas.height);
const memCanvas =  document.createElement('canvas');
const memCtx = memCanvas.getContext('2d');


// Última posición conocida
let pos = { x: 0, y: 0 };

// Listener asociados con la interacción con la pizarra.
// Para escritorio.
let eraser = false;

canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mousedown', setPosition);
canvas.addEventListener('mouseenter', setPosition);
canvas.addEventListener('mouseup', saveState);
canvas.addEventListener('touchend', saveState);

// Para táctil.
canvas.addEventListener('touchstart',(e) => {
    e.preventDefault();
    setPosition(e.touches[0]);
});

canvas.addEventListener('touchmove',(e) => {
    e.preventDefault();
    draw(e.touches[0],false);
});

$("#borrarbtn").on('click', () => {
    if (!eraser) {
        $("#pizarra").css('cursor', 'url("/icons/cursor-eraser.png") 6 6,auto');
    } else {
        $("#pizarra").css('cursor', 'url("/icons/cursor-pencil.png") 0 11.99,auto');
    }
    eraser = !eraser;
});

// Calcula la posición del ratón o pulsado (táctil) y la guarda en la variable global pos.
function setPosition(e) {
    const offsetParent = canvas.offsetParent;
    pos.x = e.pageX - offsetParent.offsetLeft;
    pos.y = e.pageY - offsetParent.offsetTop;
}

// Si se dan las condiciones, pinta en el lugar que marca la variable global pos. 
// Si desktop es true, tiene que darse que se esté pulsado el click izquierdo.
function draw(e, desktop = true) {
  // Botón izquierdo ha de ser pulsado si estamos en desktop
  
  if (canvas.classList.contains('disabled')) {return}
  if (e.buttons !== 1 && desktop) return;

  ctx.beginPath(); 
  const lineWidth = 6
  ctx.lineWidth = lineWidth;
  ctx.lineCap = 'round';
  let strokeStyle = '#000000';

  ctx.globalCompositeOperation = "source-over"; 

  if (eraser) {
    ctx.globalCompositeOperation = "destination-out";  
    strokeStyle = "rgb(255,255,255)";
    ctx.lineWidth = 10;
  }
  
  ctx.strokeStyle = strokeStyle;

  ctx.moveTo(pos.x, pos.y); // desde
  setPosition(e);
  ctx.lineTo(pos.x, pos.y); // hasta

  ctx.stroke(); // pintar
}

// almacenamos estado para recuperarlo tras resize
function saveState() {
    memCanvas.width = canvas.width;
    memCanvas.height = canvas.height;
    memCtx.drawImage(canvas, 0, 0);
}

function disable() {
    if (!canvas.classList.contains("disabled")) {
        canvas.classList.add("disabled");
        document.getElementById("clearbtn").disabled = true;
        document.getElementById("borrarbtn").disabled = true;
        document.getElementById("uploadsketch").disabled = true;
    }
}

function enable() { 
    canvas.classList.remove("disabled");
    document.getElementById("clearbtn").disabled = false;
    document.getElementById("borrarbtn").disabled = false;
    document.getElementById("uploadsketch").disabled = false;
}



const getSketch = async() => {
    // limpiamos la memoria
    memCtx.clearRect(0, 0, memCanvas.width, memCanvas.height);
    // Esta es la url de la imagen que se ha de enviar al servidor. Ahora bien, tenemos que
    // redimensionar el boceto al tamaño de la imagen original. Para ello...
    disable();
    let sketch = getDraw();
   
    img = new Image()
    img.onload = async() => {
        // canvas cuya única utilidad será la de poder obtener el boceto, con la resolución de la imagen
        // original.
        let canvas2 = document.createElement('canvas');
        canvas2.width = desiredWidth;
        canvas2.height = desiredHeight;
        
        canvas2.getContext('2d').drawImage(img, 0, 0,desiredWidth, desiredHeight);
        let imgUrl = canvas2.toDataURL();

        const url = `http://${site}/api/images/`;

        let formData = new FormData();

        await fetch(imgUrl)
        .then(res => res.blob())
        .then(blob => {
            const file = new File([blob], 'dot.png', blob);
            formData.append("image", file);
        });
        
        fetch(url,{
            method: 'POST',
            body: formData
        })
        .then( resp => {
            clear();
            canvas2.remove();
            enable();
        })
        .catch(err => {
            clear()
            canvas2.remove();
            console.log(err.message);
            alert("algo fue mal");
            enable();
        }); 
    }
    img.src = sketch;
 
}

const clear = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

const getDraw = () => {
    let urlImg = canvas.toDataURL();
    return urlImg;
}

$("#clearbtn").click(clear);
$("#uploadsketch").click(getSketch);
