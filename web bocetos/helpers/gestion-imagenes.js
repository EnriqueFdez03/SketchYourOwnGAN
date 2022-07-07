const path = require('path');
const fs = require('fs');  

const subirImagen = ( files ) => {
    return new Promise( (resolve, reject) =>{
        const { image } = files;
        d = new Date().getTime()

        img_path = path.join(__dirname, `../bocetos/${d}.jpg`)   
        image.mv(img_path, function(err) {
            if (err) {
                return reject(err);
            } 
            return resolve("ok");
        });
               
    });
}

module.exports = {
    subirImagen
}