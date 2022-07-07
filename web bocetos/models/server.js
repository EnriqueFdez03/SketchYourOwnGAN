const express = require('express');
const { createServer } = require("http");
const cors = require('cors');
const fileUpload = require('express-fileupload');


class Servidor {

    constructor() {
        this.app  = express();
        this.port = process.env.PORT;
        this.server = createServer(this.app);

        this.paths = {
            images:    '/api/images',
        }

        // Middlewares
        this.middlewares();
        // Rutas de mi aplicación
        this.routes();
    }

    middlewares() {
        // CORS
        this.app.use( cors() );
        // Lectura y parseo del body
        this.app.use( express.json() );
        // Directorio Público
        this.app.use( express.static('public') );
        
        //Fileupload - carga de archivos
        this.app.use(fileUpload({
            useTempFiles : true,
            tempFileDir : '/tmp/',
            createParentPath: true
        }));

    }

    routes() {
        this.app.use( this.paths.images, require('../routes/images'));
    }


    listen() {
        this.server.listen( this.port, () => {
            console.log('Servidor corriendo en puerto', this.port );
        });
    }

}

module.exports = {Servidor};
