const { Router } = require('express');

const { cargarArchivo } = require('../controllers/images');
const { validarArchivoSubir } = require('../middlewares/validar-archivo');
const { validarCampos } = require('../middlewares/validar-campos');

const router = Router();

router.post( '/', [
    validarArchivoSubir,
    validarCampos
], cargarArchivo );

module.exports = router;