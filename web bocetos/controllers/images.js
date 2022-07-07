const { response } = require("express");
const { subirImagen, subirSketch, borrarImagenPath, getImagen } = require("../helpers/gestion-imagenes");
const path = require('path');

const cargarArchivo = async(req,res = response) => {
    subirImagen(req.files)
        .then(path => {
            res.status(201).json({
                path
            });
        })
        .catch(err => {
            res.status(400).json({
                err
            });
        });

}


module.exports = {
    cargarArchivo,
}