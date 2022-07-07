const validarArchivoSubir = (req, res, next) => {

    if (!req.files || Object.keys(req.files).length === 0 || !req.files.image) {
        return res.status(400).json({msg: 'El post debe tener imagen'});
    }

    next();
}
 
module.exports = {
    validarArchivoSubir
}